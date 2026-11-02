from __future__ import annotations

import argparse
import asyncio
import os
import pickle

from config import DatasetConfig, LLMConfig, ModelConfig, ensure_parent_dir
from retriever.graph_retriever import GraphRetriever
from retriever.text_retriever import TextRetriever
from utils.client import AsyncLLMClient
from utils.eval_utils import evaluate_qa, print_qa_metrics
from utils.io_utils import load_qas, maybe_subsample
from utils.query_process import batch_get_query_graph_patterns, batch_run_qa
from utils.sample_utils import uniform_sample_preserve_edges


def build_graph_retriever(args) -> GraphRetriever:
    if os.path.exists(args.graph_cache_path) and not args.force_reload:
        print(f"[Step] Load cached graph retriever: {args.graph_cache_path}")
        with open(args.graph_cache_path, "rb") as f:
            graph_retriever = pickle.load(f)
        graph_retriever.rebuild_runtime(args.text_retriever_model_name, args.device)
        return graph_retriever

    print("[Step] Build graph retriever from KG/entity/relation files")
    graph_retriever = GraphRetriever(
        entity_file=args.entity_path,
        relation_file=args.relation_path,
        graph_file=args.kg_path,
        text_retriever_model_name=args.text_retriever_model_name,
        device=args.device,
        dataset_name=args.dataset_name,
    )
    ensure_parent_dir(args.graph_cache_path)
    with open(args.graph_cache_path, "wb") as f:
        pickle.dump(graph_retriever, f)
    print(f"[Step] Graph retriever cached to: {args.graph_cache_path}")
    return graph_retriever


def run_text_retrieval(queries, answers, args):
    print("\n========== Step 1: Text retrieval ==========")
    text_retriever = TextRetriever(args.text_retriever_model_name, device=args.device)
    ensure_parent_dir(args.text_index_path)
    facts, text_index = text_retriever.build_or_load_kg_index(
        kg_path=args.kg_path,
        cache_path=args.text_index_path,
        force_reload=args.force_reload,
    )
    text_results = text_retriever.search_similar_facts_batch(
        queries=queries,
        index=text_index,
        facts=facts,
        top_k=args.text_top_k,
        batch_size=128,
    )
    text_results_text = [[fact for fact, _ in result] for result in text_results]
    return text_results_text


async def run_graph_retrieval(queries, answers, text_results_text, args, llm_client):
    print("\n========== Step 2: Graph pattern extraction + graph retrieval ==========")
    graph_retriever = build_graph_retriever(args)

    print("[Step] Extract graph patterns from queries and retrieved text facts")
    graph_patterns = await batch_get_query_graph_patterns(
        queries=queries,
        all_results=text_results_text,
        prompt_path=args.graph_pattern_prompt,
        llm_client=llm_client,
        qa_retrieve_cnt=args.qa_retrieve_cnt,
        concurrency=args.concurrency,
    )

    print("[Step] Retrieve graph facts with parsed graph patterns")
    graph_results_text = []
    filter_texts = []
    for fact_pattern, time_pattern in graph_patterns:
        graph_facts = graph_retriever.get_facts(fact_pattern, time_pattern)
        graph_text = graph_retriever.facts_to_text(graph_facts)
        graph_results_text.append(graph_text)
        filter_texts.append(" ".join(fact_pattern) + f", from {time_pattern[0]} to {time_pattern[1]}")

    return graph_patterns, graph_results_text, filter_texts


async def run_graph_qa(queries, answers, qtypes, atypes, graph_patterns, graph_results_text, text_results_text, args, llm_client):
    print("\n========== Step 3: QA over graph retrieval results ==========")
    qa_inputs = []
    filter_texts = []

    for (fact_pattern, time_pattern), graph_text, text_text in zip(graph_patterns, graph_results_text, text_results_text):
        filter_text = " ".join(fact_pattern) + f", from {time_pattern[0]} to {time_pattern[1]}"
        filter_texts.append(filter_text)

        if len(graph_text) == 0:
            qa_inputs.append(text_text[: args.qa_retrieve_cnt])
        elif len(graph_text) >= args.qa_retrieve_cnt:
            qa_inputs.append(uniform_sample_preserve_edges(graph_text, sample_size=args.qa_retrieve_cnt))
        else:
            qa_inputs.append(graph_text)

    predictions, _ = await batch_run_qa(
        queries=queries,
        facts_list=qa_inputs,
        prompt_path=args.qa_prompt,
        llm_client=llm_client,
        concurrency=args.concurrency,
        filters=filter_texts,
    )
    metrics = evaluate_qa(answers, predictions, qtypes=qtypes, atypes=atypes)
    print_qa_metrics("Graph QA", metrics)
    return predictions, metrics


async def main(args):
    print("\n========== Load dataset ==========")
    queries, answers, qtypes, atypes = load_qas(args.qa_path)
    queries, answers, qtypes, atypes = maybe_subsample(
        queries, answers, qtypes, atypes, args.random_seed, args.sample_size
    )
    print(f"[Info] Loaded {len(queries)} questions from {args.qa_path}")

    llm_client = None
    if args.run_graph_retrieval or args.run_graph_qa:
        llm_client = AsyncLLMClient(
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.llm_model_name,
            cache_path=args.cache_path,
        )

    text_results_text = None
    graph_patterns = None
    graph_results_text = None

    if args.run_text_retrieval or args.run_graph_retrieval or args.run_graph_qa:
        text_results_text = run_text_retrieval(queries, answers, args)

    if args.run_graph_retrieval or args.run_graph_qa:
        graph_patterns, graph_results_text, _ = await run_graph_retrieval(
            queries, answers, text_results_text, args, llm_client
        )

    if args.run_graph_qa:
        await run_graph_qa(
            queries,
            answers,
            qtypes,
            atypes,
            graph_patterns,
            graph_results_text,
            text_results_text,
            args,
            llm_client,
        )

    print("\n========== Done ==========")


if __name__ == "__main__":
    dataset = DatasetConfig()
    model = ModelConfig()
    llm = LLMConfig()

    parser = argparse.ArgumentParser(description="Run text retrieval, graph retrieval, and QA evaluation.")
    parser.add_argument("--kg-path", default=dataset.kg_path)
    parser.add_argument("--qa-path", default=dataset.qa_path)
    parser.add_argument("--entity-path", default=dataset.entity_path)
    parser.add_argument("--relation-path", default=dataset.relation_path)
    parser.add_argument("--text-index-path", default=dataset.text_index_path)
    parser.add_argument("--graph-cache-path", default=dataset.graph_cache_path)
    parser.add_argument("--graph-pattern-prompt", default=dataset.graph_pattern_prompt)
    parser.add_argument("--qa-prompt", default=dataset.qa_prompt)
    parser.add_argument("--text-retriever-model-name", default=model.text_retriever_model_name)
    parser.add_argument("--device", default=model.device)
    parser.add_argument("--text-top-k", type=int, default=model.text_top_k)
    parser.add_argument("--qa-retrieve-cnt", type=int, default=model.qa_retrieve_cnt)
    parser.add_argument("--concurrency", type=int, default=model.concurrency)
    parser.add_argument("--random-seed", type=int, default=model.random_seed)
    parser.add_argument("--sample-size", type=int, default=model.sample_size)
    parser.add_argument("--api-key", default=llm.api_key)
    parser.add_argument("--base-url", default=llm.base_url)
    parser.add_argument("--llm-model-name", default=llm.model_name)
    parser.add_argument("--cache-path", default=llm.cache_path)
    parser.add_argument("--dataset-name", default=dataset.name)
    parser.add_argument("--force-reload", action="store_true")

    parser.add_argument("--run-text-retrieval", action="store_true", help="Run text retrieval and report Hits@N.")
    parser.add_argument("--run-graph-retrieval", action="store_true", help="Run graph pattern parsing and graph retrieval.")
    parser.add_argument("--run-graph-qa", action="store_true", help="Run QA using graph retrieval results.")
    args = parser.parse_args()

    if not (args.run_text_retrieval or args.run_graph_retrieval or args.run_graph_qa):
        args.run_text_retrieval = True
        args.run_graph_retrieval = True

    asyncio.run(main(args))
