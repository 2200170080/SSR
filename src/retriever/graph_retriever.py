from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
from tqdm import tqdm

from retriever.text_retriever import TextRetriever


class GraphRetriever:
    def __init__(
        self,
        entity_file: str,
        relation_file: str,
        graph_file: str,
        text_retriever_model_name: str,
        device: str = "cuda:0",
        dataset_name: str | None = None,
    ) -> None:
        self.entity = set()
        self.relation = set()
        self.entity_list: list[str] = []
        self.relation_list: list[str] = []
        self.graphdict = defaultdict(list)
        self.text_retriever_model_name = text_retriever_model_name
        self.device = device
        self.dataset_name = dataset_name

        self.text_retriever = None
        self.entity_index = None
        self.relation_index = None
        self.entity_embeddings = None
        self.relation_embeddings = None

        self.load_dict_file(entity_file, relation_file)
        self.init_text_retriever(device=device)
        self.load_graph_file(graph_file)

    def load_dict_file(self, entity_file: str, relation_file: str) -> None:
        with open(entity_file, "r") as f:
            entity = json.load(f)
        with open(relation_file, "r") as f:
            relation = json.load(f)
        self.entity_list = [x.replace("_", " ") for x in entity.keys()]
        self.relation_list = [x.replace("_", " ") for x in relation.keys()]
        self.entity = set(self.entity_list)
        self.relation = set(self.relation_list)

    def init_text_retriever(self, device: str) -> None:
        self.device = device
        self.text_retriever = TextRetriever(self.text_retriever_model_name, device=device)

        if os.path.exists(self._emb_path("entity")):
            self.entity_embeddings = np.load(self._emb_path("entity"))
        else:
            self.entity_embeddings = self.text_retriever.get_embedding(self.entity_list)
            np.save(self._emb_path("entity"), self.entity_embeddings)

        if os.path.exists(self._emb_path("relation")):
            self.relation_embeddings = np.load(self._emb_path("relation"))
        else:
            self.relation_embeddings = self.text_retriever.get_embedding(self.relation_list)
            np.save(self._emb_path("relation"), self.relation_embeddings)

        self.entity_index = self.text_retriever.build_faiss_index_from_embeddings(self.entity_embeddings)
        self.relation_index = self.text_retriever.build_faiss_index_from_embeddings(self.relation_embeddings)

    def search_similar_entity(self, query: str) -> str:
        query_emb = self.text_retriever.get_single_embedding(query)
        _, indices = self.entity_index.search(query_emb, 1)
        return self.entity_list[indices[0][0]]

    def search_similar_relation(self, query: str) -> str:
        query_emb = self.text_retriever.get_single_embedding(query)
        _, indices = self.relation_index.search(query_emb, 1)
        return self.relation_list[indices[0][0]]

    def load_graph_file(self, graph_file: str) -> None:
        with open(graph_file, "r") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Loading graph facts"):
                parts = line.strip().split("\t")
                if len(parts) == 4:
                    h, r, t, fact_time = parts
                else:
                    h, r, t, time1, time2 = parts
                    fact_time = (time1, time2)
                h = h.replace("_", " ").strip()
                r = r.replace("_", " ").strip()
                t = t.replace("_", " ").strip()
                self.graphdict[(h, r, "?")].append((h, r, t, fact_time))
                self.graphdict[("?", r, t)].append((h, r, t, fact_time))
                self.graphdict[(h, "?", t)].append((h, r, t, fact_time))
                self.graphdict[(h, r, t)].append((h, r, t, fact_time))
        for key in tqdm(self.graphdict.keys(), desc="Sorting graph facts"):
            self.graphdict[key].sort(key=self._sort_time_key)

    @staticmethod
    def _parse_time(raw_time: str) -> datetime:
        raw_time = raw_time.strip()
        if raw_time == "?":
            return datetime.max
        if raw_time.isdigit():
            year = int(raw_time)
            if year <= 0:
                year = 1
                # raise ValueError(f"Year must be >= 0: {raw_time}")
            return datetime(year, 1, 1, 0, 0, 0)
        if "-" in raw_time and len(raw_time.split("-")) == 3 and " " not in raw_time:
            try:
                y, m, d = raw_time.split("-")
                if int(y) <= 0:
                    y = 1
                return datetime(int(y), int(m), int(d), 0, 0, 0)
            except ValueError:
                return None
        try:
            return datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

    def _sort_time_key(self, fact) -> datetime:
        fact_time = fact[-1]
        if isinstance(fact_time, tuple):
            start_time = self._parse_time(fact_time[0])
            end_time = self._parse_time(fact_time[1])
        else:
            start_time = self._parse_time(fact_time)
            end_time = start_time
        known = [t for t in (start_time, end_time) if t is not None]
        return min(known) if known else datetime.max

    def _fact_matches_time(self, fact, start_time: str, end_time: str) -> bool:
        fact_time = fact[-1]
        if isinstance(fact_time, tuple):
            fact_start = self._parse_time(fact_time[0])
            fact_end = self._parse_time(fact_time[1])
        else:
            fact_start = self._parse_time(fact_time)
            fact_end = fact_start

        if fact_start is None or fact_end is None:
            return True

        if start_time != "?":
            query_start = self._parse_time(start_time)
            if query_start is not None and fact_end < query_start:
                return False
        if end_time != "?":
            query_end = self._parse_time(end_time)
            if query_end is not None and fact_start > query_end:
                return False
        return True

    def get_facts(self, fact_pattern, time_pattern=("?", "?")):
        h, r, t = fact_pattern
        start_time, end_time = time_pattern

        if h != "?" and h not in self.entity:
            print(f"[GraphRetriever] Normalize head entity: {h} -> ", end="")
            h = self.search_similar_entity(h)
            print(h)
        if r != "?" and r not in self.relation:
            print(f"[GraphRetriever] Normalize relation: {r} -> ", end="")
            r = self.search_similar_relation(r)
            print(r)
        if t != "?" and t not in self.entity:
            print(f"[GraphRetriever] Normalize tail entity: {t} -> ", end="")
            t = self.search_similar_entity(t)
            print(t)

        candidate_facts = self.graphdict.get((h, r, t), [])
        if start_time == "?" and end_time == "?":
            return candidate_facts
        return [fact for fact in candidate_facts if self._fact_matches_time(fact, start_time, end_time)]

    @staticmethod
    def facts_to_text(facts) -> list[str]:
        if not facts:
            return []
        fact_text = []
        for fact in facts:
            if isinstance(fact[-1], tuple):
                fact_text.append(f"{fact[0]}, {fact[1]}, {fact[2]} from {fact[3][0]} to {fact[3][1]}")
            else:
                fact_text.append(f"{fact[0]}, {fact[1]}, {fact[2]} in {fact[3]}")
        return fact_text

    def __getstate__(self):
        state = self.__dict__.copy()
        state["text_retriever"] = None
        state["entity_index"] = None
        state["relation_index"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def rebuild_runtime(self, text_retriever_model_name: str, device: str) -> None:
        self.text_retriever_model_name = text_retriever_model_name
        self.init_text_retriever(device)

    def _emb_path(self, name: str) -> str:
        prefix = self.dataset_name or "dataset"
        folder = os.path.join("artifacts", "shared", "indexes")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{prefix}_{name}_embeddings.npy")
