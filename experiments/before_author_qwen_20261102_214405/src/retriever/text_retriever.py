from __future__ import annotations

import json
import os
import pickle
from typing import Iterable

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class TextRetriever:
    def __init__(self, model_name: str, device: str = "cuda:0") -> None:
        self.model_name = model_name
        self.tokenizer, self.model, self.device = self.load_model(model_name, device)

    @staticmethod
    def load_model(model_name: str, device: str = "cuda:0"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        torch_device = torch.device(device)
        model.to(torch_device)
        return tokenizer, model, torch_device

    def get_embedding(self, texts: list[str], batch_size: int = 1024) -> np.ndarray:
        if not texts:
            hidden_size = getattr(self.model.config, "hidden_size", 1024)
            return np.zeros((0, hidden_size), dtype=np.float32)

        tokenized_batches = []
        for i in range(0, len(texts), batch_size):
            tokenized_batches.append(
                self.tokenizer(
                    texts[i:i + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
            )

        all_embeddings = []
        with torch.inference_mode():
            for batch in tqdm(tokenized_batches, desc="Encoding texts"):
                encoded_input = {k: v.to(self.device) for k, v in batch.items()}
                model_output = self.model(**encoded_input)
                embeddings = model_output.last_hidden_state[:, 0, :]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0).cpu().numpy().astype(np.float32)

    def get_single_embedding(self, text: str | list[str]) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        return self.get_embedding(text, batch_size=1)

    @staticmethod
    def build_faiss_index_from_embeddings(embeddings: np.ndarray):
        embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    def create_faiss_index(self, facts: list[str]):
        fact_embeddings = self.get_embedding(facts)
        return self.build_faiss_index_from_embeddings(fact_embeddings)

    def build_or_load_kg_index(
        self,
        kg_path: str,
        cache_path: str,
        force_reload: bool = False,
    ):
        if os.path.exists(cache_path) and not force_reload:
            print(f"[Step] Load cached text retrieval index: {cache_path}")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data["facts"], data["index"]

        print("[Step] Build text retrieval index from KG")
        triple_list = []
        with open(kg_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().replace("_", " ").split("\t")
                if parts and len(parts) >= 4:
                    triple_list.append(parts)

        if not triple_list:
            raise ValueError(f"No facts found in KG file: {kg_path}")

        if len(triple_list[0]) == 4:
            facts = [f"{f[0]} {f[1]} {f[2]} in {f[3]}." for f in triple_list]
        else:
            facts = [f"{f[0]}, {f[1]}, {f[2]} from {f[3]} to {f[4]}." for f in triple_list]

        index = self.create_faiss_index(facts)
        with open(cache_path, "wb") as f:
            pickle.dump({"facts": facts, "index": index}, f)
        return facts, index

    def search_similar_facts(self, query: str, index, facts: list[str], top_k: int = 50):
        query_emb = self.get_embedding([query], batch_size=1)
        scores, indices = index.search(query_emb, top_k)
        return [(facts[i], float(scores[0][j])) for j, i in enumerate(indices[0])]

    def search_similar_facts_batch(
        self,
        queries: list[str],
        index,
        facts: list[str],
        top_k: int = 50,
        batch_size: int = 128,
    ):
        query_embeddings = self.get_embedding(queries, batch_size=512)
        all_results = []

        for start in tqdm(range(0, len(query_embeddings), batch_size), desc="Searching similar facts"):
            end = min(start + batch_size, len(query_embeddings))
            batch_embeddings = query_embeddings[start:end]
            scores, indices = index.search(batch_embeddings, top_k)

            for i in range(end - start):
                all_results.append(
                    [(facts[idx], float(scores[i][j])) for j, idx in enumerate(indices[i])]
                )
        return all_results
