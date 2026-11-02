from __future__ import annotations

import math


def uniform_sample_preserve_edges(items: list[str], sample_size: int = 50) -> list[str]:
    sample_size = max(4, sample_size)
    n = len(items)
    if n <= sample_size:
        return items.copy()

    required_indices = [0, 1, n - 2, n - 1]
    middle_sample_count = sample_size - 4
    middle_start = 2
    middle_end = n - 3
    middle_len = middle_end - middle_start + 1

    middle_indices = []
    if middle_sample_count > 0 and middle_len > 0:
        for i in range(middle_sample_count):
            if middle_sample_count > 1:
                pos = math.floor(middle_len * i / (middle_sample_count - 1))
            else:
                pos = 0
            middle_indices.append(middle_start + pos)

    all_indices = sorted(required_indices + middle_indices)
    unique_indices = []
    seen = set()
    for idx in all_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    return [items[idx] for idx in unique_indices]
