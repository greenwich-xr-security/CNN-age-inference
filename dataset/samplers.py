"""Batch samplers that group samples by user."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

from torch.utils.data import Sampler


def _build_user_index_map(user_ids: Sequence[object]) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = defaultdict(list)
    for idx, uid in enumerate(user_ids):
        mapping[str(uid)].append(idx)
    return dict(mapping)


class GroupedBatchSampler(Sampler[list[int]]):
    """Yield batches that contain group_size samples per user."""

    def __init__(
        self,
        user_ids: Sequence[object],
        *,
        batch_size: int,
        group_size: int = 2,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive (got {batch_size}).")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive (got {group_size}).")
        if batch_size % group_size != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by group_size ({group_size})."
            )
        self.batch_size = batch_size
        self.group_size = group_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.users_per_batch = batch_size // group_size
        self.epoch = 0
        self._user_to_indices = _build_user_index_map(user_ids)
        self._chunk_count = sum(
            int(math.ceil(len(indices) / group_size)) for indices in self._user_to_indices.values()
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_chunks(self, rng: random.Random) -> List[List[int]]:
        chunks: List[List[int]] = []
        for indices in self._user_to_indices.values():
            if not indices:
                continue
            idxs = list(indices)
            if self.shuffle:
                rng.shuffle(idxs)
            for start in range(0, len(idxs), self.group_size):
                chunk = idxs[start : start + self.group_size]
                if len(chunk) < self.group_size:
                    # Pad with random samples from the same user to keep group_size.
                    while len(chunk) < self.group_size:
                        chunk.append(rng.choice(idxs))
                chunks.append(chunk)
        return chunks

    def __iter__(self) -> Iterable[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        chunks = self._build_chunks(rng)
        if not chunks:
            return
        if self.shuffle:
            rng.shuffle(chunks)
        batch: List[int] = []
        for chunk in chunks:
            batch.extend(chunk)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self._chunk_count == 0:
            return 0
        if self.drop_last:
            return self._chunk_count // self.users_per_batch
        return int(math.ceil(self._chunk_count / self.users_per_batch))


class DistributedGroupedBatchSampler(GroupedBatchSampler):
    """Distributed variant that splits grouped batches across ranks."""

    def __init__(
        self,
        user_ids: Sequence[object],
        *,
        batch_size: int,
        group_size: int = 2,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_replicas: int,
        rank: int,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive (got {num_replicas}).")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}) (got {rank}).")
        super().__init__(
            user_ids,
            batch_size=batch_size,
            group_size=group_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterable[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        chunks = self._build_chunks(rng)
        if not chunks:
            return
        if self.shuffle:
            rng.shuffle(chunks)

        required_multiple = self.num_replicas * self.users_per_batch
        total_chunks = len(chunks)
        if self.drop_last:
            total_chunks = total_chunks - (total_chunks % required_multiple)
            chunks = chunks[:total_chunks]
        else:
            target = int(math.ceil(total_chunks / required_multiple)) * required_multiple
            while len(chunks) < target:
                chunks.append(rng.choice(chunks))
            total_chunks = len(chunks)

        if total_chunks == 0:
            return

        chunks_rank = chunks[self.rank:total_chunks:self.num_replicas]
        batch: List[int] = []
        for chunk in chunks_rank:
            batch.extend(chunk)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self._chunk_count == 0:
            return 0
        required_multiple = self.num_replicas * self.users_per_batch
        if self.drop_last:
            total_chunks = self._chunk_count - (self._chunk_count % required_multiple)
        else:
            total_chunks = int(math.ceil(self._chunk_count / required_multiple)) * required_multiple
        if total_chunks == 0:
            return 0
        chunks_per_rank = total_chunks // self.num_replicas
        return chunks_per_rank // self.users_per_batch
