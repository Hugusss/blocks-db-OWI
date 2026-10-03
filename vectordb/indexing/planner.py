"""Build plan for a parquet source: which rows each block indexes.

Pure: it turns the footers of the source files (:class:`FileInfo`) into
exactly ``num_index`` non-empty, contiguous blocks with dense positional
ids, before any function runs.

Row groups are the unit of work. When the files hold at least as many
row groups as blocks, every block is a run of whole row groups, chosen
so that block sizes are as even as the row-group edges allow. When they
hold fewer, row groups are sliced at exact row boundaries instead, and a
function decodes a whole row group to keep its part of it.

Ids are ``id_offset + position`` for every row in file order across the
sorted files, so they are unique and reproducible; rejected rows leave
gaps that the workers count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from vectordb.errors import BlocksDBError
from vectordb.utils.parquet import FileInfo


class PlanError(BlocksDBError, ValueError):
    """A build that must not start; the message names the numbers."""


@dataclass(frozen=True)
class Range:
    """Rows ``[start, end)`` of one row group, with the id of row ``start``."""

    uri: str
    row_group: int
    start: int
    end: int
    id_offset: int

    @property
    def rows(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class BlockPlan:
    block: int
    ranges: tuple[Range, ...]

    @property
    def rows(self) -> int:
        return sum(part.rows for part in self.ranges)

    @property
    def first_id(self) -> int:
        return self.ranges[0].id_offset

    @property
    def last_id(self) -> int:
        last = self.ranges[-1]
        return last.id_offset + last.rows - 1


@dataclass(frozen=True)
class Plan:
    blocks: tuple[BlockPlan, ...]
    total_vectors: int
    dimension: int
    dialect: str
    sources: tuple[str, ...]

    @property
    def min_block_rows(self) -> int:
        return min(block.rows for block in self.blocks)

    @property
    def max_block_rows(self) -> int:
        return max(block.rows for block in self.blocks)

    @property
    def imbalance(self) -> float:
        """Largest block over smallest: how uneven the row-group edges
        made the split."""
        return self.max_block_rows / self.min_block_rows

    @property
    def num_index(self) -> int:
        return len(self.blocks)

    def with_sources(self, mapping: dict) -> "Plan":
        """The same plan with every local path replaced by the URI it was
        uploaded to, so the workers read what the client stored."""
        def moved(uri: str) -> str:
            return mapping.get(uri, uri)

        blocks = tuple(
            BlockPlan(
                block=block.block,
                ranges=tuple(
                    Range(
                        uri=moved(part.uri),
                        row_group=part.row_group,
                        start=part.start,
                        end=part.end,
                        id_offset=part.id_offset,
                    )
                    for part in block.ranges
                ),
            )
            for block in self.blocks
        )
        return Plan(
            blocks=blocks,
            total_vectors=self.total_vectors,
            dimension=self.dimension,
            dialect=self.dialect,
            sources=tuple(moved(uri) for uri in self.sources),
        )


def largest_nlist(rows: int) -> int:
    """The largest IVF list count FAISS trains without a warning on
    ``rows`` vectors (it wants at least 39 training points per list)."""
    return max(1, rows // 39)


def suggested_nlist(rows: int) -> int:
    """Four times the square root of ``rows``, the low end of FAISS's
    guidance, capped at :func:`largest_nlist`."""
    return min(largest_nlist(rows), max(1, round(4 * math.sqrt(rows))))


def plan(
    files: Sequence[FileInfo],
    num_index: int,
    *,
    k: int | None = None,
    features: int | None = None,
) -> Plan:
    """Plan exactly ``num_index`` blocks over ``files`` in sorted path order."""
    if not files:
        raise PlanError("no source files")
    if num_index < 1:
        raise PlanError(f"num_index must be at least 1, got {num_index}")
    ordered = sorted(files, key=lambda info: info.uri)
    _check_shape(ordered, features)

    groups = _row_groups(ordered)  # (uri, row_group, rows, id_offset) in order
    total = sum(rows for _, _, rows, _ in groups)
    if num_index > total:
        raise PlanError(
            f"num_index {num_index} exceeds the {total} rows of the source:"
            " every block must hold at least one vector"
        )
    if len(groups) >= num_index:
        blocks = _whole_groups(groups, num_index, total)
    else:
        blocks = _sliced(groups, num_index, total)

    result = Plan(
        blocks=tuple(blocks),
        total_vectors=total,
        dimension=ordered[0].dimension,
        dialect=ordered[0].dialect,
        sources=tuple(info.uri for info in ordered),
    )
    _check_invariants(result, num_index, total)
    if k is not None and k > result.min_block_rows:
        smallest = min(result.blocks, key=lambda block: block.rows)
        raise PlanError(
            f"k (IVF lists per block) is {k} but block {smallest.block} holds"
            f" only {result.min_block_rows} rows; FAISS needs at least k"
            f" training points per block. Suggested k:"
            f" {suggested_nlist(result.min_block_rows)}"
            f" (at most {largest_nlist(result.min_block_rows)})."
            " Rows rejected while reading lower this further, so the worker"
            " checks it again against the rows it actually kept"
        )
    return result


def _check_invariants(result: Plan, num_index: int, total: int) -> None:
    """What the query path assumes about any index: exactly num_index
    blocks, none empty, every row in exactly one of them."""
    if len(result.blocks) != num_index:
        raise PlanError(
            f"planned {len(result.blocks)} blocks for num_index {num_index}"
        )
    planned = sum(block.rows for block in result.blocks)
    if planned != total:
        raise PlanError(f"planned {planned} rows for a source of {total}")
    empty = [block.block for block in result.blocks if block.rows == 0]
    if empty:
        raise PlanError(f"blocks {empty} would hold no rows")


def _check_shape(files: Sequence[FileInfo], features: int | None) -> None:
    dialects = {info.dialect for info in files}
    if len(dialects) > 1:
        raise PlanError(f"mixed dialects across files: {sorted(dialects)}")
    dimensions = {info.dimension for info in files}
    if len(dimensions) > 1:
        raise PlanError(f"mixed vector dimensions across files: {sorted(dimensions)}")
    (dimension,) = dimensions
    if features is not None and dimension != features:
        raise PlanError(
            f"features is {features} but the source vectors have {dimension} values"
        )


def _row_groups(files: Sequence[FileInfo]):
    groups = []
    offset = 0
    for info in files:
        for index, rows in enumerate(info.row_groups):
            if rows == 0:
                continue
            groups.append((info.uri, index, rows, offset))
            offset += rows
    return groups


def _boundaries(total: int, num_index: int) -> list[int]:
    """Row counts at which an evenly split source changes block."""
    return [round(total * j / num_index) for j in range(1, num_index)]


def _whole_groups(groups, num_index: int, total: int) -> list[BlockPlan]:
    # edges[i] = rows before row group i; a block boundary must sit on
    # an edge, chosen nearest to the even split while leaving enough
    # edges for the boundaries still to place
    edges = [0]
    for _, _, rows, _ in groups:
        edges.append(edges[-1] + rows)
    chosen = []
    previous = 0
    for j, target in enumerate(_boundaries(total, num_index), start=1):
        lowest = previous + 1
        highest = len(groups) - (num_index - j)  # leave one group per later block
        candidates = range(lowest, highest + 1)
        index = min(candidates, key=lambda i: (abs(edges[i] - target), i))
        chosen.append(index)
        previous = index
    cuts = [0, *chosen, len(groups)]
    blocks = []
    for block, (first, last) in enumerate(zip(cuts, cuts[1:])):
        ranges = tuple(
            Range(uri=uri, row_group=rg, start=0, end=rows, id_offset=offset)
            for uri, rg, rows, offset in groups[first:last]
        )
        blocks.append(BlockPlan(block=block, ranges=ranges))
    return blocks


def _sliced(groups, num_index: int, total: int) -> list[BlockPlan]:
    boundaries = [0, *_boundaries(total, num_index), total]
    blocks = []
    for block, (begin, finish) in enumerate(zip(boundaries, boundaries[1:])):
        ranges = []
        for uri, rg, rows, offset in groups:
            lo = max(begin, offset)
            hi = min(finish, offset + rows)
            if lo < hi:
                ranges.append(
                    Range(uri=uri, row_group=rg, start=lo - offset, end=hi - offset, id_offset=lo)
                )
        blocks.append(BlockPlan(block=block, ranges=tuple(ranges)))
    return blocks
