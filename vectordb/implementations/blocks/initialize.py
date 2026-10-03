import faiss
import json
import os
import tempfile
import time
from lithops import Storage

import numpy as np

from vectordb.implementations.blocks.indexing import FaissIVFIndex
from vectordb.errors import BlocksDBError
from vectordb.implementations.blocks.partitioning import BlockPartitioner


class BlockTooSmall(BlocksDBError, ValueError):
    """A block whose surviving rows cannot train its IVF lists."""


def generate_index_blocks(id, obj, params, n_blocks, storage: Storage):

    start = time.time()

    csv_data = obj.data_stream.read().decode("utf-8")

    partitioner = BlockPartitioner(n_blocks)

    blocks = partitioner.partition(csv_data)

    index_builder = FaissIVFIndex(params)

    key_id = id * n_blocks

    for ids, vectors, tags_dict in blocks:

        index = index_builder.build(ids, vectors)

        faiss.write_index(index, f"/tmp/{key_id}.ann")

        storage.upload_file(
            f"/tmp/{key_id}.ann",
            params.storage_bucket,
            f"indexes/{params.dataset}/{params.implementation}/centroid_{key_id}.ann",
        )

        if tags_dict:
            tags_key = f"indexes/{params.dataset}/{params.implementation}/centroid_{key_id}_tags.json"
            storage.put_object(
                params.storage_bucket,
                tags_key,
                json.dumps(tags_dict).encode("utf-8")
            )

            reverse = {}
            for vid_str, vt in tags_dict.items():
                for k, v in vt.items():
                    reverse.setdefault(f"{k}:{v}", []).append(int(vid_str))
            reverse_key = f"indexes/{params.dataset}/{params.implementation}/centroid_{key_id}_reverse_tags.json"
            storage.put_object(
                params.storage_bucket,
                reverse_key,
                json.dumps(reverse).encode("utf-8")
            )

        key_id += 1

    return time.time() - start


def build_block_from_parquet(block_plan, params, storage: Storage):
    """Map function of the parquet path: one block per task.

    Reads the row ranges of ``block_plan`` through the parquet reader,
    assigns ``id_offset + position`` as the vector id, trains and fills
    one IVF block exactly as the CSV path does, and uploads the block
    plus its provenance map ``idmap/block_{i}.parquet`` (``id``,
    ``record_id``, ``chunk_idx``; canonical files carry their own id, or
    the index id when they have none, as ``record_id`` text and
    ``chunk_idx`` 0). Returns the block number, the rows kept, the rows
    rejected and the seconds the block took.

    Raises :class:`BlockTooSmall` when the rows that survive reading are
    fewer than the IVF list count, which the planner can only bound from
    the footers.
    """
    # imported here, so the CSV build of this module runs on an image
    # without pyarrow
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vectordb.utils.parquet import iter_ranges

    start = time.time()
    dimension = params.features
    # the plan says how many rows this block can hold, so the matrix is
    # allocated once and filled in place; rejected rows leave it short
    # and it is trimmed at the end
    matrix = np.empty((block_plan.rows, dimension), dtype=np.float32)
    all_ids = np.empty(block_plan.rows, dtype=np.int64)
    chunk_idx = np.empty(block_plan.rows, dtype=np.int64)
    record_ids: list[str] = []
    kept = 0
    rejected = 0
    for uri, parts in _by_file(block_plan.ranges):
        ranges = [(part.row_group, part.start, part.end) for part in parts]
        for part, rows in zip(parts, iter_ranges(uri, dimension, ranges)):
            end = kept + rows.kept
            matrix[kept:end] = rows.vectors
            all_ids[kept:end] = part.id_offset + rows.positions
            rejected += rows.rejected
            if rows.record_ids is not None:
                record_ids.extend(rows.record_ids)
                chunk_idx[kept:end] = rows.chunk_idx
            else:
                # a canonical file without an id column: the index id is the record id
                ids = all_ids[kept:end] if rows.ids is None else rows.ids
                record_ids.extend(str(value) for value in ids.tolist())
                chunk_idx[kept:end] = 0
            kept = end
    matrix = matrix[:kept]
    all_ids = all_ids[:kept]
    chunk_idx = chunk_idx[:kept]

    if kept < params.k:
        # FAISS refuses to train fewer points than lists with a C++ error
        # that names neither the block nor the cause; this one does
        raise BlockTooSmall(
            f"block {block_plan.block}: {kept} usable rows out of"
            f" {block_plan.rows} planned ({rejected} rejected while reading)"
            f", fewer than k = {params.k} IVF lists. Lower k, use fewer"
            " blocks, or fix the source rows"
        )
    index = FaissIVFIndex(params).build(all_ids, matrix)

    prefix = f"indexes/{params.dataset}/{params.implementation}"
    with tempfile.TemporaryDirectory() as workdir:
        block_path = os.path.join(workdir, f"centroid_{block_plan.block}.ann")
        faiss.write_index(index, block_path)
        storage.upload_file(block_path, params.storage_bucket, f"{prefix}/centroid_{block_plan.block}.ann")

        idmap_path = os.path.join(workdir, f"block_{block_plan.block}.parquet")
        pq.write_table(
            pa.table(
                {
                    "id": pa.array(all_ids, pa.int64()),
                    "record_id": pa.array(record_ids, pa.string()),
                    "chunk_idx": pa.array(chunk_idx, pa.int64()),
                }
            ),
            idmap_path,
        )
        storage.upload_file(idmap_path, params.storage_bucket, f"{prefix}/idmap/block_{block_plan.block}.parquet")

    return {
        "block": block_plan.block,
        "rows": int(len(all_ids)),
        "rejected": int(rejected),
        "seconds": time.time() - start,
    }


def _by_file(ranges):
    """Consecutive ranges of the same file, in plan order."""
    groups = []
    for part in ranges:
        if groups and groups[-1][0] == part.uri:
            groups[-1][1].append(part)
        else:
            groups.append((part.uri, [part]))
    return groups


def get_index_builder():
    return generate_index_blocks