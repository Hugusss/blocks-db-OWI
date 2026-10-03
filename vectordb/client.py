import json
import os
import time
import boto3
from botocore.exceptions import ClientError
import numpy as np
import csv
from typing import List, Tuple, Optional

from .utils.dataset_ops import (
    upload_dataset,
    delete_dataset,
)

from .utils.query_ops import (
    get_vectors_by_id,
    list_vectors,
    list_vectors_paginated,
)

from .utils.index_ops import (
    list_indexes,
    save_index_config,
    load_index_config,
    delete_indexes,
)

from .utils.vector_tracking import VectorIndexTracker
from .utils.idmap import idmap_prefix, select as select_provenance
from .errors import BlocksDBError
from .utils.parquet import ParquetSourceError
from .indexing.prepare import prepare_build


from .serverless_vectordb import ServerlessVectorDB

BLOCK_SIZE = 500000  # ~500KB per block for CSV blocks


class NotAvailableOnParquet(BlocksDBError, RuntimeError):
    """A CSV-path feature asked of an index built from parquet."""


class IndexExists(BlocksDBError, RuntimeError):
    """A parquet build asked for a name that already holds an index or a CSV
    dataset."""


class NoIndex(BlocksDBError, ValueError):
    """A query asked for a dataset that has no index to search.

    It is a ValueError, so code that catches ValueError catches it too."""


class QueryMismatch(BlocksDBError, ValueError):
    """A query the index cannot answer as it stands."""


def check_queries(vectors, features):
    """Stop a query that the index cannot answer before any function is invoked.

    faiss asserts on the dimension inside the map function, with a message
    that names neither the index nor the query, and an empty batch would
    invoke the functions to search nothing.
    """
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise QueryMismatch(f"a query needs at least one vector, got an array of shape {vectors.shape}")
    if vectors.shape[1] != features:
        raise QueryMismatch(
            f"the index holds vectors of {features} dimensions,"
            f" the query has {vectors.shape[1]}"
        )


_TAGS_UNAVAILABLE = "a parquet build indexes no tags, so tag filters and get-by-tags are not available"
_VECTORS_UNAVAILABLE = (
    "its vectors cannot be read back, since it keeps no source.csv;"
    " provenance() maps ids to their source records"
)
_PUT_UNAVAILABLE = (
    "it is immutable: added vectors would reuse its positional ids and have no"
    " provenance; build it again with the new sources included"
)
_REINDEX_UNAVAILABLE = (
    "reindexing deletes the blocks and rebuilds them from a source.csv it does not have;"
    " build it again with initialize-database --format parquet"
)
_CSV_BUILD_UNAVAILABLE = (
    "a CSV build writes its blocks over the ones already there and orphans the id map;"
    " delete the dataset first, or build under another name"
)


def build_csv_blocks_from_local(csv_path: str):
    """Read a local CSV file and build csv_blocks (byte-offset chunks) + last_vid.

    Returns (blocks, last_vid).
    """
    blocks = []
    current_offset = 0
    current_block_start_id = None
    current_block_size = 0
    last_vid = None

    with open(csv_path) as f:
        for line in f:
            if not line.strip():
                continue
            id_str = line.split(",")[0]
            vec_id = int(id_str)

            if current_block_start_id is None:
                current_block_start_id = vec_id

            current_block_size += len(line) + 1

            if current_block_size >= BLOCK_SIZE:
                blocks.append({
                    "start_id": current_block_start_id,
                    "end_id": vec_id,
                    "offset": current_offset,
                    "size": current_block_size
                })
                current_offset += current_block_size
                current_block_start_id = None
                current_block_size = 0

            last_vid = vec_id

    if current_block_start_id is not None and last_vid is not None:
        blocks.append({
            "start_id": current_block_start_id,
            "end_id": last_vid,
            "offset": current_offset,
            "size": current_block_size
        })

    return blocks, last_vid
from .infra import refresh_lithops_credentials
from .utils.s3_utils import is_s3express_bucket

class VectorDBClient:

    def __init__(self, bucket: str, region: str = None, sqs_queue_url: str = None, dynamodb_table_name: str = None, wait_timeout: float = None):
        """Initialize client with S3 bucket, optional region, SQS queue URL and DynamoDB table name.

        ``wait_timeout`` is how many seconds a build or a query waits for
        functions that never start, counted from the last start or finish:
        None derives it from the backend's function timeout, 0 waits forever.
        """
        self.bucket = bucket
        self.sqs_queue_url = sqs_queue_url
        self.wait_timeout = wait_timeout
        # dataset names found without a parquet index, see _refuse_on_parquet
        self._no_parquet_index = set()

        if region:
            self.s3 = boto3.client("s3", region_name=region)
        else:
            self.s3 = boto3.client("s3")

        self.tracker = VectorIndexTracker(bucket, region, sqs_queue_url=sqs_queue_url, table_name=dynamodb_table_name)

    def create_dataset(self, name: str, csv_path: str):
        """
        Upload a local CSV file as a new dataset.
        Renames automatically to vectors_{name}.csv
        """

        self._refuse_on_parquet(name, _CSV_BUILD_UNAVAILABLE, fresh=True)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found.")

        upload_dataset(self.bucket, name, csv_path)

        print(f"Dataset '{name}' created successfully.")

    def delete_dataset(self, name: str):
        """Delete a dataset and all its data from S3 and DynamoDB."""
        delete_dataset(self.bucket, name)
        self.tracker.delete_tracking(name)
        print(f"Dataset '{name}' deleted.")

    def put_vectors(self, dataset_name: str, vectors: List[Tuple[int, List[float]]], auto_index: bool = False, tags: dict = None, per_vector_tags: List[Optional[dict]] = None):
        """
        Add new vectors to the pending storage (not yet indexed).

        Args:
            dataset_name: Name of the dataset
            vectors: List of (id, vector) tuples
            auto_index: If True and threshold reached, trigger indexing (requires Lambda setup)
            tags: Optional dict of key-value tags (e.g. {"source": "web"})
            per_vector_tags: Optional list of tag dicts, one per vector (3rd CSV column)

        Returns:
            Number of vectors added
        """
        self.refuse_put_on_parquet(dataset_name)
        key = self.tracker.put_vectors(dataset_name, vectors, tags=tags, per_vector_tags=per_vector_tags)
        print(f"Added {len(vectors)} vectors to pending storage for dataset '{dataset_name}' -> {key}")
        return len(vectors)

    def put_vector(self, dataset_name: str, vector_id: int, vector: List[float], tags: dict = None, per_vector_tags: dict = None):
        """Add a single vector to pending storage."""
        pvt = [per_vector_tags] if per_vector_tags else None
        return self.put_vectors(dataset_name, [(vector_id, vector)], tags=tags, per_vector_tags=pvt)

    def get_pending_vectors(self, dataset_name: str) -> List[Tuple[int, List[float]]]:
        """Get all vectors that are pending indexing."""
        return self.tracker.get_pending_vectors(dataset_name)

    def has_pending_vectors(self, dataset_name: str) -> bool:
        """Check if there are pending vectors that need indexing."""
        return self.tracker.has_pending_vectors(dataset_name)

    def is_vector_indexed(self, dataset_name: str, vector_id: int) -> bool:
        """Check if a specific vector is indexed."""
        return self.tracker.is_indexed(dataset_name, vector_id)

    def get_indexed_ids(self, dataset_name: str) -> set:
        """Get all indexed vector IDs."""
        return self.tracker.get_indexed_ids(dataset_name)

    def get_indexed_count(self, dataset_name: str) -> int:
        """Get count of indexed vectors from DynamoDB counter."""
        return self.tracker.get_next_id(dataset_name)

    def mark_vectors_indexed(self, dataset_name: str, indexed_ids: List[int]):
        """Mark vectors as indexed (called after reindex)."""
        self.tracker.mark_vectors_indexed(dataset_name, indexed_ids)

    def refresh_credentials(self):
        """Refresh AWS credentials in Lithops config from ~/.aws/credentials."""
        return refresh_lithops_credentials()

    def list_datasets(self):
        """Every dataset name in the bucket, in the order S3 lists them:
        the names under datasets/ first, then those only an index names."""
        datasets = []
        paginator = self.s3.get_paginator("list_objects_v2")

        # datasets/{name}/source.csv for a CSV build, and
        # datasets/{name}/source/... for a parquet build from local files
        for page in paginator.paginate(Bucket=self.bucket, Prefix="datasets/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                parts = key.split("/")
                if len(parts) >= 3 and (key.endswith("/source.csv") or parts[2] == "source"):
                    if parts[1] not in datasets:
                        datasets.append(parts[1])

        # a parquet build that read its sources in place from s3:// leaves
        # nothing under datasets/; its saved configuration,
        # indexes/{name}/{implementation}/config.json, names it
        for page in paginator.paginate(Bucket=self.bucket, Prefix="indexes/"):
            for obj in page.get("Contents", []):
                parts = obj["Key"].split("/")
                if len(parts) == 4 and parts[3] == "config.json" and parts[1] not in datasets:
                    datasets.append(parts[1])

        return datasets

    def parquet_config(self, dataset_name: str, indexes=None):
        """The saved configuration of the index a dataset holds when it was
        built from parquet, or None: no index, or one built from CSV.
        ``indexes`` is the result of ``list_indexes`` when the caller
        already has it."""
        if indexes is None:
            indexes = self.list_indexes(dataset_name)
        for implementation, num_index in indexes:
            config = load_index_config(self.bucket, dataset_name, implementation, num_index)
            if config.get("source_format") == "parquet":
                return config
        return None

    def _refuse_on_parquet(self, dataset_name: str, reason: str, fresh: bool = False):
        """Stop a CSV-path feature on an index built from parquet, before it
        reads a source.csv that does not exist or deletes blocks it cannot
        rebuild.

        Only a parquet build writes an id map beside its blocks, and it
        builds blocks only, so a listing capped at one key answers for an
        index of any size. A name found without one is remembered for the
        life of the client, so a loop of put_vector asks S3 once; a refusal
        is not remembered, so a parquet index deleted elsewhere does not
        keep its name refused. ``fresh`` asks again: what overwrites or
        deletes an index must not trust an answer another process may have
        changed since.
        """
        if not fresh and dataset_name in self._no_parquet_index:
            return
        listing = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=idmap_prefix(dataset_name, "blocks"), MaxKeys=1)
        if "Contents" in listing:
            self._no_parquet_index.discard(dataset_name)
            raise NotAvailableOnParquet(f"'{dataset_name}' was built from parquet: {reason}")
        self._no_parquet_index.add(dataset_name)

    def refuse_put_on_parquet(self, dataset_name: str):
        """Stop vectors from being added to an index built from parquet."""
        self._refuse_on_parquet(dataset_name, _PUT_UNAVAILABLE)

    def refuse_tags_on_parquet(self, dataset_name: str):
        """Stop a tag filter or a get-by-tags on an index built from parquet."""
        self._refuse_on_parquet(dataset_name, _TAGS_UNAVAILABLE)

    def get_vectors(self, dataset_name: str, ids):
        """Get vectors by their IDs from the dataset."""
        self._refuse_on_parquet(dataset_name, _VECTORS_UNAVAILABLE)
        return get_vectors_by_id(self.bucket, dataset_name, ids)

    def list_vectors(self, dataset_name: str, limit: int = 100):
        """List first N vectors from the dataset."""
        self._refuse_on_parquet(dataset_name, _VECTORS_UNAVAILABLE)
        return list_vectors(self.bucket, dataset_name, limit)

    def list_vectors_paginated(self, dataset_name: str, start=0, limit=100):
        """List vectors with pagination (start offset, limit)."""
        self._refuse_on_parquet(dataset_name, _VECTORS_UNAVAILABLE)
        return list_vectors_paginated(
            self.bucket,
            dataset_name,
            start,
            limit
        )
    
    def list_indexes(self, dataset_name: str):
        """List available index configs for a dataset."""
        return list_indexes(self.bucket, dataset_name)
    
    def index_dataset(self, dataset_name: str, config: dict, num_workers: int = 16, save_config: bool = True, track_indexed: bool = True, setup_auto_indexer: bool = True, csv_blocks: tuple = None):
        """
        Run indexing for a dataset using provided configuration.

        Args:
            dataset_name (str): Name of the dataset to index.
            config (dict): Indexing configuration (must include 'implementation' and 'num_index').
            num_workers (int): Number of workers for parallel indexing.
            save_config (bool): If True, saves the index configuration to S3 after indexing.
            track_indexed (bool): If True, tracks indexed vector IDs for hybrid queries.
            setup_auto_indexer (bool): If True, initializes DynamoDB state and vector tracking for auto-indexer.
                                       Set False for pure benchmark comparisons.
            csv_blocks (tuple, optional): (blocks, last_vid) pre-built from local file.
                                          If provided, avoids re-downloading CSV from S3.
        Returns:
            dict: Timing stats from the indexing process.
        """

        self._refuse_on_parquet(dataset_name, _CSV_BUILD_UNAVAILABLE, fresh=True)

        config["dataset"] = dataset_name
        config["storage_bucket"] = self.bucket

        print(f"Initializing ServerlessVectorDB for dataset '{dataset_name}'...")
        sv_vectordb = ServerlessVectorDB(wait_timeout=self.wait_timeout, **config)

        filename = f"datasets/{dataset_name}/source.csv"

        print("Starting indexing...")
        total_times = sv_vectordb.indexing(filename, num_workers)
        print("Indexing completed.")

        features = config.get("features", 96)
        num_index = config.get("num_index", 16)
        total_vectors = config.get("num_vectors", -1)
        if total_vectors <= 0:
            total_vectors = self._get_vector_count(dataset_name)
        
        bytes_per_vector = 8 + (features * 8)
        vectors_per_block = total_vectors // num_index
        bytes_per_block = vectors_per_block * bytes_per_vector

        use_s3express = is_s3express_bucket(self.bucket)

        csv_block_size = max(50 * bytes_per_vector, bytes_per_block)
        csv_block_size = config.get("csv_block_size", csv_block_size)

        print(f"\nBlock size info:")
        print(f"  Total vectors: {total_vectors:,}")
        print(f"  Features: {features}")
        print(f"  Bytes per vector: {bytes_per_vector}")
        print(f"  Blocks: {num_index}")
        print(f"  Vectors per block: {vectors_per_block:,}")
        print(f"  Bytes per block: {bytes_per_block:,}")

        if not use_s3express:
            print(f"\nTo sync Lambda threshold, run:")
            print(f"  blocks-db update-threshold {bytes_per_block} --dataset {dataset_name} --bucket {self.bucket}")

        if save_config:
            config["total_vectors"] = total_vectors
            config["bytes_per_vector"] = bytes_per_vector
            config["bytes_per_block"] = bytes_per_block
            config["csv_block_size"] = csv_block_size
            t0 = time.time()
            self.save_index_config(dataset_name, config)
            if setup_auto_indexer:
                self._setup_auto_indexer_state(dataset_name, config)
            total_times["save_config"] = time.time() - t0

        t0 = time.time()
        self._aggregate_centroid_tags_to_ddb(dataset_name, num_index, config)
        total_times["ddb_tags_time"] = time.time() - t0

        if track_indexed and setup_auto_indexer:
            t0 = time.time()
            self._track_indexed_vectors_from_csv(dataset_name, features, use_s3express=use_s3express, prebuilt_blocks=csv_blocks, csv_block_size=csv_block_size)
            total_times["tracking_time"] = time.time() - t0

        return total_times
    
    def index_parquet_dataset(self, dataset_name: str, sources: List[str], config: dict, save_config: bool = True, replace: bool = False):
        """
        Build an immutable index from parquet files.

        The blocks are planned from the file footers and checked against the
        disk of a function and the argument limit of Lithops before anything
        is written, deleted, uploaded or invoked. Local files are uploaded
        under datasets/{dataset_name}/source/; s3:// URIs are read in place.
        A failed or interrupted build removes the blocks written so far;
        functions still running may write theirs afterwards. The id of
        a vector is the position of its row across the files in sorted path
        order; the file's own id (id or record_id) is kept as record_id and
        returned by provenance().

        Args:
            dataset_name: Name of the dataset
            sources: Parquet files, as local paths or s3:// URIs; expand a directory
                     or a prefix first with vectordb.indexing.prepare.expand_sources
            config: Index configuration; must declare features, num_index and k
            save_config: If True, seeds the id counter in DynamoDB with the number
                         of source rows and saves the index configuration to S3
            replace: If True, deletes an existing parquet index of the same name
                     before building; otherwise a name that holds an index raises
                     IndexExists. A name that holds a CSV dataset (source.csv or
                     pending vectors) raises IndexExists either way

        Returns:
            dict: Timing stats from the indexing process, plus rows (vectors kept),
                  rejected (rows rejected) and blocks (the report of each block)
        """
        # the name is about to hold a parquet index, whatever it held before
        self._no_parquet_index.discard(dataset_name)
        # plan from the footers first: a build that cannot succeed
        # must not have uploaded its sources beforehand
        local = [source for source in sources if not source.startswith("s3://")]
        for source in local:
            if not os.path.exists(source):
                raise ParquetSourceError(f"{source}: not found")
        build_plan, sealed = prepare_build(list(sources), config)
        # a CSV dataset of the same name keeps pending vectors that its
        # queries would still search, with ids the new index may also use:
        # it goes with delete-dataset, never as a side effect of a build
        source_csv, pending = f"datasets/{dataset_name}/source.csv", f"pending/{dataset_name}/"
        csv_data = [source_csv] if self._has_key(source_csv) else []
        if self.s3.list_objects_v2(Bucket=self.bucket, Prefix=pending, MaxKeys=1).get("Contents"):
            csv_data.append(pending)
        if csv_data:
            raise IndexExists(
                f"'{dataset_name}' holds a CSV dataset ({', '.join(csv_data)});"
                " delete it with delete_dataset() (delete-dataset) before a parquet build"
            )
        prefix = f"indexes/{dataset_name}/{sealed['implementation']}/"
        existing = sum(len(page.get("Contents", [])) for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix))
        if existing and not replace:
            raise IndexExists(
                f"'{dataset_name}' already holds an index ({existing} objects under {prefix});"
                " pass replace=True (--replace) to delete it and build again"
            )

        # the key of a local file keeps its path below the common root, so
        # two files of the same name in different directories get two keys.
        # The keys are known before the upload, so the plan is checked with
        # the URIs the functions will read
        root = os.path.commonpath([os.path.abspath(s) for s in local]) if local else ""
        if local and os.path.isfile(root):
            root = os.path.dirname(root)
        keys = {}
        for source in local:
            relative = os.path.relpath(os.path.abspath(source), root) if root else os.path.basename(source)
            keys[source] = f"datasets/{dataset_name}/source/{relative}"
        uploaded = {os.path.abspath(source): f"s3://{self.bucket}/{key}" for source, key in keys.items()}
        sealed["source_keys"] = [
            uploaded.get(os.path.abspath(uri), uri) for uri in sealed["source_keys"]
        ]
        build_plan = build_plan.with_sources(uploaded) if uploaded else build_plan

        sealed["dataset"] = dataset_name
        sealed["storage_bucket"] = self.bucket
        sv_vectordb = ServerlessVectorDB(wait_timeout=self.wait_timeout, **sealed)
        sv_vectordb.check_plan(build_plan)

        if save_config:
            # the one DynamoDB write of this path goes first: a missing table
            # or permission must cost no upload, no function and no index
            # deleted. The seed is the footer total, so the gaps rejected
            # rows leave in the id space are never handed out from the
            # counter.
            self.tracker.initialize_next_id(dataset_name, build_plan.total_vectors, strict=True)
        if existing:
            self._delete_blocks(dataset_name, sealed["implementation"], "the previous index")
        for source, key in keys.items():
            self.s3.upload_file(source, self.bucket, key)

        print(f"Plan: {build_plan.total_vectors:,} rows in {build_plan.num_index} blocks"
              f" (smallest {build_plan.min_block_rows:,}, largest {build_plan.max_block_rows:,},"
              f" imbalance {build_plan.imbalance:.2f}x), dimension {build_plan.dimension}")
        print("Starting indexing...")
        try:
            times = sv_vectordb.indexing_from_plan(build_plan)
        except BaseException:
            # a half-built index must not be left behind. BaseException, not
            # Exception: an interrupted client (Ctrl-C) leaves the same
            # half-built index as a failed function
            self._delete_blocks(dataset_name, sealed["implementation"], "the failed build")
            raise
        print(f"Indexing completed: {times['rows']:,} vectors kept, {times['rejected']:,} rows rejected.")

        # what the build actually indexed, not what the footers promised
        sealed["num_vectors"] = times["rows"]
        sealed["rejected"] = times["rejected"]
        if save_config:
            t0 = time.time()
            save_index_config(self.bucket, dataset_name, sealed, s3_client=self.s3)
            times["save_config"] = time.time() - t0
        return times

    def _has_key(self, key: str) -> bool:
        """Whether ``key`` exists. It is asked for by name: a directory
        bucket lists only prefixes that end in '/'."""
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as error:
            if error.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def _delete_blocks(self, dataset_name: str, implementation: str, what: str) -> None:
        """Remove every object of the index; ``what`` names it in the message."""
        prefix = f"indexes/{dataset_name}/{implementation}/"
        pages = self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix)
        keys = [{"Key": item["Key"]} for page in pages for item in page.get("Contents", [])]
        for start in range(0, len(keys), 1000):
            self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": keys[start:start + 1000]})
        if keys:
            print(f"Removed {len(keys)} objects of {what} under {prefix}")

    def provenance(self, dataset_name: str, ids, implementation: str = "blocks") -> dict:
        """
        Map vector ids of an index built from parquet to their source records.

        Only the id map parts (idmap/block_{i}.parquet) whose id range covers
        a requested id are read. A dataset with no saved index configuration
        raises NoIndex.

        Args:
            dataset_name: Name of the dataset
            ids: Vector ids (the ids a query returns)
            implementation: Index implementation (default: "blocks")

        Returns:
            dict: {id: (record_id, chunk_idx)}; an id the index does not hold is absent
        """
        prefix = idmap_prefix(dataset_name, implementation)
        wanted = sorted({int(value) for value in ids})
        try:
            config = load_index_config(self.bucket, dataset_name, implementation, None)
        except ClientError as error:
            if error.response["Error"]["Code"] != "NoSuchKey":
                raise
            raise NoIndex(f"No index found for dataset '{dataset_name}'. Run indexing first.") from None
        blocks = config.get("block_ranges")
        if blocks:
            needed = {
                block for block, first, last in blocks
                if any(first <= value <= last for value in wanted)
            }
            keys = [f"{prefix}block_{block}.parquet" for block in sorted(needed)]
        else:
            pages = self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix)
            keys = [item["Key"] for page in pages for item in page.get("Contents", [])]
        parts = (self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read() for key in keys)
        return select_provenance(parts, wanted)

    def _get_vector_count(self, dataset_name: str) -> int:
        """Count total vectors in dataset."""
        s3 = boto3.client("s3")
        key = f"datasets/{dataset_name}/source.csv"
        try:
            response = s3.get_object(Bucket=self.bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            return sum(1 for line in content.strip().split('\n') if line.strip())
        except Exception:
            return 100000
    
    def _setup_auto_indexer_state(self, dataset_name: str, config: dict):
        """Set up DynamoDB state for auto-indexer to continue from where manual indexing left off."""

        try:
            table = self.tracker.table
            num_index = config.get("num_index", 16)
            
            table.update_item(
                Key={"centroid_id": f"{dataset_name}_CONFIG", "sk": "META"},
                UpdateExpression="SET current_accumulated_size = :zero, current_centroid_id = :next",
                ExpressionAttributeValues={":zero": 0, ":next": num_index}
            )
            print(f"Auto-indexer state initialized: next centroid will be {num_index}")
        except Exception as e:
            print(f"Warning: Could not set auto-indexer state: {e}")

    def _aggregate_centroid_tags_to_ddb(self, dataset_name: str, num_index: int, config: dict):
        """Read per-vector _tags.json from S3 and write aggregated centroid-level tags to DynamoDB."""

        implementation = config.get("implementation", "blocks")

        s3 = boto3.client("s3")
        table = self.tracker.table

        written = 0
        for cid in range(num_index):
            tags_key = f"indexes/{dataset_name}/{implementation}/centroid_{cid}_tags.json"
            try:
                raw = s3.get_object(Bucket=self.bucket, Key=tags_key)["Body"].read().decode()
                tags_dict = json.loads(raw)
            except Exception:
                continue

            aggregated = {}
            for vec_tags in tags_dict.values():
                if not isinstance(vec_tags, dict):
                    continue
                for k, v in vec_tags.items():
                    if k not in aggregated:
                        aggregated[k] = set()
                    aggregated[k].add(v)

            if not aggregated:
                continue

            tags_map = {k: sorted(v) for k, v in aggregated.items()}
            try:
                table.put_item(Item={
                    "centroid_id": f"DATASET#{dataset_name}",
                    "sk": f"CENTROID#{cid}#META",
                    "tags": tags_map
                })
                written += 1
            except Exception as e:
                print(f"  Warning: Could not save DDB tags for centroid {cid}: {e}")

        if written:
            print(f"Aggregated centroid-level tags saved to DynamoDB for {written} centroids")

    def reindex_pending(self, dataset_name: str, config: dict = None, num_workers: int = 16):
        """
        Reindex pending vectors and merge them into existing indexes.
        
        This rebuilds all indexes with the current pending vectors included.
        Pending vectors are marked as indexed after successful reindex.
        
        Args:
            dataset_name: Name of the dataset
            config: Optional config dict (uses stored config if not provided)
            num_workers: Number of workers for indexing
            
        Returns:
            Timing stats from the indexing process
        """
        self._refuse_on_parquet(dataset_name, _REINDEX_UNAVAILABLE, fresh=True)
        if not self.has_pending_vectors(dataset_name):
            print("No pending vectors to reindex.")
            return {}
        
        pending = self.get_pending_vectors(dataset_name)
        print(f"Found {len(pending)} pending vectors.")
        
        if config is None:
            indexes = self.list_indexes(dataset_name)
            if not indexes:
                raise ValueError(f"No index config found for '{dataset_name}'. Provide config explicitly.")
            implementation, num_index = indexes[0]
            config = load_index_config(self.bucket, dataset_name, implementation, num_index)
        
        pending_ids = [v[0] for v in pending]
        
        delete_indexes(self.bucket, dataset_name)
        
        config["dataset"] = dataset_name
        config["storage_bucket"] = self.bucket
        
        sv_vectordb = ServerlessVectorDB(wait_timeout=self.wait_timeout, **config)
        filename = f"datasets/{dataset_name}/source.csv"
        
        print("Rebuilding indexes with pending vectors...")
        total_times = sv_vectordb.indexing(filename, num_workers)
        print("Reindexing completed.")
        
        self.tracker.clear_pending(dataset_name)
        
        all_indexed_ids = list(self.tracker.get_indexed_ids(dataset_name)) + pending_ids
        self.tracker.create_indexed_tracking(dataset_name, all_indexed_ids)
        print(f"Now tracking {len(all_indexed_ids)} total indexed vectors.")
        
        return total_times

    def index_pending_separate(self, dataset_name: str, config: dict = None):
        """
        Create separate indexes for pending vectors without rebuilding main index.
        
        This adds pending vectors as additional data without full reindex.
        They will be searched via hybrid query until a full reindex is done.
        
        Args:
            dataset_name: Name of the dataset
            config: Optional config dict (uses stored config if not provided)
        """
        if not self.has_pending_vectors(dataset_name):
            print("No pending vectors to index.")
            return
        
        if config is None:
            indexes = self.list_indexes(dataset_name)
            if not indexes:
                raise ValueError(f"No index config found for '{dataset_name}'. Provide config explicitly.")
            implementation, num_index = indexes[0]
            config = load_index_config(self.bucket, dataset_name, implementation, num_index)
        
        pending = self.get_pending_vectors(dataset_name)
        pending_ids = [v[0] for v in pending]
        
        self.tracker.mark_vectors_indexed(dataset_name, pending_ids)
        print(f"Marked {len(pending_ids)} pending vectors as indexed for hybrid search.")

    def _track_indexed_vectors_from_csv(self, dataset_name: str, features: int, use_s3express: bool = False, prebuilt_blocks: tuple = None, csv_block_size: int = BLOCK_SIZE):
        """Read the main CSV and track all vectors as indexed + build csv_blocks for optimized get.

        If prebuilt_blocks=(blocks, last_vid) is provided, skip S3 download
        and use the pre-built blocks instead.
        """
        if prebuilt_blocks is not None:
            blocks, last_vid = prebuilt_blocks
            if blocks:
                blocks_key = f"tracking/csv_blocks_{dataset_name}.json"
                self.s3.put_object(Bucket=self.bucket, Key=blocks_key, Body=json.dumps(blocks))
                print(f"Stored {len(blocks)} pre-built CSV blocks for optimized get.")
            if last_vid is not None:
                next_id = last_vid + 1
                self.tracker.initialize_next_id(dataset_name, next_id)
                print(f"Tracked {next_id} vectors as indexed (from local file).")
                print(f"Initialized DynamoDB next_id={next_id} for atomic ID tracking.")
            return

        key = f"datasets/{dataset_name}/source.csv"
        block_size = csv_block_size
        blocks = []

        try:
            head = self.s3.head_object(Bucket=self.bucket, Key=key)
        except Exception as e:
            print(f"Warning: Could not get CSV metadata: {e}")
            return

        try:
            current_offset = 0
            current_block_start_id = None
            current_block_size = 0
            last_vid = None

            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            for raw_line in obj["Body"].iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode()
                id_str = line.split(",")[0]
                vec_id = int(id_str)

                if current_block_start_id is None:
                    current_block_start_id = vec_id

                current_block_size += len(raw_line) + 1

                if current_block_size >= block_size:
                    blocks.append({
                        "start_id": current_block_start_id,
                        "end_id": vec_id,
                        "offset": current_offset,
                        "size": current_block_size
                    })
                    current_offset += current_block_size
                    current_block_start_id = None
                    current_block_size = 0

                last_vid = vec_id

            if current_block_start_id is not None and last_vid is not None:
                blocks.append({
                    "start_id": current_block_start_id,
                    "end_id": last_vid,
                    "offset": current_offset,
                    "size": current_block_size
                })

            if blocks:
                blocks_key = f"tracking/csv_blocks_{dataset_name}.json"
                self.s3.put_object(Bucket=self.bucket, Key=blocks_key, Body=json.dumps(blocks))
                print(f"Built {len(blocks)} CSV blocks for optimized get.")

            if last_vid is not None:
                next_id = last_vid + 1
                self.tracker.initialize_next_id(dataset_name, next_id)
                print(f"Tracked {next_id} vectors as indexed.")
                print(f"Initialized DynamoDB next_id={next_id} for atomic ID tracking.")
        except Exception as e:
            print(f"Warning: Could not track indexed vectors: {e}")

    def save_index_config(self, dataset_name: str, config: dict):
        """
        Save index configuration to S3 for future reindexing.
        """
        save_index_config(self.bucket, dataset_name, config)
        print(f"Index configuration saved for dataset '{dataset_name}'.")


    def get_vector_ids_by_tags(self, dataset_name: str, filter_tags: dict, limit: int = 100) -> List[int]:
        """Get vector IDs matching ALL filter tags by scanning centroid _tags.json files."""
        self.refuse_tags_on_parquet(dataset_name)
        import json as _json
        s3 = boto3.client("s3")
        prefix = f"indexes/{dataset_name}/blocks/"
        matching_ids = []

        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if not key.endswith("_tags.json") or key.endswith("_reverse_tags.json"):
                        continue
                    try:
                        raw = s3.get_object(Bucket=self.bucket, Key=key)["Body"].read().decode()
                        tags_data = _json.loads(raw)
                        for vid_str, vt in tags_data.items():
                            if all(vt.get(k) == v for k, v in filter_tags.items()):
                                matching_ids.append(int(vid_str))
                                if len(matching_ids) >= limit:
                                    return matching_ids[:limit]
                    except Exception:
                        continue
        except Exception:
            pass

        return matching_ids[:limit]

    def delete_index_configs(self, dataset_name: str):
        """
        Delete all saved index configuration files (config.json) for a dataset.
        """
        prefix = f"indexes/{dataset_name}/"
        paginator = self.s3.get_paginator("list_objects_v2")

        deleted_count = 0

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            if "Contents" in page:
                configs_to_delete = [
                    {"Key": obj["Key"]}
                    for obj in page["Contents"]
                    if obj["Key"].endswith("config.json")
                ]
                if configs_to_delete:
                    self.s3.delete_objects(
                        Bucket=self.bucket,
                        Delete={"Objects": configs_to_delete}
                    )
                    deleted_count += len(configs_to_delete)

        print(f"Deleted {deleted_count} config(s) for dataset '{dataset_name}'")
    
    def query(self, dataset_name: str, vector: List[float], k: int = None, hybrid: bool = True, batch_size: int = None, filter_tags: dict = None, filter_mode: str = "post"):
        """
        Query a single vector. Always searches everything (indexed + pending).
        
        Args:
            dataset_name: Name of the dataset
            vector: Query vector (list of floats)
            k: Number of results (defaults to config k_result)
            hybrid: If True, include pending vectors in search (default: True)
            batch_size: Override query_batch_size for this query
            filter_tags: Optional dict of tag filters (e.g. {"source": "web"}). Only centroids/pending files matching ALL tags are searched.
            
        Returns:
            List of (id, distance) tuples sorted by distance
        """
        results, times = self.query_batch(dataset_name, [vector], k=k, hybrid=hybrid, batch_size=batch_size, filter_tags=filter_tags, filter_mode=filter_mode)
        return results[0], times

    def query_batch(self, dataset_name: str, vectors: List[List[float]], k: int = None, hybrid: bool = True, batch_size: int = None, filter_tags: dict = None, filter_mode: str = "post"):
        """
        Query multiple vectors. Always searches everything (indexed + pending) by default.
        
        Args:
            dataset_name: Name of the dataset
            vectors: List of query vectors
            k: Number of results per query
            hybrid: If True, include pending vectors in search (default: True)
            batch_size: Override query_batch_size for this query
            filter_tags: Optional dict of tag filters (e.g. {"source": "web"}). Only centroids/pending files matching ALL tags are searched.
            
        Returns:
            neighbours, times
        """

        if not vectors:
            raise QueryMismatch("No query vectors provided.")

        vectors_np = np.array(vectors)
        
        if hybrid:
            return self._query_hybrid(dataset_name, vectors_np, k, batch_size=batch_size, filter_tags=filter_tags, filter_mode=filter_mode)
        else:
            return self._query_indexed_only(dataset_name, vectors_np, k, batch_size=batch_size, filter_tags=filter_tags, filter_mode=filter_mode)
    
    def query_indexed_only(self, dataset_name: str, vector: List[float] = None, vectors: List[List[float]] = None, k: int = None, batch_size: int = None, filter_tags: dict = None, filter_mode: str = "post"):
        """
        Query only the indexed vectors (no pending).
        
        Args:
            dataset_name: Name of the dataset
            vector: Single query vector
            vectors: Multiple query vectors (alternative to single vector)
            k: Number of results
            batch_size: Override query_batch_size for this query
            filter_tags: Optional dict of tag filters (e.g. {"source": "web"})
            
        Returns:
            Search results from indexed data only
        """
        if vector is not None:
            vecs = [vector]
        elif vectors is not None:
            vecs = vectors
        else:
            raise ValueError("Provide either vector or vectors")
        if not len(vecs):
            raise QueryMismatch("No query vectors provided.")
        
        return self._query_indexed_only(dataset_name, np.array(vecs), k, batch_size=batch_size, filter_tags=filter_tags, filter_mode=filter_mode)

    def _vector_tags_match(self, vector_tags: dict, filter_tags: dict) -> bool:
        if not filter_tags:
            return True
        if not vector_tags:
            return False
        for k, v in filter_tags.items():
            if vector_tags.get(k) != v:
                return False
        return True

    def _post_filter_results(self, results, filter_tags, centroid_tags_map):
        """Post-filter query results using per-vector tags from centroid tags.
        
        centroid_tags_map: {centroid_id: {faiss_id_str: {key: val}}} loaded from _tags.json
        """
        filtered = []
        for r in results:
            qf = []
            for vid, dist, src in r:
                cid = None
                if src.startswith("centroid_"):
                    cid = int(src.split("centroid_")[1].split("_")[0]) if "_" in src else None
                tags = {}
                if cid is not None:
                    ct = centroid_tags_map.get(cid, {})
                    tags = ct.get(str(vid), {})
                if self._vector_tags_match(tags, filter_tags):
                    qf.append((vid, dist, src))
            filtered.append(qf)
        return filtered

    def _query_indexed_only(self, dataset_name: str, vectors_np: np.ndarray, k: int = None, batch_size: int = None, filter_tags: dict = None, filter_mode: str = "post"):
        """Query only the FAISS index (no pending vectors)."""
        k = k if k is not None else self._get_k_result(dataset_name)

        sv_vectordb = self._load_default_index(dataset_name, batch_size=batch_size, filter_tags=filter_tags, filter_mode=filter_mode)
        check_queries(vectors_np, sv_vectordb.params.features)
        neighbours, times = sv_vectordb.search(0, vectors_np, filter_tags=filter_tags)
        if not neighbours:
            neighbours = [[] for _ in range(len(vectors_np))]

        return neighbours, times

    def _query_hybrid(self, dataset_name: str, vectors_np: np.ndarray, k: int = None, batch_size: int = None, filter_tags: dict = None, filter_mode: str = "post"):
        """Internal hybrid query implementation."""
        k = k if k is not None else self._get_k_result(dataset_name)

        # only a missing index is a fallback: a search that fails must not
        # come back as no results at all
        results = times = None
        try:
            sv_vectordb = self._load_default_index(dataset_name, batch_size=batch_size, filter_tags=filter_tags, filter_mode=filter_mode)
        except NoIndex as missing:
            no_index = missing
        else:
            no_index = None
            check_queries(vectors_np, sv_vectordb.params.features)
            results, times = sv_vectordb.search(0, vectors_np, filter_tags=filter_tags)

        if not results and self.has_pending_vectors(dataset_name):
            from .utils.hybrid_search import brute_force_search
            unindexed = self.get_pending_vectors(dataset_name)
            if unindexed:
                results = brute_force_search(vectors_np, unindexed, k)
                for r in results:
                    r[:] = [(vid, dist, "pending") for vid, dist in r]
                times = {"fallback": "no index, searched pending only"}

        if not results:
            if no_index is not None:
                # nothing was searched: saying so beats an empty answer that
                # reads like "no neighbors"
                raise no_index
            results = [[] for _ in range(len(vectors_np))]

        times["hybrid_search"] = True
        times["has_pending"] = self.has_pending_vectors(dataset_name)
        if filter_tags:
            times["filter_tags"] = filter_tags

        return results, times
    
    def query_from_file(self, dataset_name: str, csv_path: str, hybrid: bool = True, k: int = None, batch_size: int = None, filter_tags: dict = None, filter_mode: str = "post"):
        """
        Query ALL vectors from a CSV file. Always searches everything by default.
        
        Args:
            dataset_name: Name of the dataset
            csv_path: Path to CSV file with query vectors (space-separated floats per row)
            hybrid: If True, include pending vectors (default: True)
            k: Number of results per query
            batch_size: Override query_batch_size for this query
            filter_tags: Optional dict of tag filters (e.g. {"source": "web"})
            
        Returns:
            neighbours, times
        """

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found.")

        vectors = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                vec = [float(x) for x in row[0].split(" ") if x]
                vectors.append(vec)

        if not vectors:
            raise QueryMismatch("No valid vectors found in CSV.")

        vectors_np = np.array(vectors)
        
        if hybrid:
            return self._query_hybrid(dataset_name, vectors_np, k, batch_size=batch_size, filter_tags=filter_tags, filter_mode=filter_mode)
        else:
            return self._query_indexed_only(dataset_name, vectors_np, k, batch_size=batch_size, filter_tags=filter_tags, filter_mode=filter_mode)

    def query_hybrid(self, dataset_name: str, vectors: List[List[float]], k: int = None, batch_size: int = None, filter_tags: dict = None, filter_mode: str = "post"):
        """
        Explicit hybrid query - searches both indexed and pending vectors.
        Alias for query_batch with hybrid=True.
        
        Args:
            dataset_name: Name of the dataset
            vectors: List of query vectors
            k: Number of results per query (defaults to config k_result)
            batch_size: Override query_batch_size for this query
            filter_tags: Optional dict of tag filters (e.g. {"source": "web"})
            
        Returns:
            Merged search results with (id, distance) tuples
        """
        return self.query_batch(dataset_name, vectors, k=k, hybrid=True, batch_size=batch_size, filter_tags=filter_tags, filter_mode=filter_mode)

    def _get_k_result(self, dataset_name: str) -> int:
        """Get k_result from index config."""
        try:
            config = self._load_index_config_for_search(dataset_name)
            return config.get("k_result", 10)
        except:
            return 10

    def _load_index_config_for_search(self, dataset_name: str) -> dict:
        """Load index config for search operations."""
        indexes = self.list_indexes(dataset_name)
        if not indexes:
            raise ValueError(f"No index found for dataset '{dataset_name}'.")
        
        if len(indexes) > 1:
            raise ValueError(
                f"Multiple indexes found for dataset '{dataset_name}'. "
                f"Specify implementation and num_index manually."
            )
        
        implementation, num_index = indexes[0]
        return load_index_config(self.bucket, dataset_name, implementation, num_index)
    
    def _load_default_index(self, dataset_name: str, batch_size: int = None, filter_tags: dict = None, filter_mode: str = "post"):
        """
        Load the only stored index config for a dataset.
        Raises error if none or more than one exist.
        """

        indexes = self.list_indexes(dataset_name)

        if not indexes:
            raise NoIndex(f"No index found for dataset '{dataset_name}'. Run indexing first.")

        if len(indexes) > 1:
            raise ValueError(
                f"Multiple indexes found for dataset '{dataset_name}'. "
                f"Specify implementation and num_index manually."
            )

        implementation, num_index = indexes[0]

        config = load_index_config(
            self.bucket,
            dataset_name,
            implementation,
            num_index
        )
        if filter_tags and config.get("source_format") == "parquet":
            raise NotAvailableOnParquet(f"'{dataset_name}' was built from parquet: {_TAGS_UNAVAILABLE}")

        # no function reads the source list or the block ranges a parquet
        # build seals into config.json; dropped here, they do not travel
        # with every map and reduce task
        config.pop("source_keys", None)
        config.pop("block_ranges", None)

        config["dataset"] = dataset_name
        config["storage_bucket"] = self.bucket
        config["dynamodb_table_name"] = self.tracker.table_name
        config["dynamodb_region"] = self.tracker.dynamodb.meta.client.meta.region_name

        if batch_size is not None:
            if batch_size < 1:
                raise QueryMismatch(f"the query batch size is how many blocks a function searches, so it must be at least 1, got {batch_size}")
            config["query_batch_size"] = batch_size
        if filter_tags is not None:
            config["filter_tags"] = filter_tags
        config["filter_mode"] = filter_mode

        return ServerlessVectorDB(wait_timeout=self.wait_timeout, **config)