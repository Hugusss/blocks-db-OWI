# Blocks-DB Python Client

## Initialization

```python
from vectordb.client import VectorDBClient

client = VectorDBClient(bucket="your-bucket", region="us-east-1")
# With SQS:
client = VectorDBClient(bucket="your-bucket", region="us-east-1", sqs_queue_url="https://sqs...")
# With a DynamoDB table other than BlocksDB-default:
client = VectorDBClient(bucket="your-bucket", region="us-east-1", dynamodb_table_name="your-table")
# Seconds a build or a query waits for functions that never start, counted from
# the last start or finish (default: the function timeout + 60; 0 waits forever):
client = VectorDBClient(bucket="your-bucket", region="us-east-1", wait_timeout=1200)
```

## Parquet Indexes

`index_parquet_dataset` builds an index from parquet files instead of a CSV file; the [README](../README.md) describes the column layouts it reads. An index built this way, a parquet index, is immutable, has no tags, and its vectors cannot be read back, so everything marked "refused on a parquet index" below raises `NotAvailableOnParquet` before it changes anything.

A client remembers a dataset it found without a parquet index. If another client then builds one under that name, this client still refuses `create_dataset`, `index_dataset`, `reindex_pending` and `filter_tags`, which check every time, but not the other methods.

`vectordb.indexing.prepare.expand_sources(source, list_s3=None, files="*.parquet")` turns a source into the sorted list of files `index_parquet_dataset` takes. A local directory gives the files under it at any depth, and an `s3://` prefix (a URI ending in `/`) the keys returned by `list_s3(bucket, prefix)`, a function you pass, keeping those whose file name matches the shell pattern `files`; any other source is one file.

## Dataset Management

| Method | Description |
|--------|-------------|
| `create_dataset(name, csv_path)` | Upload a local CSV file as a new dataset; refused on a parquet index |
| `delete_dataset(name)` | Delete dataset and all its data from S3 and DynamoDB, including the parquet files `index_parquet_dataset` uploaded; files read in place from `s3://` are left untouched |
| `list_datasets()` | List all datasets in the bucket, including one that has only a saved index configuration (a parquet index built from `s3://` files) |
| `save_index_config(name, config)` | Save index configuration to S3 |
| `delete_index_configs(name)` | Delete all saved config.json files for a dataset |
| `list_indexes(name)` | List available index configs for a dataset |

## Putting Vectors

| Method | Description |
|--------|-------------|
| `put_vectors(dataset_name, vectors, tags, per_vector_tags)` | Add vectors to pending storage (batch); refused on a parquet index |
| `put_vector(dataset_name, vector_id, vector, tags, per_vector_tags)` | Add a single vector to pending storage; refused on a parquet index |
| `refuse_put_on_parquet(dataset_name)` | Raise `NotAvailableOnParquet` if the dataset holds a parquet index (the check `put_vectors` makes) |
| `get_pending_vectors(dataset_name)` | Get all pending (unindexed) vectors |
| `has_pending_vectors(dataset_name)` | Check if there are pending vectors |
| `mark_vectors_indexed(dataset_name, indexed_ids)` | Mark vectors as indexed |

## Querying

| Method | Description |
|--------|-------------|
| `query(dataset_name, vector, k, hybrid, batch_size, filter_tags, filter_mode)` | Single vector query (hybrid by default) |
| `query_batch(dataset_name, vectors, k, hybrid, batch_size, filter_tags, filter_mode)` | Multi-vector query |
| `query_hybrid(dataset_name, vectors, k, batch_size, filter_tags, filter_mode)` | Explicit hybrid query (alias for `query_batch` with `hybrid=True`) |
| `query_indexed_only(dataset_name, vector, vectors, k, batch_size, filter_tags, filter_mode)` | Query only the FAISS index (skip pending) |
| `query_from_file(dataset_name, csv_path, hybrid, k, batch_size, filter_tags, filter_mode)` | Query all vectors from a CSV file |
| `get_vector_ids_by_tags(dataset_name, filter_tags, limit)` | Get vector IDs matching ALL filter tags; refused on a parquet index |
| `refuse_tags_on_parquet(dataset_name)` | Raise `NotAvailableOnParquet` if the dataset holds a parquet index (the check `get_vector_ids_by_tags` makes) |
| `get_vectors(dataset_name, ids)` | Get vectors by their IDs; refused on a parquet index (use `provenance`) |
| `list_vectors(dataset_name, limit)` | List first N vectors; refused on a parquet index |
| `list_vectors_paginated(dataset_name, start, limit)` | List vectors with pagination; refused on a parquet index |

## Index Management

| Method | Description |
|--------|-------------|
| `index_dataset(dataset_name, config, num_workers, save_config, track_indexed, setup_auto_indexer, csv_blocks)` | Run full indexing pipeline; refused on a parquet index |
| `index_parquet_dataset(dataset_name, sources, config, save_config, replace)` | Build a parquet index from `sources`, a list of parquet files: local paths (uploaded under `datasets/{name}/source/`) or `s3://` URIs (read in place); `vectordb.indexing.prepare.expand_sources` lists the files of a directory or a prefix. `config` must declare `features`, `num_index` and `k`. `replace=True` deletes a parquet index of the same name first, and a name that holds a CSV dataset raises `IndexExists` either way; a failed or interrupted build removes the blocks written so far, and functions still running may write theirs afterwards |
| `provenance(dataset_name, ids, implementation)` | Map ids of a parquet index to their source records, as `{id: (record_id, chunk_idx)}`; ids not in the index are left out. `record_id` is a string: the owi-v2 `record_id`, or a canonical row's `id` (its index id when the file has no `id` column) with `chunk_idx` 0 |
| `parquet_config(dataset_name, indexes)` | Get the saved configuration of a parquet index (`num_vectors`, `rejected`, `source_keys`, ...), or `None` for a CSV index or no index; `indexes` is the result of `list_indexes`, if already at hand |
| `reindex_pending(dataset_name, config, num_workers)` | Rebuild all indexes with pending vectors included; refused on a parquet index |
| `index_pending_separate(dataset_name, config)` | Mark pending vectors as indexed without rebuilding |
| `get_indexed_ids(dataset_name)` | Get all indexed vector IDs |
| `get_indexed_count(dataset_name)` | Get count of indexed vectors from DynamoDB counter |
| `is_vector_indexed(dataset_name, vector_id)` | Check if a specific vector is indexed |

## Credentials

| Method | Description |
|--------|-------------|
| `refresh_credentials()` | Refresh AWS credentials in Lithops config from `~/.aws/credentials` |

## Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `int` | `10` | Number of results per query |
| `hybrid` | `bool` | `True` | Include pending vectors in search |
| `batch_size` | `int` | `None` | Override `query_batch_size` (centroid .ann per map worker) |
| `filter_tags` | `dict` | `None` | Tag filters (e.g. `{"source": "web"}`); refused on a parquet index |
| `filter_mode` | `str` | `"post"` | `"post"` (overfetch + loop) or `"pre"` (reverse-index + IDSelector) |
| `auto_index` | `bool` | `False` | Trigger auto-indexing if threshold reached |
| `tags` | `dict` | `None` | Batch-level tags for `put_vectors` |
| `per_vector_tags` | `list[dict]` | `None` | Per-vector tags (3rd CSV column) |

## Errors

Every exception in this table derives from `BlocksDBError` (`vectordb.errors`, also exported by `blocks_db`), so `except BlocksDBError` catches all of them. Each one is also the builtin named beside it, so code that catches that builtin keeps working. Some input mistakes on the CSV path still raise a plain `ValueError` or `FileNotFoundError`.

| Exception | Module | Raised when |
|-----------|--------|-------------|
| `NoIndex` (a `ValueError`) | `vectordb.client` | A query or `provenance` names a dataset that has no index; a hybrid query raises it only when there are no pending vectors either |
| `QueryMismatch` (a `ValueError`) | `vectordb.client` | The query vectors do not have the index's dimension, the batch is empty, or `batch_size` is below 1 |
| `NotAvailableOnParquet` (a `RuntimeError`) | `vectordb.client` | A method or parameter marked "refused on a parquet index" is used on one; the message says why |
| `IndexExists` (a `RuntimeError`) | `vectordb.client` | `index_parquet_dataset` names a dataset that already holds an index, without `replace=True`, or that holds a CSV dataset (`source.csv` or pending vectors), with or without it |
| `PlanError` (a `ValueError`) | `vectordb.indexing.planner` | A parquet build cannot start, and nothing has been written yet; the message names the cause, such as a config without `features`, `num_index` or `k`, a config that does not fit the files, a block too big for the function's disk (`aws_lambda.ephemeral_storage`), or task arguments over Lithops' `data_limit`. `expand_sources` raises it for a source that does not exist or holds no matching file, or a prefix given without `list_s3` |
| `ParquetSourceError` (a `ValueError`) | `vectordb.utils.parquet` | A parquet file cannot be read as vectors (unreadable, or no vector column), or a local source given to `index_parquet_dataset` does not exist; the message names the file |
| `BlockTooSmall` (a `ValueError`) | `vectordb.implementations.blocks.initialize` | A block of a parquet build keeps fewer rows than `k` once its rejected rows are dropped |
| `CounterUnavailable` (a `RuntimeError`) | `vectordb.utils.vector_tracking` | `index_parquet_dataset` cannot write the id counter of the dataset in DynamoDB; nothing has been deleted or uploaded yet |
| `FunctionsTimedOut` (a `TimeoutError`) | `vectordb.utils.waiting` | Functions of a build or a query never started, and none started or finished within the wait timeout |

## Examples

### Create dataset and index

```python
client.create_dataset("mydata", "vectors.csv")
config = {
    "features": 96,
    "num_index": 16,
    "k": 512,
    "n_probe": 32,
    "implementation": "blocks",
}
client.index_dataset("mydata", config, num_workers=16)
```

### Build an index from parquet files and read back provenance

```python
from vectordb.indexing.prepare import expand_sources

sources = expand_sources("vectors/")  # every *.parquet under the directory
config = {"features": 1024, "num_index": 16, "k": 256, "n_probe": 32}
client.index_parquet_dataset("mydocs", sources, config)

results, times = client.query_indexed_only("mydocs", vector=query_vector)  # 1024 floats
ids = [hit[0] for hit in results[0]]
client.provenance("mydocs", ids)  # {id: (record_id, chunk_idx), ...}
```

### Query with tag filter

```python
results, times = client.query(
    "mydata",
    [0.1, 0.2, 0.3],
    k=10,
    filter_tags={"source": "web"},
    filter_mode="pre",
)
```

### Add vectors with tags

```python
client.put_vectors(
    "mydata",
    [(1, [0.1, 0.2]), (2, [0.3, 0.4])],
    tags={"source": "ingest"},
)
```
