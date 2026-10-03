# vectordb — Blocks-DB Python Package

## Data Flow

```
CLI (cli.py)
  └─ VectorDBClient (client.py)
       └─ ServerlessVectorDB (serverless_vectordb.py)
            ├─ Indexing: indexator.py (Lithops map/reduce on centroids)
            └─ Search:   Orchestrator (Lithops map/reduce)
```

`cli.py` parses CLI args and delegates to `VectorDBClient`.
`client.py` manages datasets (put, status, delete) and wraps both indexing (`index_dataset`, `index_parquet_dataset`) and search (`query`).
`serverless_vectordb.py` bridges client calls to the Lithops `FunctionExecutor` — it handles serialization, function dispatch, and result collection.
`indexing/indexator.py` and `orchestration/orchestrator.py` contain the actual map/reduce logic.

## Top-Level Modules

| File | Description |
|------|-------------|
| `cli.py` | CLI entry point (`blocks-db` command) |
| `client.py` | `VectorDBClient` — Python API for all operations |
| `config.py` | `InfraConfig` and `SvlessVectorDBParams` dataclasses |
| `errors.py` | `BlocksDBError` — base of the errors a user can correct |
| `serverless_vectordb.py` | `ServerlessVectorDB` — Lithops-based index/search wrapper |
| `benchmarks.py` | Recall calculation helpers |

## Submodules

| Directory | Description |
|-----------|-------------|
| `config/` | Default index config (`indexconfig.json`) |
| `core/` | Abstract base classes for backend implementations |
| `implementations/` | Backend implementations (e.g. `blocks`) |
| `indexing/` | Indexing pipeline orchestration |
| `orchestration/` | Distributed map/reduce search |
| `infra/` | AWS infrastructure provisioning |
| `utils/` | S3, DynamoDB, CSV, parquet, hybrid search utilities |

---

## 🗂️ S3 Structure

```
your-bucket/
├── vectors_<dataset>.csv              # Main dataset
├── csv_blocks_<dataset>.json          # Block index for fast ID lookup
├── pending/<dataset>/                 # Pending vectors (individual CSVs)
├── processed/<dataset>/               # Processed pending batches
├── tracking/
│   └── csv_blocks_<dataset>.json      # Byte-offset chunks for fast ID-based vector retrieval
├── indexes/<dataset>/blocks/
│   ├── config.json
│   ├── centroid_*.ann                 # FAISS index blocks
│   ├── centroid_*_tags.json           # Per-vector tags (forward index)
│   ├── centroid_*_reverse_tags.json   # Reverse index for pre-filter mode
│   └── idmap/block_*.parquet          # Parquet builds: id, record_id, chunk_idx of each vector
├── datasets/<dataset>/source.csv      # Alternative dataset path
├── datasets/<dataset>/source/         # Parquet builds: uploaded copies of local source files
└── inputs/                            # Temporary Lithops data
```

---

## Architecture

### Project Structure

```
vectordb/
├── cli.py              # CLI entry point (blocks-db command)
├── client.py           # VectorDBClient — Python API for all operations
├── config.py           # InfraConfig, SvlessVectorDBParams dataclasses
├── serverless_vectordb.py  # ServerlessVectorDB — Lithops wrapper
├── benchmarks.py       # Recall calculation helpers
├── config/             # Default index config (indexconfig.json)
├── core/               # ABCs: IndexBuilder, Partitioner, Preprocessor, QueryStrategy
├── implementations/    # Backend implementations (blocks/)
├── indexing/           # Indexing pipeline orchestration via Lithops
├── orchestration/      # Distributed map/reduce search
├── infra/              # AWS provisioning (setup.py), Lambda code, Dockerfiles
└── utils/              # S3, DynamoDB, CSV, parquet, hybrid search, tracking utilities
```

Each subdirectory contains a README.md detailing its files:

| Directory | README |
|-----------|--------|
| `config/` | [`config/README.md`](config/README.md) |
| `core/` | [`core/README.md`](core/README.md) |
| `indexing/` | [`indexing/README.md`](indexing/README.md) |
| `orchestration/` | [`orchestration/README.md`](orchestration/README.md) |
| `utils/` | [`utils/README.md`](utils/README.md) |

### Blocks & Concepts

- **Block**: a FAISS IVF centroid stored as `centroid_{id}.ann` in S3. Each block represents a region of the vector space.
- **num_index**: total number of blocks (centroids) in the dataset. The vector space is partitioned into `num_index` regions via k-means clustering during `initialize-database`.
- **k** (config): IVF k-means clusters per block during initial build. During auto-indexing, k is computed dynamically as `max(1, min(4096, len(vectors) // 4))`.
- **n_probe**: IVF search parameter — number of clusters to visit within each block during search. Higher values improve recall at the cost of latency.
- **query_batch_size**: number of centroid `.ann` files assigned to each Lithops map worker. Controls parallelism: total map workers = `ceil(num_index / query_batch_size)` + 1 (pending). Default 4.
- **csv_blocks**: byte-offset chunks of the source CSV stored at `tracking/csv_blocks_{dataset}.json`. Built during `initialize-database` for efficient Range GETs when retrieving vectors by ID (`blocks-db get`). Auto-calculated from index config or overridable via `--csv-block-size`.
- **features**: vector dimensionality (e.g. 96 for deep_100k, 384 for all-MiniLM-L6-v2, 1536 for ada-002).
- **index_mem / search_map_mem / search_reduce_mem**: Lambda memory in MB for indexing, search map, and search reduce phases respectively.

### DynamoDB Schema

Table `BlocksDB-default` (configurable), HASH=`centroid_id` (String), RANGE=`sk` (String), PAY_PER_REQUEST.

| centroid_id | sk | Attributes | Purpose | Created by |
|---|---|---|---|---|
| `{dataset}_CONFIG` | `META` | `current_accumulated_size`, `current_centroid_id` | Auto-indexer state (pending bytes and next centroid) | Lambda (`ensure_global_metadata`), client (`_setup_auto_indexer_state`) |
| `{dataset}#{id}` | `FILE#{key}` | `prefix`, `file_key`, `size`, `dataset`, `tags` | Per-centroid pending file tracking (deleted after indexing) | Lambda (`accumulate_file`) |
| `PENDING` | `FILE#{key}` | `dataset`, `ids`, `file_key` | Global pending file index | Client (`_update_pending_tracking`) |
| `DATASET#{dataset}` | `CENTROID#{id}#META` | `tags` | Aggregated centroid-level tags for pre-filter queries | Lambda (`create_index_for_centroid`), client (`_aggregate_centroid_tags_to_ddb`) |
| `{dataset}_ID_TRACKER` | `META` | `next_id`, `dataset` | Atomic ID counter for new vectors | Client (`initialize_next_id`), Lambda (`get_next_available_id_atomic`) |

### Tag System & Data Flow

Blocks-DB supports two kinds of tags, which can coexist:

#### Batch Tags

Passed via `--tags '{"source":"web"}'` on `put`. Stored in S3 object metadata (`x-amz-meta-tags`) and forwarded to the Lambda via SQS message body (SQS mode) or S3 metadata (S3 trigger mode). Applied to all vectors in the batch. Used for centroid-level and pending-file-level filtering.

#### Per-Vector Tags

Embedded as a third CSV column:

```
100000,0.152 0.106 ...,{"source":"api","priority":"high"}
```

Read by the Lambda during auto-indexing or by the client during `initialize-database`. Written to per-centroid files in S3:
- `centroid_{id}_tags.json` — forward index: `{faiss_id: {key: val, ...}, ...}`
- `centroid_{id}_reverse_tags.json` — reverse index: `{"key:val": [faiss_id, ...], ...}`

Also aggregated into DynamoDB as `DATASET#{dataset} / CENTROID#{id}#META` with `tags: {"source": ["api", "web"], "priority": ["high", "normal"]}`.

**Flow:**

1. `blocks-db put` uploads CSV to `pending/{dataset}/{timestamp}.csv` (per-vector tags in 3rd column, batch tags in S3 metadata)
2. S3 trigger or SQS fires the auto-indexer Lambda
3. Lambda accumulates the file; when threshold is reached, builds a FAISS index for one centroid
4. Lambda writes `centroid_{id}.ann`, `centroid_{id}_tags.json`, `centroid_{id}_reverse_tags.json` to S3
5. Lambda writes `DATASET#{dataset} / CENTROID#{id}#META` with aggregated tags to DynamoDB
6. Query with `--filter` reads DDB to determine which centroids to search, then uses `_tags.json` (post-filter) or `_reverse_tags.json` (pre-filter) for per-vector filtering

#### Coexistence

Both tag types can be present simultaneously. Batch tags filter at the centroid/pending-file level (fewer workers), per-vector tags filter at the individual vector level within searched centroids.

### Filter Modes

Both modes share a first step: `_get_matching_centroids()` queries DynamoDB for `DATASET#{dataset} / CENTROID#*#META` and returns only centroids whose aggregated tags match ALL filter key-value pairs. Centroids without DDB tag records are excluded when a filter is active.

#### Post-Filter (default, `--filter-mode post`)

1. Centroid-level DDB filter (same as above)
2. Search each matching centroid with overfetch: `k * post_filter_overfetch` (default 2x)
3. Load `centroid_{id}_tags.json` for each centroid
4. Iterate through candidates, keep only those whose per-vector tags match ALL filter keys
5. Trim to k

Best for: general use, dense tag distributions.

#### Pre-Filter (`--filter-mode pre`)

1. Centroid-level DDB filter (same as above)
2. Load `centroid_{id}_reverse_tags.json` for each centroid
3. Compute matching FAISS IDs as intersection of `"key:val"` lookups (AND semantics)
4. Pass `faiss.IDSelectorBatch(matching_ids)` to FAISS via `SearchParametersIVF`
5. FAISS only searches/returns vectors whose IDs are in the selector

Best for: sparse tag distributions (few vectors match), avoids overfetch misses.

A client-side post-filter fallback (`client.py:_post_filter_results`) also runs after hybrid searches.

### Auto-Indexer Lambda

The auto-indexer Lambda (`infra/lambda/lambda_code.py`) processes pending vectors incrementally.

#### Threshold

The accumulated pending size (in bytes) that triggers indexing. Override chain:

1. **Per-invocation**: `threshold_bytes` in the Lambda event payload
2. **Per-file**: S3 metadata `threshold-bytes` / `threshold_bytes`
3. **Environment variable**: `THRESHOLD_SIZE_BYTES` on the Lambda function (set by `setup` or `update-threshold`)
4. **Default**: 33,554,432 bytes (32 MB)

After `initialize-database`, `update-threshold` auto-calculates from the index config: `vectors_per_block * (8 + features * 8)`.

#### Flow

1. **Dispatch**: `lambda_handler` detects `eventSource == "aws:sqs"` → `_handle_sqs_event`, otherwise `_handle_s3_event`
2. **Accumulate**: `accumulate_file()` reads the pending CSV metadata, stores file info in DDB as `{dataset}#{centroid_id} / FILE#{key}`, and adds file size to `{dataset}_CONFIG / META.current_accumulated_size`
3. **Check**: `_check_and_index_datasets()` compares `current_accumulated_size` against threshold
4. **Claim**: Atomically increments `current_centroid_id` via DDB conditional update (prevents double-indexing by concurrent invocations)
5. **Build**: `create_index_for_centroid()` reads all pending files for the claimed centroid, parses vectors and per-vector tags, builds a FAISS IVF index, writes `.ann`, `_tags.json`, `_reverse_tags.json` to S3, and saves centroid DDB tag record
6. **Cleanup**: Deletes pending files from S3, removes DDB tracking items

#### S3 vs SQS Trigger

- **Standard S3**: S3 Event Notification on `pending/*.csv` fires Lambda directly. Default mode.
- **SQS**: CLI sends SQS message after each `put`. Lambda polls the queue (batch size 10, 20s long polling). Includes a DLQ. Required for S3 Express One Zone buckets (no S3 notification support).

### Indexing Pipeline (`initialize-database`)

The full index build via Lithops, used for initial dataset ingestion.

1. **Upload**: CLI uploads the source CSV to `datasets/{name}/source.csv`
2. **Preprocess** (optional): backend-specific chunking of the CSV
3. **Distribute** (optional): backend-specific distribution of vector data
4. **Build**: Lithops map over `num_index` centroids. Each worker builds a FAISS IVF index for its assigned centroids using `faiss.index_factory(features, "IVF{k},Flat")` with `add_with_ids`
5. **Post-process** (`client.py`):
   - Writes `centroid_{id}_tags.json` and DDB centroid tag records
   - Builds CSV blocks for fast ID lookup
   - Initializes `{dataset}_CONFIG / META` (auto-indexer starts at `current_centroid_id = num_index`)
   - Initializes `{dataset}_ID_TRACKER`
   - Optionally updates Lambda threshold
6. **Tracked vectors** are marked as indexed (not pending)

### Parquet Indexing Pipeline (`--format parquet`)

Builds an immutable index from parquet files. The files of one build share one of two column layouts, `canonical` (`vector`, optional `id`) or `owi-v2` (`record_id`, `chunk_idx`, `embedding`); any other column is ignored.

1. **Expand** (CLI): `expand_sources()` turns the source into a sorted list of files: a directory gives the files under it, at any depth, whose name matches `--files` (default `*.parquet`), an `s3://` URI ending in `/` gives the matching keys under that prefix, and any other source is one file
2. **Plan**: `prepare_build()` reads the file footers, skips files with no rows and splits the rows into exactly `num_index` contiguous blocks; the id of a vector is the position of its row across the files
3. **Check**: before anything is seeded, deleted or uploaded, the build stops when the name holds a CSV dataset (`datasets/{name}/source.csv` or `pending/{name}/`), when it already holds an index and `--replace` is not given, when the largest block does not fit the disk of a function (`aws_lambda.ephemeral_storage` in the Lithops configuration, 512 MB unless set), or when the arguments sent to the functions exceed `lithops.data_limit` (4 MiB unless set)
4. **Seed**: sets `{dataset}_ID_TRACKER` in DynamoDB to the number of source rows; a failed write stops the build
5. **Replace** (`--replace`): deletes the previous index under `indexes/{name}/blocks/`
6. **Upload**: local files are uploaded to `datasets/{name}/source/{path below their common root}`; `s3://` sources are read in place
7. **Build** (Lithops map, one `build_block_from_parquet` function per block):
   - Reads the row ranges of its block into float32 vectors and scales `owi-v2` vectors to unit length
   - Rejects rows with no vector or a vector of the wrong length, and `owi-v2` rows that cannot be scaled (all zeros, or a non-finite value)
   - Fails with `BlockTooSmall` when fewer rows than `k` are left
   - Trains and fills a FAISS IVF index and uploads it as `centroid_{i}.ann`, with its id map part `idmap/block_{i}.parquet`
   - When a function fails or the client is interrupted, the client removes the blocks written so far; functions still running may write theirs afterwards, and `--replace` or `delete-dataset` removes them
8. **Save**: writes `config.json` with the index parameters plus `source_format` (`parquet`), `source_keys`, `source_rows` (rows in the footers), `num_vectors` (vectors kept), `rejected`, `unit_norm` (`true` for `owi-v2`), `source_files_skipped` (files with no rows) and `block_ranges` (first and last id of each block)

### Lithops Search Pipeline

1. **Upload queries**: Queries are serialized as JSON and uploaded to `queries/testdata/{uuid}_queries_{dataset}_{num_index}.csv`
2. **Create map tasks**: `BlocksQueryStrategy.create_map_tasks()` determines which centroids to search:
   - If `filter_tags` is set: queries DDB for matching centroids + pending files
   - One task per `query_batch_size` centroids, one task for pending files
3. **Map phase** (Lithops `map_function`):
   - Indexed workers: download assigned `.ann` files from S3, FAISS search with pre/post filter, return per-query results
   - Pending worker: downloads pending CSVs, brute-force search via `IndexFlatL2`
4. **Reduce preparation**: `create_reduce_iterdata()` groups map results by query index, writes intermediate files to S3 (`reduce/res_{n}.json`)
5. **Reduce phase** (Lithops `reduce_function`):
   - Merges per-query results from multiple map tasks
   - Deduplicates, sorts by distance, trims to k
   - Returns to client

### Common Pitfalls

- **Python version**: strict pin to `>=3.10,<3.11` in `pyproject.toml`. Do not use 3.11+.
- **Lithops picklability**: map/reduce functions must be module-level (picklable). Classes work only for local preprocess phases, not for distributed execution.
- **S3 trigger scope**: the Lambda is triggered only on `pending/*.csv` (prefix + suffix filter). Files outside this pattern are ignored.
- **Threshold sizing**: if the threshold is larger than the pending data, the Lambda will accumulate files without indexing. Use `update-threshold` to adjust after `initialize-database`.
- **`'dict' object has no attribute 'startswith'` warning**: visible during `setup` Lambda update, harmless — the layer update succeeds regardless.
- **A function that dies before it reports a start** (a broken runtime, a handler that fails) is not seen by Lithops' timeout, which ends only functions that started. The client gives up on it when no function has started or finished for the backend's `runtime_timeout` + 60 s (`lithops.execution_timeout` + 60 s on a backend without one, such as localhost) (`FunctionsTimedOut`); functions that keep starting or finishing, however slowly, keep the wait alive. `wait_timeout=0` on the client waits forever.
- **The functions run the `vectordb` of the runtime image**: Lithops sends with a job only the modules the image does not have, and the image installs the package. A change under `vectordb/` reaches the functions once the image is built again from the repository root (`BUILDX_NO_DEFAULT_ATTESTATIONS=1 lithops runtime build -f vectordb/infra/Dockerfile.lambda -b aws_lambda <runtime-name>`; Lambda refuses an image that carries the attestation manifests Docker adds by default) and the deployed functions are deleted (`lithops runtime delete <runtime-name> -b aws_lambda`), or under a new runtime name. `setup` builds the image the same way, so export `BUILDX_NO_DEFAULT_ATTESTATIONS=1` before `blocks-db setup` too.
- **Pending tracking during auto-index**: after the Lambda processes pending files, they are deleted from S3 and their DDB tracking items are removed. The status command will show no pending vectors.
- **Parquet ids are row positions**: the id of a vector, and of a query result, is the position of its row across the source files in sorted path order, not the `id` or `record_id` the file gives it. Rejected rows leave gaps in the ids. `provenance()` maps an id to `(record_id, chunk_idx)`; for a `canonical` file, that is the file's `id` as text (the index id when the file has none) and `0`.
- **Parquet indexes are immutable**: they have no pending vectors, no auto-indexer and no tags. `put`, `get` by id or with `--limit`, `get-by-tags`, `query --filter`, `reindex_pending()` and a CSV build over the same name stop with `NotAvailableOnParquet`. To change the index, build it again with `--format parquet --replace`.
- **Parquet index config**: `features`, `num_index` and `k` must be declared. Without `k`, the error suggests a value for the smallest block.
- **Function image for parquet builds**: the functions of a parquet build import pyarrow and `vectordb.utils.parquet`, so the runtime image must hold both. Build it from the repository root with `BUILDX_NO_DEFAULT_ATTESTATIONS=1 lithops runtime build -f vectordb/infra/Dockerfile.lambda -b aws_lambda <runtime-name>` (Lambda refuses an image that carries the attestation manifests Docker adds by default), then run `lithops runtime delete <runtime-name> -b aws_lambda` so the functions are created again from the new image, or build it under a new runtime name and set that name in the Lithops configuration. A new `ephemeral_storage` applies the same way: a deployed function keeps its size until it is deleted and created again.
