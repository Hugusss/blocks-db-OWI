import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import boto3

from .client import VectorDBClient, build_csv_blocks_from_local
from .errors import BlocksDBError
from .infra import run_setup, refresh_lithops_credentials, get_infra_config
from .config import DEFAULT_INFRA_CONFIG
from .utils.s3_utils import is_s3express_bucket, parse_express_az
from .utils.vector_utils import load_vectors_with_ids_from_csv, load_vectors_with_ids_and_tags_from_csv, load_vectors_from_csv


CONFIG_DIR = Path.home() / ".blocks-db-config"
CONFIG_FILE = CONFIG_DIR / "backend_config.json"


def load_backend_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


# What a user can get wrong, or the cloud refuse: said as a message instead of
# a traceback. Anything else is a bug and keeps its traceback.
EXPECTED_ERRORS = (BlocksDBError,)


def main():
    try:
        _run()
    except EXPECTED_ERRORS as error:
        sys.exit(f"Error: {error}")


def _run():
    parser = argparse.ArgumentParser(
        prog="blocks-db",
        description="Blocks-DB: Serverless Vector Database",
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, max_help_position=40)
    )

    parser.add_argument("--bucket", help="S3 bucket")
    parser.add_argument("--region", help="AWS region")
    parser.add_argument("--table-name", help="DynamoDB table (default: BlocksDB-default)")
    parser.add_argument("--wait-timeout", type=float, help="Seconds a build or a query waits for functions that never start, counted from the last start or finish (default: the function timeout + 60; 0 waits forever)")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── setup ─────────────────────────────────────────────────
    setup_parser = subparsers.add_parser("setup", help="Setup infrastructure (Lambda, DynamoDB, S3 triggers)")
    setup_parser.add_argument("--bucket", required=True, help="S3 bucket for storage")
    setup_parser.add_argument("--runtime-name", default=None, help="Runtime image name in ECR")
    setup_parser.add_argument("--function-name", default=None, help="Lambda function name")
    setup_parser.add_argument("--table-name", default=argparse.SUPPRESS, help="DynamoDB table name")
    setup_parser.add_argument("--layer-name", default=None, help="Lambda layer name")
    setup_parser.add_argument("--role-name", default=None, help="Lambda execution role name")
    setup_parser.add_argument("--threshold", type=int, default=None, help="Auto-indexer threshold in bytes (default: 5242880)")
    setup_parser.add_argument("--skip-vector-table", action="store_true", help="Skip DynamoDB table creation")
    setup_parser.add_argument("--skip-runtime", action="store_true", help="Skip Lithops runtime build")
    setup_parser.add_argument("--s3express", action="store_true", help="Use S3 Express One Zone (auto-enables SQS)")
    setup_parser.add_argument("--sqs", action="store_true", help="Use SQS instead of S3 notifications for auto-indexer trigger")

    # ── update-threshold ─────────────────────────────────────
    threshold_parser = subparsers.add_parser("update-threshold", help="Update auto-indexer block size threshold")
    threshold_parser.add_argument("threshold_bytes", nargs="?", type=int, default=None, help="Threshold in bytes (optional, auto-calculate if not provided)")
    threshold_parser.add_argument("--dataset", default=None, help="Dataset name (for auto-calculate)")
    threshold_parser.add_argument("--bucket", default=None, help="S3 bucket (required if --dataset used)")

    # ── deploy-lambda ───────────────────────────────────
    deploy_lambda_parser = subparsers.add_parser("deploy-lambda", help="Deploy/update Lambda code")
    deploy_lambda_parser.add_argument("--function", default=None, help="Lambda function name")

    subparsers.add_parser("refresh-credentials", help="Refresh AWS credentials")

    configure_parser = subparsers.add_parser("configure", help="Save default bucket and region")
    configure_parser.add_argument("--bucket", required=True, help="Default S3 bucket")
    configure_parser.add_argument("--region", default="us-east-1", help="AWS region")
    configure_parser.add_argument("--sqs", action="store_true", help="Use SQS for auto-indexer notifications")
    configure_parser.add_argument("--table-name", default=argparse.SUPPRESS, help="DynamoDB table used by the client")
    configure_parser.add_argument("--wait-timeout", type=float, default=argparse.SUPPRESS, help="Seconds to wait for functions that never start, counted from the last start or finish (0 waits forever)")

    # ── initialize-database ───────────────────────────────────
    init_parser = subparsers.add_parser("initialize-database", help="Upload initial dataset and create index")
    init_parser.add_argument("name", help="Dataset name")
    init_parser.add_argument("source", help="CSV file with vectors; with --format parquet, a parquet file, a directory of parquet files, or an s3:// URI (a prefix only when it ends with '/')")
    init_parser.add_argument("--format", choices=("csv", "parquet"), default="csv", help="Source format (default: csv). parquet builds an immutable index: no csv_blocks, no auto-indexer; features, num_index and k must be declared in the config")
    init_parser.add_argument("--config", required=True, help="Path to index config JSON")
    init_parser.add_argument("--replace", action="store_true", help="With --format parquet: delete an existing parquet index of the same name and build again (a CSV dataset of that name is removed with delete-dataset first)")
    init_parser.add_argument("--files", default=None, help="With --format parquet: shell pattern for the file names read from a directory or s3:// prefix (default: *.parquet); '*_embeddings.parquet' skips the records files the Open Web Index publishes beside its embeddings")
    init_parser.add_argument("--workers", type=int, default=None, help="Number of indexing workers (default: 16; CSV builds only)")
    init_parser.add_argument("--no-update-threshold", action="store_true", help="Skip auto-update threshold after indexing (CSV builds only)")
    init_parser.add_argument("--skip-auto-indexer", action="store_true", help="Skip DynamoDB state init and vector tracking (for pure benchmarks; CSV builds only)")
    init_parser.add_argument("--build-local", action="store_true", help="Build csv_blocks from local file (skip S3 re-download during tracking; CSV builds only)")
    init_parser.add_argument("--csv-block-size", type=int, default=None, help="CSV block size in bytes for optimized vector reads (default: auto-calculated from index config; CSV builds only)")

    # ── put ───────────────────────────────────────────────────
    put_parser = subparsers.add_parser("put", help="Add new vectors (stored as individual files)")
    put_parser.add_argument("name", help="Dataset name")
    put_parser.add_argument("csv_path", help="CSV file with vectors (id, vector values, optional 3rd col: JSON tags)")
    put_parser.add_argument("--single", action="store_true", help="Treat as single vector per file (one vector per file)")
    put_parser.add_argument("--tags", type=str, default=None, help='JSON dict of tags (e.g. \'{"source":"web","category":"news"}\')')

    # ── query ─────────────────────────────────────────────────
    query_parser = subparsers.add_parser("query", help="Query vectors (searches indexed + pending)")
    query_parser.add_argument("name", help="Dataset name")
    query_group = query_parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--vector", help="Space-separated float values")
    query_group.add_argument("--file", help="CSV file with query vectors")
    query_parser.add_argument("--k", type=int, default=10, help="Number of results")
    query_parser.add_argument("--indexed-only", action="store_true", help="Search only indexed vectors (skip pending)")
    query_parser.add_argument("--batch-size", type=int, default=None, help="Override query_batch_size (centroid .ann per map worker)")
    query_parser.add_argument("--filter", type=str, default=None, help='JSON dict of tag filters (e.g. \'{"source":"web"}\'), AND semantics, only centroids/pending matching ALL tags are searched')
    query_parser.add_argument(
        "--filter-mode", default=None, choices=["post", "pre"],
        help="Tag filter mode: post (default, overfetch + loop) or pre (reverse-index + IDSelector)"
    )

    # ── get-by-tags ────────────────────────────────────────────
    get_tags_parser = subparsers.add_parser("get-by-tags", help="Get vector IDs matching tags")
    get_tags_parser.add_argument("name", help="Dataset name")
    get_tags_parser.add_argument("--filter", required=True, type=str, help='JSON dict of tag filters (e.g. \'{"source":"web"}\')')
    get_tags_parser.add_argument("--limit", type=int, default=100, help="Max IDs to return (default: 100)")

    # ── delete-dataset ────────────────────────────────────────
    del_parser = subparsers.add_parser("delete-dataset", help="Delete dataset and all its data")
    del_parser.add_argument("name", help="Dataset name")
    del_parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    # ── status ────────────────────────────────────────────────
    status_parser = subparsers.add_parser("status", help="Show dataset status")
    status_parser.add_argument("name", help="Dataset name")
    status_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed info")

    # ── get ───────────────────────────────────────────────────
    get_parser = subparsers.add_parser("get", help="Get vectors by ID, list vectors, or show pending")
    get_parser.add_argument("name", help="Dataset name")
    get_parser.add_argument("ids", nargs="*", type=int, help="Vector IDs (optional)")
    get_parser.add_argument("--limit", type=int, default=None, help="Limit number of vectors to list")
    get_parser.add_argument("--pending", action="store_true", help="Get pending vectors instead of main")

    args = parser.parse_args()

    if args.command == "initialize-database":
        # an option of the other format is refused, not ignored; an option
        # not given is None, or False for a flag, and any other value (0
        # included) was typed by the user
        csv_only = ("--workers", "--no-update-threshold", "--skip-auto-indexer", "--build-local", "--csv-block-size")
        parquet_only = ("--replace", "--files")
        for option in csv_only if args.format == "parquet" else parquet_only:
            value = getattr(args, option[2:].replace("-", "_"))
            if value is not None and value is not False:
                init_parser.error(f"{option} does not apply to --format {args.format}")

    # a path given on the command line that does not exist ends the command
    # here, named, before anything is opened or uploaded; a FileNotFoundError
    # raised later is a bug and keeps its traceback. A parquet source is
    # checked when it is expanded.
    paths = []
    if args.command == "initialize-database":
        paths = [args.config] if args.format == "parquet" else [args.config, args.source]
    elif args.command == "put":
        paths = [args.csv_path]
    elif args.command == "query" and args.file:
        paths = [args.file]
    for path in paths:
        if not os.path.exists(path):
            sys.exit(f"Error: {path}: not found")

    VISIBLE_COMMANDS = {
        "setup", "configure", "refresh-credentials", "update-threshold",
        "initialize-database", "put", "query", "status", "get",
        "get-by-tags", "deploy-lambda", "delete-dataset"
    }

    if not args.command:
        parser.print_help()
        return

    if args.command not in VISIBLE_COMMANDS:
        print(f"Command '{args.command}' is not available.")
        print("Run 'blocks-db --help' to see available commands.")
        return

    file_config = load_backend_config()

    bucket = args.bucket or os.getenv("SVDB_BUCKET") or file_config.get("bucket")
    region = args.region or os.getenv("SVDB_REGION") or file_config.get("region")
    table_name = args.table_name or os.getenv("SVDB_DYNAMODB_TABLE") or file_config.get("dynamodb_table_name")
    wait_timeout = args.wait_timeout
    if wait_timeout is None and os.getenv("SVDB_WAIT_TIMEOUT"):
        try:
            wait_timeout = float(os.environ["SVDB_WAIT_TIMEOUT"])
        except ValueError:
            parser.error(f"SVDB_WAIT_TIMEOUT must be a number of seconds, got '{os.environ['SVDB_WAIT_TIMEOUT']}'")
    if wait_timeout is None:
        wait_timeout = file_config.get("wait_timeout")

    commands_without_bucket = ["setup", "configure"]
    if args.command not in commands_without_bucket and not bucket:
        parser.error(
            "Bucket not provided. Use --bucket, set SVDB_BUCKET, or run 'blocks-db configure'."
        )

    sqs_queue_url = file_config.get("sqs_queue_url")
    client = None
    if args.command not in commands_without_bucket:
        client = VectorDBClient(bucket=bucket, region=region, sqs_queue_url=sqs_queue_url, dynamodb_table_name=table_name, wait_timeout=wait_timeout)

    def _fmt_result(r):
        if len(r) == 3:
            return f"(id={r[0]}, dist={r[1]:.4f}, src={r[2]})"
        return str(r)

    # ── setup ────────────────────────────────────────────────
    if args.command == "setup":
        overrides = {}
        if args.runtime_name:
            overrides["runtime_name"] = args.runtime_name
        if args.function_name:
            overrides["lambda_function_name"] = args.function_name
        if args.table_name:
            overrides["dynamodb_table_name"] = args.table_name
        if args.layer_name:
            overrides["layer_name"] = args.layer_name
        if args.role_name:
            overrides["lambda_role_name"] = args.role_name
        if args.threshold:
            overrides["threshold_size_bytes"] = args.threshold
        use_s3express = args.s3express or is_s3express_bucket(args.bucket)
        use_sqs = args.sqs or use_s3express
        if use_s3express:
            overrides["use_s3express"] = True
        if use_sqs:
            overrides["sqs_use_sqs"] = True
        run_setup(
            s3_bucket=args.bucket,
            config_overrides=overrides,
            create_vector_table=not args.skip_vector_table,
            build_runtime=not args.skip_runtime,
        )
        region = args.region or os.getenv("AWS_DEFAULT_REGION") or boto3.Session().region_name or "us-east-1"
        CONFIG_DIR.mkdir(exist_ok=True)
        function_name = args.function_name or "blocksdb-autoindexer-default"
        config_data = {"bucket": args.bucket, "region": region, "lambda_function_name": function_name}
        if args.table_name:
            config_data["dynamodb_table_name"] = args.table_name
        # the wait is configure's to save; setup only carries it over
        wait_timeout = load_backend_config().get("wait_timeout")
        if wait_timeout is not None:
            config_data["wait_timeout"] = wait_timeout
        if use_s3express:
            config_data["s3express"] = True
            az = parse_express_az(args.bucket)
            if az:
                config_data["express_az"] = az
        if use_sqs:
            config_data["use_sqs"] = True
            sqs_queue_name = overrides.get("sqs_queue_name", f"blocksdb-pending-{args.bucket.replace('.', '-')}")
            try:
                sqs = boto3.client("sqs", region_name=region)
                sqs_queue_url = sqs.get_queue_url(QueueName=sqs_queue_name)["QueueUrl"]
                config_data["sqs_queue_url"] = sqs_queue_url
            except Exception:
                pass
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f, indent=4)
        print(f"Configuration saved to {CONFIG_FILE}")

    elif args.command == "update-threshold":
        from .infra import update_lambda_threshold
        update_lambda_threshold(
            threshold_bytes=args.threshold_bytes,
            dataset_name=args.dataset,
            s3_bucket=args.bucket
        )

    elif args.command == "refresh-credentials":
        refresh_lithops_credentials()

    elif args.command == "deploy-lambda":
        from .infra import deploy_lambda_code
        deploy_lambda_code(args.function)

    elif args.command == "configure":
        CONFIG_DIR.mkdir(exist_ok=True)
        use_s3express = is_s3express_bucket(args.bucket)
        use_sqs = args.sqs or use_s3express
        config_data = {"bucket": args.bucket, "region": args.region}
        if args.table_name:
            config_data["dynamodb_table_name"] = args.table_name
        if args.wait_timeout is not None:
            config_data["wait_timeout"] = args.wait_timeout
        if use_s3express:
            config_data["s3express"] = True
            az = parse_express_az(args.bucket)
            if az:
                config_data["express_az"] = az
        if use_sqs:
            config_data["use_sqs"] = True
            sqs_queue_name = f"blocksdb-pending-{args.bucket.replace('.', '-')}"
            try:
                sqs = boto3.client("sqs", region_name=args.region)
                sqs_queue_url = sqs.get_queue_url(QueueName=sqs_queue_name)["QueueUrl"]
                config_data["sqs_queue_url"] = sqs_queue_url
            except Exception:
                pass
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f, indent=4)
        print(f"Configuration saved to {CONFIG_FILE}")

    # ── initialize-database ───────────────────────────────────
    elif args.command == "initialize-database" and args.format == "parquet":
        from .indexing.prepare import expand_sources

        with open(args.config) as f:
            config = json.load(f)

        def list_s3(bucket_name, prefix):
            pages = client.s3.get_paginator("list_objects_v2").paginate(Bucket=bucket_name, Prefix=prefix)
            return [item["Key"] for page in pages for item in page.get("Contents", [])]

        sources = expand_sources(args.source, list_s3, files="*.parquet" if args.files is None else args.files)
        print(f"\n=== Building index for '{args.name}' from {len(sources)} parquet file(s) ===")
        times = client.index_parquet_dataset(args.name, sources, config, replace=args.replace)
        print(f"Index built successfully.")
        print(f"Timing: {json.dumps(times, indent=2)}")

    elif args.command == "initialize-database":
        print(f"\n=== Uploading dataset '{args.name}' ===")

        csv_blocks = None
        if args.build_local:
            t0 = time.time()
            csv_blocks = build_csv_blocks_from_local(args.source)
            local_build_time = time.time() - t0
            blocks, last_vid = csv_blocks
            print(f"Built {len(blocks)} csv_blocks from local file (last_vid={last_vid}) in {local_build_time:.3f}s.")

        t0 = time.time()
        client.create_dataset(args.name, args.source)
        upload_time = time.time() - t0
        print(f"Dataset uploaded in {upload_time:.3f}s.")

        with open(args.config) as f:
            config = json.load(f)

        if args.csv_block_size is not None:
            config["csv_block_size"] = args.csv_block_size

        print(f"\n=== Building index ===")
        times = client.index_dataset(
            dataset_name=args.name,
            config=config,
            num_workers=16 if args.workers is None else args.workers,
            setup_auto_indexer=not args.skip_auto_indexer,
            csv_blocks=csv_blocks
        )
        times["upload_dataset"] = upload_time
        print(f"Index built successfully.")
        print(f"Timing: {json.dumps(times, indent=2)}")

        use_s3express = is_s3express_bucket(bucket) if bucket else False
        if not args.no_update_threshold:
            print(f"\n=== Updating auto-indexer threshold ===")
            from .infra import update_lambda_threshold
            update_lambda_threshold(
                threshold_bytes=None,
                dataset_name=args.name,
                s3_bucket=bucket
            )

    # ── put ───────────────────────────────────────────────────
    elif args.command == "put":
        client.refuse_put_on_parquet(args.name)
        tags = json.loads(args.tags) if args.tags else None
        if tags:
            print(f"Batch tags: {tags}")

        print(f"\n=== Putting vectors into '{args.name}' ===")

        if args.single:
            with open(args.csv_path) as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or not row[0].strip():
                        continue
                    try:
                        vec_id = int(row[0])
                        vec = [float(x) for x in row[1].strip().split() if x]
                        pvt = None
                        if len(row) > 2 and row[2].strip():
                            try:
                                pvt = json.loads(row[2])
                            except (json.JSONDecodeError, ValueError):
                                pass
                        key = client.tracker.put_vector(args.name, vec_id, vec, tags=tags, per_vector_tags=pvt)
                        print(f"  Uploaded vector {vec_id} -> {key}")
                    except (ValueError, IndexError) as e:
                        print(f"  Skipping invalid line: {row} ({e})")
        else:
            has_3rd_col = False
            with open(args.csv_path, "r", newline="") as f:
                import csv as csv_mod
                first = csv_mod.reader(f)
                for r in first:
                    if len(r) > 2 and r[2].strip():
                        has_3rd_col = True
                    break

            if has_3rd_col:
                vectors_with_tags = load_vectors_with_ids_and_tags_from_csv(args.csv_path)
                vectors = [(v[0], v[1]) for v in vectors_with_tags]
                per_vector_tags = [v[2] for v in vectors_with_tags]
            else:
                vectors = load_vectors_with_ids_from_csv(args.csv_path)
                per_vector_tags = None

            if not vectors:
                print("No vectors found in file.")
                return

            if len(vectors) == 1:
                pvt = per_vector_tags[0] if per_vector_tags else None
                vec_id, vec = vectors[0]
                key = client.tracker.put_vector(args.name, vec_id, vec, tags=tags, per_vector_tags=pvt)
                print(f"Uploaded 1 vector -> {key}")
            else:
                key = client.tracker.put_vectors(args.name, vectors, tags=tags, per_vector_tags=per_vector_tags)
                print(f"Uploaded {len(vectors)} vectors -> {key}")

        print(f"\nVectors added to pending. Use 'blocks-db query {args.name}' to search them.")

    # ── query ─────────────────────────────────────────────────
    elif args.command == "query":
        print(f"\n=== Querying '{args.name}' ===")

        hybrid = not args.indexed_only
        filter_tags = json.loads(args.filter) if args.filter else None
        filter_mode = args.filter_mode or "post"
        if filter_tags:
            print(f"Filter tags: {filter_tags}")
        if filter_mode != "post":
            print(f"Filter mode: {filter_mode}")

        if args.vector:
            vector = [float(x) for x in args.vector.split()]
            results, times = client.query(args.name, vector, k=args.k, hybrid=hybrid, batch_size=args.batch_size, filter_tags=filter_tags, filter_mode=filter_mode)
            print(f"Query: {vector[:5]}...")
            print(f"Results: {[_fmt_result(r) for r in results[:args.k]]}")
        elif args.file:
            results, times = client.query_from_file(args.name, args.file, hybrid=hybrid, k=args.k, batch_size=args.batch_size, filter_tags=filter_tags, filter_mode=filter_mode)
            print(f"Results ({len(results)} queries):")
            for i, res in enumerate(results[:5]):
                print(f"  Query {i}: {[_fmt_result(r) for r in res[:args.k]]}")
            if len(results) > 5:
                print(f"  ... and {len(results) - 5} more queries")

        if args.batch_size:
            print(f"Batch size: {args.batch_size} (override)")
        print(f"\nSearch mode: {'hybrid (indexed + pending)' if hybrid else 'indexed only'}")
        if filter_tags:
            print(f"Filter tags: {filter_tags}")
        if filter_mode != "post":
            print(f"Filter mode: {filter_mode}")
        if "error" not in times:
            print(f"Query times: {json.dumps(times, indent=2)}")

    # ── get-by-tags ───────────────────────────────────────────
    elif args.command == "get-by-tags":
        import json as _json
        filter_tags = _json.loads(args.filter) if args.filter else {}
        if not filter_tags:
            print("No filter provided.")
            return
        client.refuse_tags_on_parquet(args.name)

        print(f"\n=== Getting vectors by tags for '{args.name}' ===")
        print(f"Filter: {filter_tags}")

        s3 = boto3.client("s3")
        prefix = f"indexes/{args.name}/blocks/"
        matching_ids = None

        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if not key.endswith("_tags.json") or key.endswith("_reverse_tags.json"):
                        continue
                    try:
                        raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
                        tags_data = _json.loads(raw)
                        for vid_str, vt in tags_data.items():
                            if all(vt.get(k) == v for k, v in filter_tags.items()):
                                if matching_ids is None:
                                    matching_ids = set()
                                matching_ids.add(int(vid_str))
                    except Exception as e:
                        print(f"  Error reading {key}: {e}")
        except Exception as e:
            print(f"  Error listing centroids: {e}")

        if matching_ids:
            sorted_ids = sorted(matching_ids)[:args.limit]
            print(f"\nFound {len(matching_ids)} matching vectors (showing {len(sorted_ids)}):")
            for vid in sorted_ids:
                print(f"  {vid}")
        else:
            print("No matching vectors found.")

    # ── delete-dataset ────────────────────────────────────────
    elif args.command == "delete-dataset":
        if not args.yes:
            confirm = input(f"Delete dataset '{args.name}' and ALL its data? [y/N] ")
            if confirm.lower() != "y":
                print("Aborted.")
                return
        client.delete_dataset(args.name)

    # ── status ────────────────────────────────────────────────
    elif args.command == "status":
        print(f"\n=== Status for '{args.name}' ===\n")

        indexes = client.list_indexes(args.name)
        parquet = client.parquet_config(args.name, indexes)
        if parquet is not None:
            # an immutable index has nothing pending, and its counts come
            # from the build: the id counter was seeded with the source
            # rows, rejected ones included
            print(f"Dataset: {args.name}")
            print("  Indexed: YES")
            print("  Source format: parquet")
            print(f"  Indexed vectors: {parquet['num_vectors']}")
            print(f"  Rejected rows: {parquet['rejected']}")
            if args.verbose:
                print(f"  Blocks: {parquet['num_index']}")
                print(f"  Source files: {len(parquet['source_keys'])}")
            return

        has_index = len(indexes) > 0
        has_pending = client.has_pending_vectors(args.name)
        
        # Use DynamoDB counter for count (faster)
        indexed_count = client.get_indexed_count(args.name)
        if indexed_count == 0:
            # Fall back to S3 tracking
            indexed_ids = client.get_indexed_ids(args.name)
            indexed_count = len(indexed_ids)

        print(f"Dataset: {args.name}")
        print(f"  Indexed: {'YES' if has_index else '✗'}")
        print(f"  Pending vectors: {'YES' if has_pending else 'NO' if has_index else 'Status failed'}")
        print(f"  Indexed vectors: {indexed_count}")

        if has_pending:
            pending_files = client.tracker.get_pending_files(args.name)
            print(f"  Pending files: {len(pending_files)}")

        if args.verbose:
            indexed_ids = client.get_indexed_ids(args.name)
            print(f"\n  Indexed IDs sample: {sorted(indexed_ids)[:10]}...")

            if has_pending:
                pending = client.get_pending_vectors(args.name)
                print(f"\n  Pending vectors: {len(pending)}")
                for vid, vec in pending[:3]:
                    print(f"    ID {vid}: {vec[:5]}...")
                if len(pending) > 3:
                    print(f"    ... and {len(pending) - 3} more")

    # ── get ───────────────────────────────────────────────────
    elif args.command == "get":
        if args.pending:
            pending = client.get_pending_vectors(args.name)
            if pending:
                limit = args.limit or 10
                print(f"\nPending vectors for '{args.name}' ({len(pending)}):")
                for vid, vec in pending[:limit]:
                    print(f"  ID {vid}: {vec[:5]}...")
                if len(pending) > limit:
                    print(f"  ... and {len(pending) - limit} more")
            else:
                print(f"No pending vectors for '{args.name}'.")
        elif args.ids:
            vectors = client.get_vectors(args.name, args.ids)
            if not vectors:
                print("No vectors found for the given IDs.")
                return

            has_duplicates = False
            for vid in args.ids:
                if vid in vectors and isinstance(vectors[vid], list):
                    has_duplicates = True
                    break

            if has_duplicates:
                print(f"\nFound {len(vectors)} vector(s), {sum(1 for v in vectors.values() if isinstance(v, list))} with duplicates:")
            else:
                print(f"\nFound {len(vectors)} vector(s):")

            for vid in args.ids:
                if vid not in vectors:
                    print(f"  ID {vid}: NOT FOUND")
                    continue

                result = vectors[vid]

                if isinstance(result, list):
                    print(f"  ID {vid}: [DUPLICATES FOUND]")
                    for i, entry in enumerate(result):
                        vec_preview = entry["vector"][:5]
                        source = entry["source"]
                        file_info = f" ({entry['file']})" if source == "pending" else ""
                        print(f"    [{i+1}] {vec_preview}... ({source}{file_info})")
                else:
                    vec_preview = result["vector"][:5]
                    source = result["source"]
                    print(f"  ID {vid}: {vec_preview}... ({source})")
        elif args.limit:
            vectors = client.list_vectors(args.name, limit=args.limit)
            if vectors:
                print(f"\nFirst {len(vectors)} vectors in '{args.name}':")
                for vid, vec in vectors.items():
                    print(f"  ID {vid}: {vec[:5]}...")
            else:
                print(f"No vectors in '{args.name}'.")
        else:
            print("Specify vector IDs, --limit, or --pending")


if __name__ == "__main__":
    main()
