import boto3
import orjson
from datetime import datetime
from typing import List, Tuple, Set, Optional
from boto3.dynamodb.conditions import Key
import tempfile
import os
import csv
import io
import json
import time

from ..errors import BlocksDBError
from .s3_client import s3


class CounterUnavailable(BlocksDBError, RuntimeError):
    """The id counter of a dataset could not be written."""


class VectorIndexTracker:
    DYNAMODB_TABLE_NAME = "BlocksDB-default"

    def __init__(self, bucket: str, region: str = None, sqs_queue_url: str = None, table_name: str = None):
        self.bucket = bucket
        self.sqs_queue_url = sqs_queue_url
        # the default keeps existing deployments working; an account shared by
        # several deployments gives each one its own table
        self.table_name = table_name or self.DYNAMODB_TABLE_NAME
        if region:
            self.dynamodb = boto3.resource("dynamodb", region_name=region)
            self.s3 = boto3.client("s3", region_name=region)
        else:
            self.dynamodb = boto3.resource("dynamodb")
            self.s3 = boto3.client("s3")
        self.table = self.dynamodb.Table(self.table_name)
        if sqs_queue_url:
            self.sqs = boto3.client("sqs", region_name=region) if region else boto3.client("sqs")

    def get_indexed_files_key(self, dataset_name: str) -> str:
        return f"indexed_files_{dataset_name}.json"

    def get_indexed_files(self, dataset_name: str) -> Set[str]:
        indexed_key = self.get_indexed_files_key(dataset_name)
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=indexed_key)
            return set(orjson.loads(obj["Body"].read()))
        except self.s3.exceptions.NoSuchKey:
            return set()

    def get_pending_prefix(self, dataset_name: str) -> str:
        return f"pending/{dataset_name}/"

    def get_pending_file_key(self, dataset_name: str, file_id: str = None) -> str:
        if file_id is None:
            file_id = str(int(time.time() * 1000))
        return f"pending/{dataset_name}/{file_id}.csv"

    def get_indexed_ids_key(self, dataset_name: str) -> str:
        return f"tracking/indexed_ids_{dataset_name}.json"

    def initialize_next_id(self, dataset_name: str, next_id: int, strict: bool = False):
        """Initialize the next available ID counter in DynamoDB for a dataset.

        By default a failed write is only printed. With ``strict`` it raises,
        naming the table and region: a build that relies on the counter must
        not go ahead without it.
        """
        try:
            self.table.put_item(Item={
                "centroid_id": f"{dataset_name}_ID_TRACKER",
                "sk": "META",
                "next_id": next_id,
                "dataset": dataset_name
            })
            print(f"Initialized next_id={next_id} for dataset '{dataset_name}' in DynamoDB")
        except Exception as e:
            if strict:
                region = self.dynamodb.meta.client.meta.region_name
                raise CounterUnavailable(
                    f"cannot seed the id counter of '{dataset_name}' in DynamoDB table"
                    f" '{self.table_name}' ({region}): {e}"
                ) from e
            print(f"Error initializing next_id in DynamoDB: {e}")

    def get_next_id_atomic(self, dataset_name: str, count: int) -> int:
        """Atomically get and reserve the next `count` IDs. Returns the starting ID."""
        try:
            response = self.table.update_item(
                Key={"centroid_id": f"{dataset_name}_ID_TRACKER", "sk": "META"},
                UpdateExpression="SET next_id = if_not_exists(next_id, :zero) + :inc",
                ExpressionAttributeValues={":inc": count, ":zero": 0},
                ReturnValues="ALL_NEW"
            )
            # Return the starting ID (value after increment minus count)
            return response["Attributes"]["next_id"] - count
        except Exception as e:
            print(f"Error getting next ID atomically: {e}")
            raise

    def get_next_id(self, dataset_name: str) -> int:
        """Get the next available ID from DynamoDB counter."""
        try:
            response = self.table.get_item(
                Key={"centroid_id": f"{dataset_name}_ID_TRACKER", "sk": "META"}
            )
            item = response.get("Item", {})
            return int(item.get("next_id", 0))
        except Exception as e:
            print(f"Error reading next_id from DynamoDB: {e}")
            return 0

    def put_vectors(self, dataset_name: str, vectors: List[Tuple[int, List[float]]], create_file: bool = True, tags: dict = None, per_vector_tags: List[Optional[dict]] = None) -> str:
        if not vectors:
            return None
        file_id = str(int(time.time() * 1000))
        key = self.get_pending_file_key(dataset_name, file_id)

        has_per_vector_tags = per_vector_tags is not None and any(t is not None for t in per_vector_tags)
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        if has_per_vector_tags:
            writer.writerow(["id", "vector", "tags"])
            for i, (vec_id, vec) in enumerate(vectors):
                vec_str = " ".join(str(x) for x in vec)
                t = per_vector_tags[i] if i < len(per_vector_tags) else None
                if t:
                    writer.writerow([vec_id, vec_str, json.dumps(t)])
                else:
                    writer.writerow([vec_id, vec_str])
        else:
            writer.writerow(["id", "vector"])
            for vec_id, vec in vectors:
                vec_str = " ".join(str(x) for x in vec)
                writer.writerow([vec_id, vec_str])
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        extra_args = {}
        if tags:
            extra_args["Metadata"] = {"tags": json.dumps(tags)}
        s3.put_object(Bucket=self.bucket, Key=key, Body=csv_bytes, **extra_args)

        if self.sqs_queue_url:
            try:
                self.sqs.send_message(
                    QueueUrl=self.sqs_queue_url,
                    MessageBody=json.dumps({
                        "bucket": self.bucket,
                        "key": key,
                        "file_size": len(csv_bytes),
                        "tags": tags,
                    })
                )
            except Exception as e:
                print(f"SQS notification error (non-fatal): {e}")

        new_ids = [v[0] for v in vectors]
        self._update_pending_tracking(dataset_name, new_ids, key, tags)

        return key

    def put_vector(self, dataset_name: str, vector_id: int, vector: List[float], tags: dict = None) -> str:
        return self.put_vectors(dataset_name, [(vector_id, vector)], tags=tags)

    def _update_pending_tracking(self, dataset_name: str, vector_ids: List[int], file_key: str, tags: dict = None):
        try:
            item = {
                "centroid_id": "PENDING",
                "sk": f"FILE#{file_key}",
                "dataset": dataset_name,
                "file_key": file_key,
                "vector_ids": vector_ids[:1000] if vector_ids else [],
                "timestamp": int(time.time())
            }
            if tags:
                item["tags"] = json.dumps(tags)
            self.table.put_item(Item=item)
        except Exception as e:
            print(f"DynamoDB tracking error (non-fatal): {e}")

    def get_pending_file_ids(self, dataset_name: str) -> List[str]:
        try:
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('centroid_id').eq('PENDING')
            )
            return [item['file_key'] for item in response.get('Items', []) if item.get('dataset') == dataset_name]
        except Exception:
            return []

    def get_pending_tracking_items(self, dataset_name: str) -> List[dict]:
        try:
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('centroid_id').eq('PENDING')
            )
            return [item for item in response.get('Items', []) if item.get('dataset') == dataset_name]
        except Exception:
            return []

    def clear_pending_file_tracking(self, dataset_name: str):
        items = self.get_pending_tracking_items(dataset_name)
        for item in items:
            try:
                self.table.delete_item(Key={"centroid_id": "PENDING", "sk": f"FILE#{item['file_key']}"})
            except:
                pass

    def get_pending_files(self, dataset_name: str) -> List[str]:
        prefix = self.get_pending_prefix(dataset_name)
        indexed_files = self.get_indexed_files(dataset_name)
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            if "Contents" not in response:
                return []
            files = []
            for obj in response["Contents"]:
                key = obj["Key"]
                if key.endswith(".csv") and key != prefix and obj.get("Size", 0) > 0 and key not in indexed_files:
                    files.append(key)
            return files
        except Exception:
            return []

    def has_pending_vectors(self, dataset_name: str) -> bool:
        files = self.get_pending_files(dataset_name)
        return len(files) > 0

    def get_pending_vectors(self, dataset_name: str) -> List[Tuple[int, List[float]]]:
        vectors = []
        files = self.get_pending_files(dataset_name)
        for key in files:
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=key)
                for raw_line in obj["Body"].iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode()
                    if line.startswith("id,"):
                        continue
                    parts = line.split(",", 2)
                    if len(parts) >= 2:
                        try:
                            vec_id = int(parts[0])
                            vec = [float(x) for x in parts[1].strip().split() if x]
                            vectors.append((vec_id, vec))
                        except ValueError:
                            continue
            except Exception as e:
                print(f"Error reading {key}: {e}")
        return vectors

    def create_indexed_tracking(self, dataset_name: str, indexed_ids: List[int]):
        indexed_key = self.get_indexed_ids_key(dataset_name)
        self.s3.put_object(Bucket=self.bucket, Key=indexed_key, Body=orjson.dumps(indexed_ids))

    def mark_vectors_indexed(self, dataset_name: str, indexed_ids: List[int]):
        existing = self.get_indexed_ids(dataset_name)
        existing.update(indexed_ids)
        self.create_indexed_tracking(dataset_name, list(existing))

    def get_indexed_ids(self, dataset_name: str) -> Set[int]:
        # Try DynamoDB counter first for total count, fall back to S3 for actual IDs
        next_id = self.get_next_id(dataset_name)
        if next_id > 0:
            # Return a set with just the count info (for len() to work)
            # This is a hack for backward compatibility with status command
            return set(range(next_id))
        
        # Fall back to S3
        indexed_key = self.get_indexed_ids_key(dataset_name)
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=indexed_key)
            return set(orjson.loads(obj["Body"].read()))
        except self.s3.exceptions.NoSuchKey:
            return set()

    def is_indexed(self, dataset_name: str, vector_id: int) -> bool:
        indexed_ids = self.get_indexed_ids(dataset_name)
        return vector_id in indexed_ids

    def clear_pending(self, dataset_name: str):
        files = self.get_pending_files(dataset_name)
        if files:
            objects = [{"Key": k} for k in files]
            self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": objects})
        self.clear_pending_file_tracking(dataset_name)

    def _delete_ddb_dataset_items(self, dataset_name: str):
        """Delete all DynamoDB items belonging to a dataset."""
        try:
            self.table.delete_item(
                Key={"centroid_id": f"{dataset_name}_CONFIG", "sk": "META"}
            )
        except:
            pass

        try:
            self.table.delete_item(
                Key={"centroid_id": f"{dataset_name}_ID_TRACKER", "sk": "META"}
            )
        except:
            pass

        try:
            response = self.table.query(
                KeyConditionExpression=(
                    boto3.dynamodb.conditions.Key('centroid_id').eq(f"DATASET#{dataset_name}")
                )
            )
            for item in response.get("Items", []):
                try:
                    self.table.delete_item(
                        Key={"centroid_id": item["centroid_id"], "sk": item["sk"]}
                    )
                except:
                    pass
        except:
            pass

        try:
            response = self.table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('centroid_id').begins_with(f"{dataset_name}#")
            )
            for item in response.get("Items", []):
                try:
                    self.table.delete_item(
                        Key={"centroid_id": item["centroid_id"], "sk": item["sk"]}
                    )
                except:
                    pass
        except:
            pass

    def delete_tracking(self, dataset_name: str):
        files = self.get_pending_files(dataset_name)
        if files:
            objects = [{"Key": k} for k in files]
            self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": objects})
        self.clear_pending_file_tracking(dataset_name)
        indexed_key = self.get_indexed_ids_key(dataset_name)
        indexed_files_key = self.get_indexed_files_key(dataset_name)
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=indexed_key)
            self.s3.delete_object(Bucket=self.bucket, Key=indexed_files_key)
        except:
            pass
        self._delete_ddb_dataset_items(dataset_name)