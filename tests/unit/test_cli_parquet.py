"""The command line's parquet build: which files it hands to the client."""

import json

import pytest

from vectordb import cli
from vectordb.implementations.blocks.initialize import BlockTooSmall
from vectordb.indexing.planner import PlanError
from vectordb.utils.parquet import ParquetSourceError
from vectordb.utils.vector_tracking import CounterUnavailable
from vectordb.utils.waiting import FunctionsTimedOut


def test_files_narrows_a_directory_source(tmp_path, monkeypatch):
    day = tmp_path / "day" / "language=spa"
    day.mkdir(parents=True)
    for name in ("metadata_0_embeddings.parquet", "metadata_0_records.parquet"):
        (day / name).write_bytes(b"")
    config = tmp_path / "index.json"
    config.write_text(json.dumps({"num_index": 1}))
    handed = {}

    class RecordingClient:
        def __init__(self, **kwargs):
            pass

        def index_parquet_dataset(self, name, sources, config, replace=False):
            handed.update(name=name, sources=sources, replace=replace)
            return {}

    monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli, "CONFIG_FILE", tmp_path / "backend_config.json")
    monkeypatch.setattr(cli, "VectorDBClient", RecordingClient)
    monkeypatch.setattr("sys.argv", ["blocks-db", "--bucket", "b", "initialize-database", "ds", str(tmp_path / "day"),
                                     "--format", "parquet", "--files", "*_embeddings.parquet", "--config", str(config)])
    cli.main()
    assert handed == {"name": "ds", "sources": [str(day / "metadata_0_embeddings.parquet")], "replace": False}
    monkeypatch.setattr("sys.argv", ["blocks-db", "--bucket", "b", "initialize-database", "ds", str(tmp_path / "day"),
                                     "--format", "parquet", "--config", str(config), "--replace"])
    cli.main()
    assert handed["replace"] is True


def build_that_raises(error, tmp_path, monkeypatch):
    """Run a parquet build whose client raises, and give back the exit code."""

    class RefusingClient:
        def __init__(self, **kwargs):
            pass

        def index_parquet_dataset(self, name, sources, config, replace=False):
            raise error

    (tmp_path / "a.parquet").write_bytes(b"")
    config = tmp_path / "index.json"
    config.write_text("{}")
    monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli, "CONFIG_FILE", tmp_path / "backend_config.json")
    monkeypatch.setattr(cli, "VectorDBClient", RefusingClient)
    monkeypatch.setattr("sys.argv", ["blocks-db", "--bucket", "b", "initialize-database", "ds", str(tmp_path / "a.parquet"),
                                     "--format", "parquet", "--config", str(config)])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    return str(stopped.value.code)


def test_an_existing_index_ends_the_command_with_the_reason(tmp_path, monkeypatch):
    from vectordb.client import IndexExists

    error = IndexExists("'ds' already holds an index; pass --replace to delete it and build again")
    assert build_that_raises(error, tmp_path, monkeypatch).startswith("Error: 'ds' already holds an index")


@pytest.mark.parametrize("error", [
    PlanError("k (IVF lists per block) must be declared"),
    ParquetSourceError("day/metadata_0_records.parquet: no known vector dialect"),
    BlockTooSmall("block 0 kept 12 rows for 1650 IVF lists"),
    CounterUnavailable("cannot seed the id counter of 'ds' in DynamoDB"),
    FunctionsTimedOut("3 of 8 functions never started, and no function started or finished in the last 960 s"),
], ids=["plan", "source", "block", "counter", "timeout"])
def test_a_build_that_cannot_go_on_says_why_instead_of_a_traceback(error, tmp_path, monkeypatch):
    # a traceback for something the user can correct reads like a crash
    assert build_that_raises(error, tmp_path, monkeypatch) == f"Error: {error}"


def test_a_missing_config_file_is_a_message_too(tmp_path, monkeypatch):
    class UnusedClient:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli, "CONFIG_FILE", tmp_path / "backend_config.json")
    monkeypatch.setattr(cli, "VectorDBClient", UnusedClient)
    (tmp_path / "a.parquet").write_bytes(b"")
    monkeypatch.setattr("sys.argv", ["blocks-db", "--bucket", "b", "initialize-database", "ds", str(tmp_path / "a.parquet"),
                                     "--format", "parquet", "--config", str(tmp_path / "missing.json")])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert str(stopped.value.code).startswith("Error: ")
