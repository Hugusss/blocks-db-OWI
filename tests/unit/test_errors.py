"""Every error a user can correct shares one base and keeps its builtin."""

import pickle

import pytest

from vectordb import cli
from vectordb.client import IndexExists, NoIndex, NotAvailableOnParquet, QueryMismatch
from vectordb.errors import BlocksDBError
from vectordb.implementations.blocks.initialize import BlockTooSmall
from vectordb.indexing.planner import PlanError
from vectordb.utils.parquet import EmptyParquetFile, ParquetSourceError
from vectordb.utils.vector_tracking import CounterUnavailable
from vectordb.utils.waiting import FunctionsTimedOut

BUILTIN = {
    NotAvailableOnParquet: RuntimeError,
    IndexExists: RuntimeError,
    NoIndex: ValueError,
    QueryMismatch: ValueError,
    PlanError: ValueError,
    ParquetSourceError: ValueError,
    EmptyParquetFile: ValueError,
    BlockTooSmall: ValueError,
    CounterUnavailable: RuntimeError,
    FunctionsTimedOut: TimeoutError,
}


@pytest.mark.parametrize("error", list(BUILTIN), ids=lambda error: error.__name__)
def test_each_error_is_a_blocks_db_error_and_still_its_builtin(error):
    assert issubclass(error, BlocksDBError)
    with pytest.raises(BUILTIN[error]):
        raise error("a message")
    # Lithops brings an error raised in a function back to the client by pickle
    copy = pickle.loads(pickle.dumps(error("a message")))
    assert type(copy) is error and str(copy) == "a message"


def test_the_command_line_ends_every_one_with_its_message(tmp_path, monkeypatch):
    class Failing:
        def __init__(self, **kwargs):
            raise PlanError("the plan needs k")

    monkeypatch.setattr(cli, "VectorDBClient", Failing)
    monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli, "CONFIG_FILE", tmp_path / "backend_config.json")
    monkeypatch.setattr("sys.argv", ["blocks-db", "--bucket", "b", "status", "ds"])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == "Error: the plan needs k"


def test_an_error_that_is_not_one_keeps_its_traceback(tmp_path, monkeypatch):
    class Broken:
        def __init__(self, **kwargs):
            raise ValueError("a defect")

    monkeypatch.setattr(cli, "VectorDBClient", Broken)
    monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli, "CONFIG_FILE", tmp_path / "backend_config.json")
    monkeypatch.setattr("sys.argv", ["blocks-db", "--bucket", "b", "status", "ds"])
    with pytest.raises(ValueError, match="a defect"):
        cli.main()


def test_the_base_is_exported_with_the_client():
    import blocks_db

    assert blocks_db.BlocksDBError is BlocksDBError
