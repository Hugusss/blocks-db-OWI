"""Parquet vector sources: the only module that knows the file dialects.

Two dialects are recognized by their column names:

- ``canonical``: ``vector`` (list of float) and, optionally, ``id``
  (integer), the parquet twin of the CSV rows. Any other column is
  ignored.
- ``owi-v2``: ``record_id`` (string), ``chunk_idx`` (integer) and
  ``embedding`` (list of float16): the Open Web Index embeddings files
  as published, consumed without conversion.

:func:`inspect` reads the footer of a file and, when the schema does
not fix the list size, the vector column of its first row group with
rows, so a build can be planned without downloading the rest;
:func:`iter_ranges` reads the ranges of one file in as few storage reads
as the plan allows, and returns dense float32 matrices with the
provenance of every row. Rows whose vector length differs from the
declared dimension are counted as rejected, never padded or truncated,
and the positions of the kept rows are returned so position-based ids
stay stable. ``owi-v2`` vectors come back at unit length, and a row
that cannot be scaled (zero or non-finite) is rejected too.

In both dialects the id of a vector in the index is the position of its
row in the plan; the file's own id (``id`` or ``record_id``) is kept as
provenance and returned by ``VectorDBClient.provenance()``.

Sources are named by URI: ``s3://bucket/key`` or a local path. One S3
filesystem is built per bucket and process, in the region of
``AWS_REGION`` or ``AWS_DEFAULT_REGION`` when either is set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyarrow import fs as pafs

from vectordb.errors import BlocksDBError

CANONICAL = "canonical"
OWI_V2 = "owi-v2"

_VECTOR_COLUMN = {CANONICAL: "vector", OWI_V2: "embedding"}
_REQUIRED = {
    CANONICAL: ("vector",),
    OWI_V2: ("record_id", "chunk_idx", "embedding"),
}

_FILESYSTEMS: dict[str, pafs.FileSystem] = {}


class ParquetSourceError(BlocksDBError, ValueError):
    """A source that cannot be read as vectors; the message says why."""


class EmptyParquetFile(ParquetSourceError):
    """A file with no rows, which a caller may skip instead of failing."""


@dataclass(frozen=True)
class FileInfo:
    """What the footer of one file says, before any data is read."""

    uri: str
    dialect: str
    dimension: int
    row_groups: tuple[int, ...]  # rows per row group, in file order


@dataclass(frozen=True)
class Rows:
    """One row group, or a slice of it, decoded.

    ``positions`` are the row offsets (within the requested slice) of
    the rows kept, so ``id_offset + position`` is the id of a vector
    even when earlier rows were rejected.
    """

    vectors: np.ndarray  # float32, shape (kept, dimension)
    positions: np.ndarray  # int64, shape (kept,)
    rejected: int
    ids: np.ndarray | None = None  # canonical: the file's own integer ids, if it has any
    record_ids: list[str] | None = None  # owi-v2
    chunk_idx: np.ndarray | None = None  # owi-v2

    @property
    def kept(self) -> int:
        return len(self.positions)


def detect_dialect(column_names) -> str:
    names = set(column_names)
    matches = [d for d, required in _REQUIRED.items() if set(required) <= names]
    if len(matches) > 1:
        raise ParquetSourceError(
            f"ambiguous dialect: columns {sorted(names)} satisfy both"
            f" {sorted(matches)}; a source must be one or the other"
        )
    if not matches:
        raise ParquetSourceError(
            "no known vector dialect: columns "
            f"{sorted(names)}; expected {list(_REQUIRED[CANONICAL])}"
            f" (canonical, plus an optional 'id') or {list(_REQUIRED[OWI_V2])} (owi-v2)"
        )
    return matches[0]


def _filesystem(uri: str):
    """The filesystem and path for a URI, one filesystem per bucket."""
    if not uri.startswith("s3://"):
        return pafs.LocalFileSystem(), os.path.abspath(os.path.expanduser(uri))
    path = uri[len("s3://"):]
    bucket = path.split("/", 1)[0]
    filesystem = _FILESYSTEMS.get(bucket)
    if filesystem is None:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        try:
            # with no region declared, pyarrow asks S3 for the bucket's own
            # region, which is a network call on the first open
            filesystem = pafs.S3FileSystem(region=region) if region else pafs.FileSystem.from_uri(uri)[0]
        except (pa.ArrowException, OSError, ValueError) as error:
            raise ParquetSourceError(f"{uri}: cannot reach the store ({error})") from None
        _FILESYSTEMS[bucket] = filesystem
    return filesystem, path


def _open(uri: str):
    """An open parquet reader for ``uri``; every failure is named."""
    filesystem, path = _filesystem(uri)
    try:
        handle = filesystem.open_input_file(path)
        # pre_buffer coalesces the column chunks of a read into few large
        # requests
        return handle, pq.ParquetFile(handle, pre_buffer=True)
    except (pa.ArrowException, OSError, ValueError) as error:
        raise ParquetSourceError(f"{uri}: cannot read ({error})") from None


def inspect(uri: str) -> FileInfo:
    """Description of one file from its footer (plus the vector column
    of the first row group with rows, when the schema does not fix the
    list size).

    Raises :class:`EmptyParquetFile` for a file with no rows.
    """
    handle, reader = _open(uri)
    try:
        metadata = reader.metadata
        row_groups = tuple(
            metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)
        )
        if sum(row_groups) == 0:
            raise EmptyParquetFile(f"{uri}: no rows")
        schema = reader.schema_arrow
        try:
            dialect = detect_dialect(schema.names)
        except ParquetSourceError as error:
            # a directory source stands for many files: say which one
            raise ParquetSourceError(f"{uri}: {error}") from None
        column = _VECTOR_COLUMN[dialect]
        kind = schema.field(column).type
        if pa.types.is_fixed_size_list(kind):
            dimension = kind.list_size
        elif pa.types.is_list(kind) or pa.types.is_large_list(kind):
            dimension = _first_vector_length(uri, reader, column, row_groups)
        else:
            raise ParquetSourceError(
                f"{uri}: column '{column}' is {kind}, not a list of numbers"
            )
        # the decoder casts the values to float32, and would take booleans or
        # digits written as text for coordinates
        element = kind.value_type
        if not (pa.types.is_floating(element) or pa.types.is_integer(element)):
            raise ParquetSourceError(
                f"{uri}: column '{column}' holds {element} values, not numbers"
            )
    except (pa.ArrowException, OSError) as error:
        raise ParquetSourceError(f"{uri}: cannot read ({error})") from None
    finally:
        handle.close()
    return FileInfo(uri=uri, dialect=dialect, dimension=dimension, row_groups=row_groups)


def _first_vector_length(uri: str, reader: pq.ParquetFile, column: str, row_groups) -> int:
    # inspect() refuses a file with no rows before it asks for the dimension
    first = next(index for index, rows in enumerate(row_groups) if rows)
    values = reader.read_row_group(first, columns=[column]).column(column)
    for item in values:
        if item.is_valid:
            return len(item.values)
    raise ParquetSourceError(
        f"{uri}: no vector in the first row group with rows; the dimension"
        " cannot be read from the schema either"
    )


def iter_ranges(uri: str, dimension: int, ranges):
    """Decode several ``(row_group, start, end)`` ranges of one file.

    The file is opened once, and consecutive whole row groups are read in
    one call. Yields one :class:`Rows` per requested range, in order.
    """
    handle, reader = _open(uri)
    try:
        names = reader.schema_arrow.names
        dialect = detect_dialect(names)
        columns = list(_REQUIRED[dialect])
        if dialect == CANONICAL and "id" in names:
            columns.append("id")  # the file's own ids, kept as provenance
        sizes = [
            reader.metadata.row_group(index).num_rows
            for index in range(reader.metadata.num_row_groups)
        ]
        for batch in _batches(list(ranges), sizes):
            if len(batch) == 1 and not _is_whole(batch[0], sizes):
                row_group, start, end = batch[0]
                table = _read_one(uri, reader, row_group, columns)
                yield _decode(uri, dialect, dimension, table, row_group, start, end)
                continue
            groups = [row_group for row_group, _, _ in batch]
            table = _read_many(uri, reader, groups, columns)
            offset = 0
            for row_group, _, _ in batch:
                rows = sizes[row_group]
                yield _decode(uri, dialect, dimension, table.slice(offset, rows), row_group, 0, rows)
                offset += rows
    except (pa.ArrowException, OSError) as error:
        raise ParquetSourceError(f"{uri}: cannot read ({error})") from None
    finally:
        handle.close()


def _is_whole(part, sizes) -> bool:
    row_group, start, end = part
    return start == 0 and (end is None or end == sizes[row_group])


def _batches(ranges, sizes):
    """Runs of consecutive whole row groups, and single sliced ranges."""
    batch: list = []
    for part in ranges:
        whole = _is_whole(part, sizes)
        if whole and batch and _is_whole(batch[-1], sizes) and part[0] == batch[-1][0] + 1:
            batch.append(part)
            continue
        if batch:
            yield batch
        batch = [part]
    if batch:
        yield batch


def _read_one(uri, reader, row_group, columns):
    try:
        return reader.read_row_group(row_group, columns=columns)
    except (pa.ArrowException, IndexError) as error:
        raise ParquetSourceError(
            f"{uri}: cannot read row group {row_group} ({error})"
        ) from None


def _read_many(uri, reader, row_groups, columns):
    try:
        return reader.read_row_groups(row_groups, columns=columns)
    except (pa.ArrowException, IndexError) as error:
        raise ParquetSourceError(
            f"{uri}: cannot read row groups {row_groups[0]}..{row_groups[-1]}"
            f" ({error})"
        ) from None


def _decode(uri, dialect, dimension, table, row_group, start, end) -> Rows:
    if end is None:
        end = table.num_rows
    if not 0 <= start <= end <= table.num_rows:
        raise ParquetSourceError(
            f"{uri}: rows [{start}, {end}) outside row group {row_group}"
            f" of {table.num_rows} rows"
        )
    table = table.slice(start, end - start)

    column = table.column(_VECTOR_COLUMN[dialect])
    lengths = pc.fill_null(pc.list_value_length(column), -1)
    keep = pc.equal(lengths, dimension)
    kept = table.filter(keep)
    positions = np.flatnonzero(keep.to_numpy(zero_copy_only=False)).astype(np.int64)
    rejected = table.num_rows - kept.num_rows

    flat = pc.list_flatten(kept.column(_VECTOR_COLUMN[dialect]))
    vectors = flat.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    vectors = vectors.reshape(kept.num_rows, dimension)

    if dialect == CANONICAL:
        return Rows(
            vectors=vectors,
            positions=positions,
            rejected=rejected,
            ids=_integers(uri, kept, "id") if "id" in kept.column_names else None,
        )
    # scaled to unit length, the L2 distance ranks like the cosine; a
    # vector with a zero or non-finite norm cannot be scaled and is rejected
    norms = np.linalg.norm(vectors, axis=1)
    scalable = np.isfinite(norms) & (norms > 0)
    if not scalable.all():
        kept = kept.filter(pa.array(scalable))
        positions = positions[scalable]
        rejected += int(np.count_nonzero(~scalable))
        vectors, norms = vectors[scalable], norms[scalable]
    vectors = vectors / norms[:, None]
    return Rows(
        vectors=vectors,
        positions=positions,
        rejected=rejected,
        record_ids=_strings(uri, kept, "record_id"),
        chunk_idx=_integers(uri, kept, "chunk_idx"),
    )


def _integers(uri: str, table: pa.Table, name: str) -> np.ndarray:
    """An integer column as int64, refusing nulls rather than turning
    them into a sentinel through float64."""
    column = table.column(name)
    if column.null_count:
        raise ParquetSourceError(
            f"{uri}: column '{name}' has {column.null_count} null values"
            " in a row whose vector is well formed; ids must be present"
        )
    try:
        return column.cast(pa.int64()).to_numpy(zero_copy_only=False)
    except pa.ArrowInvalid as error:
        raise ParquetSourceError(f"{uri}: column '{name}' is not integral ({error})") from None


def _strings(uri: str, table: pa.Table, name: str) -> list[str]:
    column = table.column(name)
    if column.null_count:
        raise ParquetSourceError(
            f"{uri}: column '{name}' has {column.null_count} null values"
            " in a row whose vector is well formed"
        )
    return column.to_pylist()
