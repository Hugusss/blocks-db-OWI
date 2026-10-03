"""The base of the named errors a user can correct.

The named errors that Blocks-DB raises for a request the data cannot
support, a source that cannot be read, or a cloud service that refused a
write derive from :class:`BlocksDBError` (the Errors table of
docs/python-client.md lists them), and also from the builtin each extends,
so code that catches ``ValueError``, ``RuntimeError`` or ``TimeoutError``
keeps working. The command line ends them with their message; any other
exception keeps its traceback.
"""


class BlocksDBError(Exception):
    """An error a user can correct; its message says what to change."""
