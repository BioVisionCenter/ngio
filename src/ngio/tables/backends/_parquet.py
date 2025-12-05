from ngio.tables.backends._py_arrow_backends import PyArrowBackend


class ParquetTableBackend(PyArrowBackend):
    """A class to load and write small tables in Parquet format."""

    def __init__(
        self,
    ):
        """Initialize the ParquetTableBackend."""
        super().__init__(
            table_name="table.parquet",
            table_format="parquet",
        )

    @staticmethod
    def backend_name() -> str:
        """Return the name of the backend."""
        return "parquet"
