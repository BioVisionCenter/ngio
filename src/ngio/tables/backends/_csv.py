from ngio.tables.backends._py_arrow_backends import PyArrowBackend


class CsvTableBackend(PyArrowBackend):
    """A class to load and write small tables in CSV format."""

    def __init__(
        self,
    ):
        """Initialize the CsvTableBackend."""
        super().__init__(
            table_name="table.csv",
            table_format="csv",
        )

    @staticmethod
    def backend_name() -> str:
        """Return the name of the backend."""
        return "csv"
