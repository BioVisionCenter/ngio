"""Custom warning classes for the ngio package."""


class NgioDeprecationWarning(DeprecationWarning):
    """Warning for deprecated ngio API usage."""


class NgioUserWarning(UserWarning):
    """Warning for ngio user-facing behavioral issues."""
