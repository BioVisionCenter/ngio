"""Custom warning classes for the ngio package."""


class NgioDeprecationWarning(DeprecationWarning):
    """Warning for deprecated ngio API usage."""


class NgioUserWarning(UserWarning):
    """Warning for ngio user-facing behavioral issues."""


class NgioFutureWarning(FutureWarning):
    """Warning that a default is going to change under a caller who did not pick.

    `FutureWarning` rather than `DeprecationWarning` on purpose: nothing is
    being removed, and Python hides `DeprecationWarning` from end users by
    default — which is the wrong audience to hide a silent behaviour change
    from.
    """
