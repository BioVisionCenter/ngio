"""Custom warning classes for the ngio package."""

from inspect import currentframe
from pathlib import Path


def stacklevel_of_first_caller() -> int:
    """Return the `stacklevel` that points at the first frame outside ngio.

    A fixed level cannot work: the same warning is often reachable through
    different depths of ngio's own code — `get_wells()` reaches its fan-out
    through two frames and `images_paths()` through three — so any constant
    blames ngio's own source for one of them. That matters twice over: the
    reader is told to edit a file they do not own, and the `warnings` module
    keys its deduplication on the reported location, so every caller would
    collapse onto one entry.
    """
    package_root = str(Path(__file__).parents[1])
    frame = currentframe()
    level = 0
    while frame is not None:
        level += 1
        frame = frame.f_back
        if frame is not None and not frame.f_code.co_filename.startswith(package_root):
            return level
    return 2


class NgioDeprecationWarning(DeprecationWarning):
    """Warning for deprecated ngio API usage."""


class NgioUserWarning(UserWarning):
    """Warning for ngio user-facing behavioural issues."""


class NgioFutureWarning(FutureWarning):
    """Warning that a default is going to change under a caller who did not pick.

    `FutureWarning` rather than `DeprecationWarning` on purpose: nothing is
    being removed, and Python hides `DeprecationWarning` from end users by
    default — which is the wrong audience to hide a silent behaviour change
    from.
    """
