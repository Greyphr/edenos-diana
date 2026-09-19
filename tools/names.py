"""Validation for names used as filesystem path components.

Owner names (memory data dirs) and voice profile names
(identity/voiceprints/) are substituted directly into os.path.join()
calls. A careless or hostile value like "../x" would escape the intended
data directory; rejecting anything outside a strict, conservative charset
before any join is the entire defense.
"""

import re

# Safe as a single path component: alphanumerics, underscore, hyphen,
# 1-64 characters. No separators, dots, spaces, or control characters.
NAME_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
_NAME_RE = re.compile(NAME_PATTERN)


def validate_name(name: str, *, context: str = "name") -> str:
    """Return ``name`` when safe to use as one path component, else raise.

    Raises ``ValueError`` with the offending value so a caller can see
    exactly what was rejected, e.g. ``"../x"`` trying to escape a data dir.
    """
    if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
        raise ValueError(
            f"Invalid {context} {name!r}: must match {NAME_PATTERN} "
            "(letters, digits, '_', '-'; 1-64 chars)"
        )
    return name