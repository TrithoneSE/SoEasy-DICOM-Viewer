"""Anonymous display (View → Hide patient identity, Ctrl+Shift+H): patient names, birth dates and study dates /
times are hidden everywhere on screen — image overlay, window titles, series list, tooltips — for screenshots,
teaching and presentations. The DICOM files are not changed, and DICOM tags shown on request still show them."""

from __future__ import annotations

enabled = False
ANONYMOUS = "Anonymous"


def name(raw):
    """Patient name for display ('^' separators as spaces)."""
    if enabled:
        return ANONYMOUS
    return (str(raw or "") or "(no name)").replace("^", " ").strip()


def date(text):
    """A formatted date / time, or '' while the identity is hidden."""
    return "" if enabled else text
