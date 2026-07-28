"""Transcript normalization shared by the build and the evaluation.

The training CSV stores prosody-annotated transcripts:

    희재야.|||HL 육박 칠일 길다.|||LH 회장이란 책임감에 짓눌리면|||LHL 못 버텨.||||M

Gemini's transcripts (the query side at inference) carry no such markers. If
the index keeps them and the query does not, situational retrieval compares
marked-up documents against clean queries and the similarity is dominated by
that formatting difference. Both sides go through this function.

Only four tone tags occur in the data: HL, LH, LHL, M.
"""

from __future__ import annotations

import re

# Two or more pipes, optionally followed by one tone tag. Longest tag first so
# "LHL" is not partially consumed as "LH".
_MARKER = re.compile(r"\|{2,}\s*(?:LHL|HL|LH|M)?")
_SPACES = re.compile(r"\s+")


def normalize_transcript(text: str) -> str:
    """Strip prosody markers and collapse whitespace."""
    if not text:
        return ""
    return _SPACES.sub(" ", _MARKER.sub(" ", text)).strip()
