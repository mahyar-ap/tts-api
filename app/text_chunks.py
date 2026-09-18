"""Shared text chunking for streaming synthesis."""

from __future__ import annotations

_SENTENCE_ENDINGS = frozenset(".!?؟…")
_SENTENCE_CLOSERS = frozenset("\"'»”)]}")
_CLAUSE_BOUNDARIES = frozenset(",،;؛:")


def split_text_for_streaming(text: str, max_chunk_length: int = 300) -> list[str]:
    """Split text into bounded chunks, preferring complete sentences.

    Packs sentences together up to ``max_chunk_length`` (job / batch path).
    """

    return _split_text(text, max_chunk_length, pack=True)


def split_sentences_for_stream(text: str, max_chunk_length: int = 300) -> list[str]:
    """One sentence per chunk so TTS can emit the first audio early.

    Oversized sentences are still split at ``max_chunk_length``.
    """

    return _split_text(text, max_chunk_length, pack=False)


def _split_text(text: str, max_chunk_length: int, *, pack: bool) -> list[str]:
    if max_chunk_length <= 0:
        raise ValueError("Chunk length must be positive.")
    text = text.strip()
    if not text:
        raise ValueError("Text cannot be empty.")

    chunks: list[str] = []
    current = ""
    for sentence in _split_natural_sentences(text):
        for part in _split_oversized_sentence(sentence, max_chunk_length):
            if not pack:
                chunks.append(part)
                continue
            combined = f"{current} {part}" if current else part
            if len(combined) <= max_chunk_length:
                current = combined
                continue
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks


def _split_natural_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        character = text[index]
        if character in "\r\n":
            if sentence := text[start:index].strip():
                sentences.append(sentence)
            index += 1
            while index < len(text) and text[index].isspace():
                index += 1
            start = index
            continue
        if character not in _SENTENCE_ENDINGS:
            index += 1
            continue
        if (
            character == "."
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isdigit()
            and text[index + 1].isdigit()
        ):
            index += 1
            continue

        end = index + 1
        while end < len(text) and (
            text[end] in _SENTENCE_ENDINGS or text[end] in _SENTENCE_CLOSERS
        ):
            end += 1
        if sentence := text[start:end].strip():
            sentences.append(sentence)
        while end < len(text) and text[end].isspace():
            end += 1
        start = end
        index = end

    if sentence := text[start:].strip():
        sentences.append(sentence)
    return sentences


def _split_oversized_sentence(text: str, limit: int) -> list[str]:
    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        minimum = max(1, limit // 2)
        split_at = 0
        for index in range(limit - 1, minimum - 1, -1):
            if remaining[index] in _CLAUSE_BOUNDARIES:
                split_at = index + 1
                break
        if not split_at:
            whitespace = remaining.rfind(" ", minimum, limit + 1)
            if whitespace > 0:
                split_at = whitespace
        if not split_at:
            split_at = limit
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts
