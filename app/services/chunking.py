import re
from dataclasses import dataclass

# ~2000 chars is roughly 400-500 tokens: big enough to hold a complete idea, small enough
# that retrieval stays precise and 5 chunks fit easily in the LLM prompt.
MIN_CHUNK_CHARS = 50  # drop page headers, page numbers and other scraps


@dataclass
class Chunk:
    page: int  # 1-based; a chunk never spans two pages, so citations are exact
    index: int  # position in the whole document, starting at 0
    text: str


def normalise(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "implemen-\ntation" -> "implementation"
    paragraphs = re.split(r"\n\s*\n", text)
    return "\n\n".join(re.sub(r"\s+", " ", p).strip() for p in paragraphs if p.strip())


def split_sentences(text: str) -> list[str]:
    sentences = []
    for paragraph in text.split("\n\n"):
        sentences.extend(s for s in re.split(r"(?<=[.!?])\s+", paragraph) if s)
    return sentences


def hard_split(sentence: str, size: int) -> list[str]:
    """A single sentence longer than `size` is cut at the last space before the limit."""
    pieces = []
    while len(sentence) > size:
        cut = sentence.rfind(" ", 0, size)
        if cut <= 0:
            cut = size
        pieces.append(sentence[:cut].strip())
        sentence = sentence[cut:].strip()
    if sentence:
        pieces.append(sentence)
    return pieces


def joined_length(units: list[str]) -> int:
    return sum(len(u) for u in units) + max(len(units) - 1, 0)  # +1 space between units


def overlap_tail(units: list[str], overlap: int) -> list[str]:
    """The last sentences of a chunk (up to `overlap` chars) that start the next chunk."""
    tail: list[str] = []
    for unit in reversed(units):
        if joined_length([unit] + tail) > overlap:
            break
        tail.insert(0, unit)
    return tail


def chunk_page_text(text: str, size: int, overlap: int) -> list[str]:
    units = []
    for sentence in split_sentences(normalise(text)):
        units.extend(hard_split(sentence, size))

    chunks = []
    current: list[str] = []
    has_new_text = False  # False while `current` holds only overlap carried from before
    for unit in units:
        if current and joined_length(current + [unit]) > size:
            if has_new_text:
                chunks.append(" ".join(current))
            current = overlap_tail(current, overlap)
            has_new_text = False
            if current and joined_length(current + [unit]) > size:
                current = []  # overlap would push this chunk over the limit
        current.append(unit)
        has_new_text = True
    if current and has_new_text:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages: list[tuple[int, str]], size: int, overlap: int) -> list[Chunk]:
    chunks = []
    for page_number, text in pages:
        for piece in chunk_page_text(text, size, overlap):
            if len(piece) >= MIN_CHUNK_CHARS:
                chunks.append(Chunk(page=page_number, index=len(chunks), text=piece))
    return chunks
