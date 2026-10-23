from app.services.chunking import chunk_pages, hard_split, normalise


def sentences(prefix: str, count: int) -> str:
    return " ".join(f"{prefix} sentence number {i} is written here." for i in range(count))


def test_chunking_never_crosses_pages():
    pages = [(1, sentences("Alpha", 30)), (2, sentences("Beta", 30))]
    chunks = chunk_pages(pages, size=300, overlap=80)
    assert {c.page for c in chunks} == {1, 2}
    for chunk in chunks:
        marker = "Alpha" if chunk.page == 1 else "Beta"
        other = "Beta" if chunk.page == 1 else "Alpha"
        assert marker in chunk.text
        assert other not in chunk.text
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunking_overlap_and_size_limits():
    chunks = chunk_pages([(1, sentences("Gamma", 40))], size=300, overlap=100)
    assert len(chunks) > 2
    assert all(len(c.text) <= 300 for c in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        # The next chunk starts with a sentence that ended the previous one.
        first_sentence = current.text.split(". ")[0].rstrip(".") + "."
        assert first_sentence in previous.text


def test_long_sentence_is_hard_split():
    long_sentence = " ".join(["word"] * 400)  # one 2000+ char "sentence" with no full stop
    pieces = hard_split(long_sentence, 300)
    assert len(pieces) > 1
    assert all(len(p) <= 300 for p in pieces)
    chunks = chunk_pages([(1, long_sentence)], size=300, overlap=50)
    assert all(len(c.text) <= 300 for c in chunks)


def test_normalise_fixes_hyphenated_line_breaks_and_whitespace():
    result = normalise("This is the implemen-\ntation   of  it.")
    assert result == "This is the implementation of it."


def test_tiny_chunks_are_dropped():
    assert chunk_pages([(1, "Page 3"), (2, "x")], size=300, overlap=50) == []
