import tiktoken

from src.ask_document.chunking import chunk_text, clean_text


def test_clean_text_preserves_paragraph_separation():
    text = "First   paragraph.\r\n\r\n\r\nSecond\tparagraph."
    assert clean_text(text) == "First paragraph.\n\nSecond paragraph."


def test_chunk_text_uses_configured_token_limit():
    text = "Paragraph one.\n\n" + ("Important information. " * 700)
    chunks = chunk_text(text, chunk_size=512, overlap=64)
    encoding = tiktoken.get_encoding("cl100k_base")

    assert len(chunks) > 1
    assert all(len(encoding.encode(chunk)) <= 512 for chunk in chunks)
