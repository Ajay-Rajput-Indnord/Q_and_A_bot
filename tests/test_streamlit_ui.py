from src.ask_document.prompts import REFUSAL_TEXT


def test_refusal_text_is_stable_for_ui_display():
    assert REFUSAL_TEXT == "I don't know from this document."
