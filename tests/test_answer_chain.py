from src.ask_document import answer_chain
from src.ask_document.prompts import REFUSAL_TEXT, build_user_prompt


def test_prompt_contains_only_numbered_evidence():
    prompt = build_user_prompt("What happened?", [{"text": "The event happened yesterday."}])

    assert "[Chunk 1]" in prompt
    assert "The event happened yesterday." in prompt
    assert "What happened?" in prompt


def test_invalid_model_citation_returns_refusal(monkeypatch):
    class Message:
        content = "The answer is supported. [Chunk 9]"

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    class FakeCompletions:
        def create(self, **kwargs):
            return Response()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(answer_chain, "retrieve", lambda *args, **kwargs: [{"chunk_id": "c1", "text": "Evidence", "metadata": {}}])

    output = answer_chain.answer_question("Question", 0, client=FakeClient())

    assert output["answer"] == REFUSAL_TEXT
