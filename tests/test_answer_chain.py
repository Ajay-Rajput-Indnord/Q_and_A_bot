from src.ask_document import answer_chain
from src.ask_document.prompts import REFUSAL_TEXT, build_user_prompt
from src.ask_document.answer_chain import _retrieval_k_for_question


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


def test_multi_passage_questions_use_wide_context():
    assert _retrieval_k_for_question("What is the answer?", 8, "multi_passage") == 16




