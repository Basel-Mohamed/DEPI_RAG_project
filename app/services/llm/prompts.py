from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.services.types import RetrievedContext


def build_support_prompt() -> ChatPromptTemplate:
    """Return the shared grounded prompt used by all answer generators."""

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a customer support assistant for a retrieval-augmented "
                "generation system. Answer only with information grounded in the "
                "provided context. If the context is incomplete, missing, or does "
                "not support the answer, say that you do not have enough verified "
                "information. Keep the answer concise, polite, and actionable.",
            ),
            (
                "human",
                "Answer the customer question using only the provided support "
                "knowledge base context.\n\n"
                "Question:\n{question}\n\n"
                "Context:\n{context}",
            ),
        ]
    )


def build_langchain_chain(model: Any) -> Any:
    """Compose prompt + chat model + string parser into one reusable chain."""

    return build_support_prompt() | model | StrOutputParser()


def build_prompt_messages(question: str, context: str) -> list[dict[str, str]]:
    """Build provider-agnostic chat messages from the shared support prompt."""

    messages = build_support_prompt().format_messages(
        question=question.strip(),
        context=context or "No relevant context provided.",
    )
    return [
        {
            "role": str(message.type),
            "content": str(message.content),
        }
        for message in messages
    ]


def stream_langchain_chain(model: Any, *, question: str, context: str) -> Iterator[str]:
    """Stream parsed text chunks from the shared LangChain chain."""

    chain = build_langchain_chain(model)
    for chunk in chain.stream(
        {
            "question": question.strip(),
            "context": context or "No relevant context provided.",
        }
    ):
        text = str(chunk)
        if text:
            yield text


def build_context_text(documents: list[RetrievedContext], include_metadata: bool) -> str:
    """Render retrieved documents into a deterministic prompt context block."""

    if not documents:
        return ""

    sections: list[str] = []
    for index, document in enumerate(documents, start=1):
        title = document.title or f"Document {index}"
        section_lines = [
            f"[Source {index}]",
            f"Document ID: {document.id or 'unknown'}",
            f"Title: {title}",
        ]

        if include_metadata:
            metadata_lines = [
                f"{key}: {value}"
                for key, value in sorted(document.metadata.items())
                if value not in (None, "")
            ]
            if metadata_lines:
                section_lines.append("Metadata:")
                section_lines.extend(metadata_lines)

        section_lines.extend(
            [
                "Content:",
                document.content.strip() or "No content provided.",
            ]
        )
        sections.append("\n".join(section_lines))

    return "\n\n".join(sections)
