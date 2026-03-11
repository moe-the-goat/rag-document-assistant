# llm.py
# Handles talking to the Ollama LLM and getting answers back.
#
# The prompt tells the model to only use the context we give it (the chunks
# we pulled from the vector store). If there's no context, we just tell the
# user to upload something first.
#
# qwen3 has this quirk where it wraps its internal reasoning in <think> tags.
# We strip those out so the user only sees the actual answer.

import re

from langchain_ollama import ChatOllama

from app.config import LLM_MODEL, OLLAMA_BASE_URL

# we tell the model pretty firmly to stick to the context and not make stuff up
RAG_PROMPT_TEMPLATE = """\
You are a helpful AI assistant. Answer the user's question based ONLY \
on the provided context from their documents. Be specific and detailed \
in your answer. If the context does not contain enough information to \
answer, say so clearly -- do not make up information.

Context from documents:
{context}

User's question: {question}

Answer:"""


def _strip_thinking_tags(text: str) -> str:
    # qwen3 sometimes dumps its chain-of-thought in <think>...</think> blocks
    # we don't want the user to see that, so strip it out
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_llm() -> ChatOllama:
    # low temperature because we want factual answers, not creative writing
    # num_predict caps the length so the model doesn't ramble forever
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=1024,
    )


def generate_answer(context: str, question: str) -> str:
    # put together the prompt, send it to Ollama, clean up the response
    llm = get_llm()

    if not context:
        # no documents uploaded yet, let the user know
        prompt = (
            "The user asked a question but no relevant documents have "
            "been uploaded yet. Please let them know they need to upload "
            "documents first.\n\n"
            f"User's question: {question}"
        )
    else:
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    response = llm.invoke(prompt)
    return _strip_thinking_tags(response.content)
