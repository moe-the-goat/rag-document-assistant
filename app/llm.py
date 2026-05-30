# llm.py
# Handles talking to the Ollama LLM and getting answers back.
#
# The prompt tells the model to only use the context we give it (the chunks
# we pulled from the vector store). If there's no context, we just tell the
# user to upload something first.
#
# We append /no_think to prompts sent to qwen3 models to disable their
# internal chain-of-thought mode, which can sometimes swallow the answer.
# For non-qwen3 models, we still strip any <think> tags as a safety net.

import re

from langchain_ollama import ChatOllama

from app.config import LLM_MODEL, OLLAMA_BASE_URL

# we tell the model pretty firmly to stick to the context and not make stuff up
RAG_PROMPT_TEMPLATE = """\
You are a helpful AI assistant. Answer the user's question based ONLY \
on the provided context from their documents. Be specific and detailed \
in your answer. If the context does not contain enough information to \
answer, say so clearly -- do not make up information.

IMPORTANT: You must generate your answer in the EXACT SAME LANGUAGE that the user used to ask their question.

Context from documents:
{context}

User's question: {question}

Answer:"""


def _strip_thinking_tags(text: str) -> str:
    # some models dump chain-of-thought in <think>...</think> blocks
    # we don't want the user to see that, so strip it out
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_llm(model_name: str) -> ChatOllama:
    # low temperature because we want factual answers, not creative writing
    # num_predict caps the length so the model doesn't ramble forever
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=1024,
    )


def generate_answer(context: str, question: str, model_name: str = LLM_MODEL) -> str:
    # put together the prompt, send it to Ollama, clean up the response
    llm = get_llm(model_name)

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

    # for qwen3 models, append /no_think to disable internal chain-of-thought
    # this prevents the model from hiding its answer inside <think> tags
    if "qwen3" in model_name.lower():
        prompt += " /no_think"

    response = llm.invoke(prompt)
    answer = _strip_thinking_tags(response.content)

    # safety net: if stripping left us with nothing, return a fallback
    if not answer:
        answer = (
            "I found relevant context from your documents but couldn't "
            "generate a clear answer. Please try rephrasing your question."
        )

    return answer
