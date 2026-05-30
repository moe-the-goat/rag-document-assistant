# llm.py
# Handles talking to the Ollama LLM and getting answers back.
#
# The prompt tells the model to only use the context we give it (the chunks
# we pulled from the vector store). If there's no context, we just tell the
# user to upload something first.
#
# qwen3 wraps its reasoning in <think> tags. We handle this gracefully:
# first we check for content after the tags; if that's empty, we extract
# the useful parts from inside the tags themselves.

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


def _clean_response(text: str) -> str:
    """
    Extract the actual answer from a model response, handling <think> tags.

    Strategy:
    1. If there's content AFTER the </think> closing tag, use that (normal case).
    2. If stripping the tags leaves nothing, extract content from INSIDE the
       last <think> block — the model sometimes puts the answer there.
    3. If all else fails, return the raw text with tags stripped.
    """
    if not text or not text.strip():
        return ""

    # check if there are think tags at all
    if "<think>" not in text:
        return text.strip()

    # try 1: grab everything AFTER the last </think> tag
    after_think = re.split(r"</think>", text, flags=re.IGNORECASE)
    if len(after_think) > 1:
        answer = after_think[-1].strip()
        if answer:
            return answer

    # try 2: the answer is trapped inside the think tags
    # extract content from the last <think>...</think> block
    think_blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if think_blocks:
        # the last think block usually contains the final reasoning + answer
        inner = think_blocks[-1].strip()
        if inner:
            return inner

    # try 3: there's an opening <think> but no closing tag (model got cut off)
    # grab everything after the <think> tag
    after_open = re.split(r"<think>", text, flags=re.IGNORECASE)
    if len(after_open) > 1:
        inner = after_open[-1].strip()
        if inner:
            return inner

    # fallback: return raw text with any tags removed
    return re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()


def get_llm(model_name: str) -> ChatOllama:
    # low temperature because we want factual answers, not creative writing
    # num_predict caps the length so the model doesn't ramble forever
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=2048,
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
    if "qwen3" in model_name.lower():
        prompt += " /no_think"

    response = llm.invoke(prompt)
    answer = _clean_response(response.content)

    # safety net: if we still got nothing, return a fallback
    if not answer:
        answer = (
            "I found relevant context from your documents but couldn't "
            "generate a clear answer. Please try rephrasing your question."
        )

    return answer
