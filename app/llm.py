# llm.py
# Handles talking to the Ollama LLM and getting answers back.
#
# Uses proper system/human message separation so the model understands
# its role clearly. The prompt encourages the model to reason, analyze,
# and infer — not just parrot back chunks verbatim.
#
# The num_ctx parameter is set high enough to handle the context chunks
# we send (8 chunks × 1000 chars ≈ 2500 tokens) plus the prompt overhead.

import re

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from app.config import LLM_MODEL, OLLAMA_BASE_URL


SYSTEM_PROMPT = """\
You are an intelligent AI document assistant. You help users understand, \
analyze, and extract insights from their uploaded documents.

Your capabilities:
- Answer questions directly using the provided document context
- Analyze and reason about the information (e.g., calculate dates, \
compare data, identify patterns, make logical inferences)
- Summarize, explain, and interpret content from the documents
- If a direct answer isn't stated in the documents but can be \
reasonably inferred from the available data, provide the inference \
and explain your reasoning
- If the documents truly don't contain enough information to answer, \
say so honestly

Rules:
- Ground your answers in the document content — don't invent facts \
that aren't supported by the text
- When making inferences, clearly state that you're inferring
- Match the language of the user's question in your response"""


USER_PROMPT_TEMPLATE = """\
Here is the relevant context from the user's documents:

---
{context}
---

Question: {question}"""


NO_CONTEXT_PROMPT = """\
The user asked a question but no relevant documents have been uploaded yet. \
Let them know they need to upload documents first, and be helpful about it.

Question: {question}"""


def _clean_response(text: str) -> str:
    """
    Extract the actual answer from a model response, handling <think> tags.

    Strategy:
    1. If there's content AFTER the </think> closing tag, use that.
    2. If stripping the tags leaves nothing, extract from INSIDE the tags.
    3. If there's an unclosed <think> tag, grab what came after it.
    4. Fallback: strip all tags and return whatever's left.
    """
    if not text or not text.strip():
        return ""

    if "<think>" not in text:
        return text.strip()

    # try 1: content after the last </think>
    after_think = re.split(r"</think>", text, flags=re.IGNORECASE)
    if len(after_think) > 1:
        answer = after_think[-1].strip()
        if answer:
            return answer

    # try 2: content inside the last <think>...</think> block
    think_blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if think_blocks:
        inner = think_blocks[-1].strip()
        if inner:
            return inner

    # try 3: unclosed <think> tag (model got cut off)
    after_open = re.split(r"<think>", text, flags=re.IGNORECASE)
    if len(after_open) > 1:
        inner = after_open[-1].strip()
        if inner:
            return inner

    # fallback: strip all tags
    return re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()


def get_llm(model_name: str) -> ChatOllama:
    """Create a ChatOllama instance with appropriate settings."""
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        num_predict=2048,
        num_ctx=8192,  # enough for 8 chunks + prompt overhead
    )


def generate_answer(context: str, question: str, model_name: str = LLM_MODEL) -> str:
    """Build a proper system + human message pair and get the LLM's response."""
    llm = get_llm(model_name)

    # build the messages list with proper role separation
    system_content = SYSTEM_PROMPT

    # for qwen3, disable thinking mode to get direct answers
    if "qwen3" in model_name.lower():
        system_content += "\n\n/no_think"

    if context:
        user_content = USER_PROMPT_TEMPLATE.format(
            context=context, question=question,
        )
    else:
        user_content = NO_CONTEXT_PROMPT.format(question=question)

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    answer = _clean_response(response.content)

    if not answer:
        answer = (
            "I found relevant context from your documents but the model "
            "couldn't generate a clear answer. This sometimes happens with "
            "smaller models. Try rephrasing your question or switching to "
            "a different model (e.g., qwen2.5)."
        )

    return answer

def generate_answer_stream(context: str, question: str, model_name: str = LLM_MODEL):
    """Build a proper system + human message pair and yield the LLM's response chunks."""
    llm = get_llm(model_name)

    # build the messages list with proper role separation
    system_content = SYSTEM_PROMPT

    # for qwen3, disable thinking mode to get direct answers
    if "qwen3" in model_name.lower():
        system_content += "\n\n/no_think"

    if context:
        user_content = USER_PROMPT_TEMPLATE.format(
            context=context, question=question,
        )
    else:
        user_content = NO_CONTEXT_PROMPT.format(question=question)

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]

    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
