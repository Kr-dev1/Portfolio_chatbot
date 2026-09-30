import time
from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from portfolio_chat.api.chat.state import State
from portfolio_chat.config import (
    GROQ_API_KEY,
)
from portfolio_chat.tools.helpers.helper import retriever
from portfolio_chat.tools.logger.guardrail import guardrail_logger

guard_rail_model = "groq:openai/gpt-oss-safeguard-20b"
chat_model = "groq:openai/gpt-oss-20b"


class TopicCheck(BaseModel):
    is_on_topic: bool = Field(
        description="True if the user query is about our specific topic, False otherwise."
    )
    reason: str = Field(
        description="Reason why the user query was rejected, empty otherwise."
    )


def guard_rail_node(state: State):
    user_input = state.user_input
    evaluator_llm = init_chat_model(
        model=guard_rail_model,
        temperature=0,
    ).with_structured_output(TopicCheck)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
              You are a strict input classifier.

              Return true ONLY if the user's question is about Kasturi's:
              - professional experience
              - employment history
              - skills
              - technologies
              - projects
              - education
              - professional background
              - portfolio

              Return false for unrelated questions, coding requests,
              general knowledge, casual conversation, or questions about
              topics unrelated to Kasturi.
              """,
            ),
            ("human", "{user_input}"),
        ]
    )

    chain = prompt | evaluator_llm

    result = cast(
        TopicCheck,
        chain.invoke({"user_input": user_input}),
    )

    guardrail_logger(reason=result.reason, user_message=user_input)

    return {"is_on_topic": result.is_on_topic, "reason": result.reason}


def chat_node(state: State):
    user_input = state.user_input
    llm = init_chat_model(
        model=chat_model,
        api_key=GROQ_API_KEY,
    )

    context = retriever.invoke(user_input)


    prompt = f"""
    You are Kasturi's portfolio assistant.

    Answer the user's question using ONLY information explicitly present
    in the provided context.

    Rules:

    1. Never infer, extrapolate, or add details that are not explicitly
       stated in the context.

    2. For experience/technology questions:
    - If the context explicitly shows Kasturi has experience with it,
      say yes and explain where.
    - If the context does not explicitly show experience with it, say:
      "No, Kasturi does not have experience with [technology]."

    3. Keep claims at the same level of specificity as the context.
    Do not add implementation details, use cases, metrics, or technologies
    unless they are explicitly mentioned.

    4. Do not mention the resume, context, retrieved documents, or sources.

    5. Refer to the person as Kasturi and male.

    Context:
    {context}

    Question:
    {user_input}
    """

    start = time.perf_counter()

    context = retriever.invoke(user_input)

    print(
        f"Retrieval: {time.perf_counter() - start:.3f}s"
    )

    llm_start = time.perf_counter()

    resp = llm.invoke(prompt)

    print(
        f"LLM: {time.perf_counter() - llm_start:.3f}s"
    )

    print(
        f"Total chat node: {time.perf_counter() - start:.3f}s"
    )

    return {"messages": [resp], "context": context, "response": resp.content}
    # start = time.perf_counter()
    # first_token_time = None
    # usage = None

    # for chunk in llm.stream(prompt):
    #     if chunk.content:
    #         if first_token_time is None:
    #             first_token_time = time.perf_counter()
    #         return {"messages": chunk.content(state)}

    #     if chunk.usage_metadata:
    #         usage = chunk.usage_metadata

    # end = time.perf_counter()

    # metrics = {
    #     "guard_rail_model": guard_rail_model,
    #     "chat_model": chat_model,
    #     "time_to_first_token": (first_token_time - start if first_token_time else None),
    #     "total_time": end - start,
    #     "input_tokens": usage.get("input_tokens") if usage else None,
    #     "output_tokens": usage.get("output_tokens") if usage else None,
    #     "total_tokens": usage.get("total_tokens") if usage else None,
    #     "reasoning_tokens": (
    #         usage.get("output_token_details", {}).get("reasoning") if usage else None
    #     ),
    # }

    # usage_logger(
    #     chat_model=metrics["chat_model"],
    #     guard_rail_model=metrics["guard_rail_model"],
    #     time_to_first_token=metrics["time_to_first_token"],
    #     input_tokens=metrics["input_tokens"],
    #     output_tokens=metrics["output_tokens"],
    #     total_tokens=metrics["total_tokens"],
    #     reasoning_tokens=metrics["reasoning_tokens"],
    #     total_time=metrics["total_time"],
    #     guardrail_triggered=False,
    # )


def guardrail_fail_node(state: State):
    user_input = state.user_input
    reason = state.reason
    llm = init_chat_model(model=chat_model, api_key=GROQ_API_KEY)

    prompt = f"""
        You are Kasturi's portfolio assistant.

        The user's question was outside the scope of the assistant.

        Explain briefly and politely that you can only help with:
        - Kasturi's professional experience
        - projects
        - skills
        - technologies
        - employment
        - education
        - professional background

        Do not answer the original question.

        User Input:
        {user_input}

        Reason for rejection:
        {reason}
        """

    response = llm.invoke(prompt)
    guardrail_logger(reason=state.reason, user_message=state.user_input)
    return {"messages": [response], "response": response.content}
