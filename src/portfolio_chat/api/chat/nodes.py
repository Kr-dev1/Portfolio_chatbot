from enum import Enum
from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from portfolio_chat.api.chat.state import State
from portfolio_chat.api.chat.tools import (
    tools,
)
from portfolio_chat.config import (
    GROQ_API_KEY,
)
from portfolio_chat.tools.helpers.helper import retriever
from portfolio_chat.tools.logger.guardrail import guardrail_logger
from portfolio_chat.tools.logger.logger import usage_logger

guard_rail_model = "groq:openai/gpt-oss-20b"
chat_model = "groq:openai/gpt-oss-120b"


class IntentEnum(str, Enum):
    FREELANCE = "freelance"
    WORK = "work"
    GENERAL = "general"


class CalendarEnum(str, Enum):
    AVAILABILITY = "availability"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class TopicCheck(BaseModel):
    is_on_topic: bool = Field(
        description="True if the user query is about our specific topic, False otherwise."
    )
    reason: str = Field(
        description="Reason why the user query was rejected, empty otherwise."
    )
    intent: IntentEnum = Field(
        description="Classify the user input into 3 main categories"
    )
    calendar_action: CalendarEnum | None = Field(
        default=None,
        description=(
            "The requested calendar operation. "
            "Return null when the user is not requesting a calendar operation."
        ),
    )


def guard_rail_and_classiier_node(state: State):
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
                You are a classifier for Kasturi's portfolio assistant.

                The assistant can answer questions about:
                1. Information contained in Kasturi's resume and portfolio.
                2. The current or previous conversation with the user.

                Classify the user's query:

                FREELANCE:
                Questions specifically about Kasturi's freelance work, freelance
                experience, or freelance availability.

                WORK:
                Questions about Kasturi's employment, work experience, companies,
                roles, or professional responsibilities.

                GENERAL:
                Questions about Kasturi or information that could be present in his
                resume or portfolio, including profile and contact information.

                CONVERSATION:
                Questions that refer to information, decisions, requests, or topics
                discussed earlier in the current conversation or previous conversation
                history.

                Set is_on_topic to true for all of the above categories.

                Set is_on_topic to false only when the question is unrelated to
                Kasturi, his portfolio, or the conversation history.

                Examples of CONVERSATION:
                - "What did I ask you earlier?"
                - "What was the approach we discussed?"
                - "What did we decide about the calendar?"
                - "What was the thing you mentioned about the scheduler?"
                - "Can you remind me what I said earlier?"
                - "Continue from where we left off."

                Do not answer the question. Only classify it.
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
    if not result.is_on_topic:
        guardrail_logger(reason=result.reason, user_message=user_input)

    calendar_action = result.calendar_action.value if result.calendar_action else None
    return {
        "is_on_topic": result.is_on_topic,
        "reason": result.reason,
        "intent": result.intent.value,
        "calendar_action": calendar_action,
    }


def chat_node(state: State):
    user_input = state.user_input
    context = retriever.invoke(user_input)

    llm = init_chat_model(
        model=chat_model,
        api_key=GROQ_API_KEY,
    )

    prompt = f"""
    You are Kasturi's portfolio assistant.

    Answer naturally and confidently using only information supported by the
    available context.

    Rules:
    1. Never guess, fabricate, or combine separate facts into a new claim.
    2. Never mention the context, retrieved documents, resume, or sources.
    3. Refer to him as Kasturi and use male pronouns.
    4. If information is unavailable, say so naturally rather than guessing.
    5. Keep answers concise and relevant to the question.

    Contact:
    - For one specific contact detail, answer directly and start with "Kasturi's".
    - For a general contact request, provide the relevant contact methods.
    - Never invent contact information.

    Freelancing:
    - Use Kasturi's freelance information for availability.
    - For freelance technology questions, combine his freelance information with
      his documented technical experience.
    - Deduplicate technologies.

    Experience and capabilities:
    - Distinguish between technologies Kasturi knows, projects he has built,
      and capabilities that are merely possible with those technologies.
    - Only claim that Kasturi has built something when the context explicitly
      supports it.
    - Do not infer a project, capability, integration, business outcome, or
      implementation detail from a technology alone.

    Examples:

    User: What is Kasturi's email?
    Assistant: Kasturi's email is me@krangan.dev.

    User: How can I contact Kasturi?
    Assistant:
    You can contact Kasturi through:
    - Email: me@krangan.dev
    - LinkedIn: linkedin.com/in/-kasturirangan
    - GitHub: github.com/kr-dev1
    - Portfolio: krangan.dev

    User: Does Kasturi have experience with Rust?
    Assistant: Kasturi does not have documented experience with Rust.

    User: Is Kasturi available for freelance work?
    Assistant: Yes, Kasturi is available for freelance software development work.

    User: What technologies can Kasturi use for freelance work?
    Assistant:
    Kasturi can work with React, Next.js, TypeScript, Node.js, Python, Vue.js,
    JavaScript, Express, Fastify, PostgreSQL, MongoDB, Prisma, LangChain,
    LangGraph, Google Gemini, OpenAI, Qdrant, RAG, and other technologies
    supported by his experience.

    User: What has Kasturi built?
    Assistant:
    Kasturi has built projects including AskRepo, a RAG-based knowledge system
    for GitHub repositories, and LegalHawk, an AI-powered contract analysis
    application.

    User: What can Kasturi build?
    Assistant:
    Kasturi has experience building full-stack web applications, AI-powered
    applications, RAG systems, and document analysis tools. His projects
    include AskRepo and LegalHawk.

    User: Can Kasturi build a Stripe subscription system?
    Assistant:
    Kasturi's experience includes full-stack application development, but his
    documented experience does not specifically establish that he has built
    Stripe subscription systems.

    User: What can Kasturi build with React?
    Assistant:
    Kasturi has used React in his full-stack and frontend projects. His
    documented projects include [relevant projects from context].

    User: What can Kasturi build?

    Assistant:
    Kasturi has experience building full-stack web applications, AI-powered
    applications, RAG systems, and document analysis tools. Projects such as
    AskRepo and LegalHawk demonstrate this experience

    Do not answer beyond what the context supports.

    Context:
    {context}

    Question:
    {user_input}
    """

    messages: list[BaseMessage] = [
        SystemMessage(content=prompt),
    ]

    if state.conversation_summary:
        messages.append(
            SystemMessage(
                content=f"""
    Previous conversation summary:
    {state.conversation_summary}
    """
            )
        )

    messages.extend(state.messages)

    llm_with_tools = llm.bind_tools(tools)
    resp = llm_with_tools.invoke(messages)
    usage = resp.usage_metadata
    if usage:
        metrics = {
            "guard_rail_model": guard_rail_model,
            "chat_model": chat_model,
            "input_tokens": usage.get("input_tokens") if usage else None,
            "output_tokens": usage.get("output_tokens") if usage else None,
            "total_tokens": usage.get("total_tokens") if usage else None,
            "reasoning_tokens": (
                usage.get("output_token_details", {}).get("reasoning")
                if usage
                else None
            ),
        }

        usage_logger(
            chat_model=metrics["chat_model"],
            guard_rail_model=metrics["guard_rail_model"],
            input_tokens=metrics["input_tokens"],
            output_tokens=metrics["output_tokens"],
            total_tokens=metrics["total_tokens"],
            reasoning_tokens=metrics["reasoning_tokens"],
            guardrail_triggered=False,
        )

    return {"messages": [resp], "context": context, "response": resp.content}


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
    return {"messages": [response], "response": response.content}
