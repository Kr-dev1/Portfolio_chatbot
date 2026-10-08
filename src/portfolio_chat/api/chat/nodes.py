import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import cast
from zoneinfo import ZoneInfo

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from portfolio_chat.api.chat.state import State
from portfolio_chat.api.chat.tools import (
    calendar_id,
    tools,
)
from portfolio_chat.config import (
    GROQ_API_KEY,
)
from portfolio_chat.scripts.google_calendar import service
from portfolio_chat.tools.helpers.helper import retriever
from portfolio_chat.tools.logger.guardrail import guardrail_logger
from portfolio_chat.tools.logger.logger import usage_logger

guard_rail_model = "gpt-6-luna"
chat_model = "groq:openai/gpt-oss-20b"
classifier_model = "gpt-6-luna"


class IntentEnum(str, Enum):
    FREELANCE = "freelance"
    WORK = "work"
    GENERAL = "general"


class CalendarEnum(str, Enum):
    AVAILABILITY = "availability"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class GetUserInfo(BaseModel):
    name: str | None = Field(
        default=None,
        description=(
            "The name of the person making the scheduling request. "
            "If the user introduces themselves, such as "
            "'I am Pranav' or 'Hey, I'm Pranav', extract 'Pranav'. "
            "Do not extract the calendar owner's name, Kasturi, "
            "when the user says they want to meet with Kasturi."
        ),
    )

    email: str | None = Field(
        default=None,
        description="Email address of the person requesting the meeting.",
    )

    start_time: datetime | None = Field(
        default=None,
        description="The requested meeting start date and time.",
    )

    end_time: datetime | None = Field(
        default=None,
        description="The requested meeting end date and time.",
    )

    attendees: list[str] = Field(
        default_factory=list,
        description=(
            "Email addresses of people other than Kasturi who should "
            "attend the meeting. Only include an email address when "
            "the user explicitly provides it."
        ),
    )


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
            "The requested calendar operation could be availability, create, update or delete"
            "Return null when the user is not requesting a calendar operation."
        ),
    )


def guard_rail_and_classiier_node(state: State):
    user_input = state.user_input

    evaluator_llm = init_chat_model(
        model=guard_rail_model,
    ).with_structured_output(TopicCheck)

    recent_messages = state.messages[-4:]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a classifier for Kasturi's portfolio assistant.

                The assistant can answer questions about:
                1. Information contained in Kasturi's resume and portfolio.
                2. The current or previous conversation with the user.
                3. Scheduling meetings with Kasturi.

                Classify the user's latest message.

                Use the recent conversation context to determine whether
                the latest message is a continuation of an earlier request.

                FREELANCE:
                Questions specifically about Kasturi's freelance work,
                freelance experience, or freelance availability.

                WORK:
                Questions about Kasturi's employment, work experience,
                companies, roles, or professional responsibilities.

                GENERAL:
                Questions about Kasturi or information that could be present
                in his resume or portfolio.

                MEETING:
                Requests related to Kasturi's calendar.

                For MEETING requests:
                - Creating a new meeting -> calendar_action = CREATE
                - Checking availability -> calendar_action = AVAILABILITY
                - Updating/rescheduling an existing meeting -> calendar_action = UPDATE
                - Cancelling/deleting an existing meeting -> calendar_action = DELETE

                MEETING requests must have is_on_topic = true.

                Examples:

                "Can you schedule a meeting with Kasturi?"
                -> is_on_topic = true
                -> intent = GENERAL
                -> calendar_action = CREATE

                "Is Kasturi free tomorrow at 5?"
                -> is_on_topic = true
                -> intent = GENERAL
                -> calendar_action = AVAILABILITY

                "Move my meeting to Friday"
                -> is_on_topic = true
                -> intent = GENERAL
                -> calendar_action = UPDATE

                "Cancel my meeting with Kasturi"
                -> is_on_topic = true
                -> intent = GENERAL
                -> calendar_action = DELETE

                Set is_on_topic to true for all of the above categories.

                Set is_on_topic to false only when the latest message is
                unrelated to Kasturi, his portfolio, meetings with Kasturi,
                or the conversation.

                IMPORTANT:
                The latest message may be a continuation of an earlier request.
                For example:

                User: "I want to schedule a meeting with Kasturi."
                Assistant: "Sure, what date and time?"
                User: "At 5 tomorrow."

                The latest message is a MEETING request and is_on_topic
                must be true.x
                Do not answer the user's question. Only classify it.
                """,
            ),
            (
                "human",
                """
                Recent conversation:
                {conversation}

                Latest user message:
                {user_input}
                """,
            ),
        ]
    )

    chain = prompt | evaluator_llm
    result = cast(
        TopicCheck,
        chain.invoke(
            {
                "conversation": recent_messages,
                "user_input": user_input,
            }
        ),
    )

    if not result.is_on_topic:
        guardrail_logger(
            reason=result.reason,
            user_message=user_input,
        )

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


def extract_user_info(state: State):
    user_input = state.user_input
    llm = init_chat_model(
        model=classifier_model,
    ).with_structured_output(GetUserInfo)

    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    prompt = f"""
    Extract calendar information from the user's message.

    Current date and time:
    {now.isoformat()}

    Timezone:
    Asia/Kolkata

    User message:
    {user_input}

    Rules:
    - The calendar owner is Kasturi.
    - The person making the request is the requester.
    - If the user introduces themselves, extract their name into `name`.
      Examples:
        "I am Pranav" -> name = "Pranav"
        "Hey, I'm John" -> name = "John"
        "This is Sarah" -> name = "Sarah"
    - If the user says they want to schedule a meeting "with Kasturi",
      Kasturi is the calendar owner and must NOT be extracted as `name`.
    - Do not put names into `attendees`.
    - `attendees` must contain only explicitly provided attendee email addresses.
    - Extract the requester's email when explicitly provided.
    - Resolve relative dates such as "today", "tomorrow", and "next Monday"
      using the current date above.
    - Extract explicit times such as "5pm", "5 PM", "17:00", etc.
    - If only a start time is provided, leave `end_time` empty.
    - Only extract information explicitly stated or directly implied by
      relative date/time expressions.
    """

    result = cast(GetUserInfo, llm.invoke(prompt))
    start_time = result.start_time or state.start_time

    if result.end_time:
        end_time = result.end_time
    elif result.start_time:
        end_time = result.start_time + timedelta(hours=1)
    else:
        end_time = state.end_time

    return {
        "name": result.name or state.name,
        "email": result.email or state.email,
        "start_time": start_time,
        "end_time": end_time,
        "attendees": result.attendees or state.attendees,
    }


def get_availability_info(state: State):
    missing = []
    if not state.start_time:
        missing.append("start_time")
    if not state.name and state.start_time:
        missing.append("name")

    if not state.email and state.start_time:
        missing.append("email")

    llm = init_chat_model(
        model=chat_model,
        api_key=GROQ_API_KEY,
        temperature=0,
    )

    prompt = ""

    if state.calendar_available is False:
        prompt = f"""
        The user wants to schedule a meeting, but the meeting slot is already booked during the same interval.
        Here are all the events from that day {state.calendar_events}.
        Ask the user for the date and time they would like to re-schedule the
        meeting since the original time they pick is already booked.

        Do not ask for their name, email, attendees, or other information yet.
        Keep the response concise and conversational.
        Return only the message shown to the user."""
    else:
        prompt = f"""
        The user wants to schedule a meeting, but the meeting {missing} missing.
        Ask the user for the missing fields.

        if the missing items dosent have start_time, Ask the user if there will be anyother attendees as well to add to the meeting

        Do not ask for any additional information yet.
        Keep the response concise and conversational.
        Return only the message shown to the user.
        """

    response = llm.invoke(prompt)
    return {"messages": [response], "response": response.content}


def check_availability(state: State):
    print(state)
    if not state.start_time or not state.end_time:
        raise ValueError("Start time and end time are required")

    def get_events(start_time: datetime, end_time: datetime):
        events = (
            service.events()
            .list(
                calendarId=calendar_id,
                timeMin=start_time.isoformat(),
                timeMax=end_time.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        return events["items"]

    # inital check for events
    check_event = get_events(state.start_time, state.end_time)

    if not check_event:
        return {"calendar_events": [], "calendar_available": True}

    start_window = state.start_time.replace(hour=10, minute=0, second=0, microsecond=0)
    end_window = state.end_time.replace(hour=18, minute=0, second=0, microsecond=0)
    check_event = get_events(start_window, end_window)
    return {"calendar_events": check_event, "calendar_available": False}


def availability_confirmation(state: State):
    req_id = uuid.uuid4()
