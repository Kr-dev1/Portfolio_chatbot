import smtplib
import uuid
from datetime import datetime, timedelta
from email.message import EmailMessage
from enum import Enum
from typing import Literal, cast
from zoneinfo import ZoneInfo

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from portfolio_chat.api.chat.state import State
from portfolio_chat.config import (
    DEEPSEEK_API_KEY,
    GROQ_API_KEY,
    PRIMARY_SPACE_MAIL,
    SPACE_MAIL,
    SPACE_PASSWORD,
)
from portfolio_chat.scripts.google_calendar import service
from portfolio_chat.tools.helpers.helper import check_events, retriever
from portfolio_chat.tools.logger.guardrail import guardrail_logger
from portfolio_chat.tools.requests.model import status
from portfolio_chat.tools.requests.request import (
    check_db_for_event,
    create_req,
    get_events_email,
    update_req,
)

guard_rail_model = "deepseek-chat"
email_model = "groq:openai/gpt-oss-20b"
chat_model = "groq:openai/gpt-oss-120b"
classifier_model = "deepseek-chat"
calendar_id = "primary"


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

    event_description: str | None = Field(
        default=None,
        description=("""A title for the meeting"""),
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


class UpdateField(BaseModel):
    field: Literal[
        "name",
        "event_description",
        "start_date_time",
        "end_date_time",
    ]
    value: str


class EditGetInfo(BaseModel):
    email: str | None = Field(
        default=None,
        description="Email address associated with the existing meeting.",
    )
    selection: int | None = Field(
        default=None,
        description="The numbered event option selected by the user.",
    )
    updates: list[UpdateField] = Field(
        default_factory=list,
        description="Fields the user wants to change. Empty if none were stated.",
    )


def guard_rail_and_classiier_node(state: State):
    user_input = state.user_input

    evaluator_llm = init_chat_model(
        model=guard_rail_model, model_provider="deepseek", api_key=DEEPSEEK_API_KEY
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
                MEETING requests should always have previous messages in consideration.
                User is already in the flow with intention of {calendar_intent}.
                !!!!A decision based on only the user's current message leads to chaos!!!!

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
                Kasturi is male.
                The latest message may be a continuation of an earlier request.
                For example:

                User: "I want to schedule a meeting with Kasturi."
                Assistant: "Sure, what date and time?"
                User: "At 5 tomorrow."

                The latest message is a MEETING request and is_on_topic
                must be true.x
                Do not answer the user's question. Only classify it.

                Donot act on the users current message always include the previous messages.

                User Input:
                {user_input}

                previous conversation:
                {conversation}

                summary of previous conversation:
                {summary}
                """,
            )
        ]
    )

    chain = prompt | evaluator_llm
    result = cast(
        TopicCheck,
        chain.invoke(
            {
                "conversation": recent_messages,
                "user_input": user_input,
                "summary": state.conversation_summary,
                "calendar_intent": state.current_calendar_flow,
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


async def chat_node(state: State):
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
    6. Never invent imformation apart form what is available.

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

    resp = await llm.ainvoke(messages)

    return {"messages": [resp], "context": context, "response": resp.content}


async def guardrail_fail_node(state: State):
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

    response = await llm.ainvoke(prompt)
    return {"messages": [response], "response": response.content}


def extract_user_info(state: State):
    user_input = state.user_input
    llm = init_chat_model(
        model=classifier_model, model_provider="deepseek", api_key=DEEPSEEK_API_KEY
    ).with_structured_output(GetUserInfo)

    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    prompt = f"""
    Extract calendar information from the user's message.

    Current date and time:
    {now.isoformat()}

    Timezone:
    Asia/Kolkata

    User message:
    {user_input, state.messages, state.conversation_summary}

    Rules:
    - The calendar owner is Kasturi.
    - The person making the request is the requester.
    - If the user introduces themselves, extract their name into `name`.
    - Meeting with kasturi cannot be a meeting name unless specified by the user.
      Examples:
        "I am Pranav" -> name = "Pranav"
        "Hey, I'm John" -> name = "John"
        "This is Sarah" -> name = "Sarah"
    - If the user says they want to schedule a meeting "with Kasturi",
      Kasturi is the calendar owner and must NOT be extracted as `name`.
    - If there is one singular email it is always the persons email.
    - Do not put names into `attendees`.
    - If the exact time and date is not present, ex:- next week at 2pm donot extract.
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

    if start_time and end_time:
        return {
            "name": result.name or state.name,
            "email": result.email or state.email,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "attendees": result.attendees or state.attendees,
            "event_description": result.event_description or state.event_description,
            "current_calendar_flow": state.current_calendar_flow,
        }


def get_availability_info(state: State):
    missing = []
    if not state.start_time:
        missing.append("start_time_date")
    if not state.name and state.start_time:
        missing.append("name")
    if not state.event_description and state.start_time:
        missing.append("meeting_summary")
    if not state.email and state.start_time:
        missing.append("email")

    llm = init_chat_model(
        model=chat_model,
        api_key=GROQ_API_KEY,
    )
    prompt = ""
    if state.calendar_available is False:
        prompt = f"""
        The user wants to schedule a meeting, but the meeting slot is already booked during the same interval.
        Here are all the events from that day {state.calendar_events}.
        Ask the user for the date and time they would like to re-schedule the
        meeting since the original time they picked is already booked.

        Do not ask for their name, email, attendees, or other information yet.
        Keep the response concise and conversational.
        Return only the message shown to the user."""
    else:
        prompt = f"""
        This means there is a meeting slot available respond based on the user input {state.user_input}
        The user wants to schedule a meeting, but the {missing} is not available.

        If name is missing it is the name of the user and not the name for the meeting.

        Do not ask for any additional information yet.
        Return only the message shown to the user.
        Request only the fields that are missing
        """

    response = llm.invoke(prompt)
    return {"messages": [response], "response": response.content}


def check_availability(state: State):
    if not state.start_time or not state.end_time:
        raise ValueError("Start time and end time are required")

    # inital check for events
    check_event = check_events(state.start_time, state.end_time)

    if not check_event:
        return {"calendar_events": [], "calendar_available": True}

    start_window = state.start_time.replace(hour=10, minute=0, second=0, microsecond=0)
    end_window = state.end_time.replace(hour=18, minute=0, second=0, microsecond=0)
    check_event = check_events(start_window, end_window)
    return {"calendar_events": check_event, "calendar_available": False}


async def save_approval_request(state: State):
    req_id = str(uuid.uuid4())
    llm = init_chat_model(model=chat_model, api_key=GROQ_API_KEY)
    event_exists = check_db_for_event(state.email, state.start_time)

    if event_exists:
        response = await llm.ainvoke(
            f"""
            The user wants to create a meeting, but they already have
            an meeting that is either approved or pending on the same day.

            Existing meeting:
            - Date and time: {event_exists.start_date_time}
            - End time: {event_exists.end_date_time}
            - Description: {event_exists.event_description}

            User's current request:
            - Requested date and time: {state.start_time}
            - Description: {state.event_description}

            Explain that they cannot create another meeting under the current
            booking rules. Suggest editing their existing meeting if needed.

            Keep the response concise and conversational.
            Do not claim that any changes have been made.
            """
        )

        return {
            "messages": [response],
            "response": response.content,
            "event_exists": True,
        }

    data = {
        "thread_id": state.thread_id,
        "req_id": req_id,
        "name": state.name,
        "email": state.email,
        "start_date_time": state.start_time,
        "end_date_time": state.end_time,
        "event_description": state.event_description,
        "status": "pending",
    }

    create_req(**data)

    response = await llm.ainvoke(
        f"""
        The user's meeting request has been saved and is pending verification.
        Meeting: {state.event_description}
        Start time: {state.start_time}

        Tell the user you have sent a request provisionally to kasturi to confirm availability outside the calendar
        and that they will receive an email from assistant@krangan.dev
        once Kasturi verifies it.

        Keep the response concise and conversational.
        Do not imply that the calendar event has already been created.
        """
    )

    return {
        "req_id": req_id,
        "messages": [response],
        "response": response.content,
        "event_exists": False,
    }


def availability_confirmation(state: State):
    data = {
        "req_id": state.req_id,
        "thread_id": state.thread_id,
        "name": state.name,
        "email": state.email,
        "start_date_time": state.start_time,
        "end_date_time": state.end_time,
        "event_description": state.event_description,
    }

    decision = interrupt(data)

    return {"booking_approved": decision.get("approved", False)}


def create_event(state: State):
    if not state.start_time or not state.end_time:
        raise ValueError("Start time and end time are required")

    event = {
        "summary": state.event_description,
        "description": f"Meeting requested by {state.name}",
        "start": {
            "dateTime": state.start_time.isoformat(),
            "timeZone": "Asia/Kolkata",
        },
        "end": {
            "dateTime": state.end_time.isoformat(),
            "timeZone": "Asia/Kolkata",
        },
        "attendees": [
            {"email": state.email},
            *[{"email": email} for email in state.attendees],
        ],
    }

    created_event = (
        service.events()
        .insert(
            calendarId=calendar_id,
            body=event,
            sendUpdates="all",
        )
        .execute()
    )
    print(created_event)

    if not state.req_id or state.booking_approved is not True:
        raise ValueError("Request ID missing or request not approved")

    approval = status.APPROVED

    update_req(req_id=state.req_id, approval=approval)

    return {
        "calendar_event_id": created_event["id"],
        "current_calendar_flow": None,
    }


def denied_request(state: State):
    if not state.req_id or state.booking_approved is not False:
        raise ValueError("Request ID missing or request not approved")

    approval = status.DECLINED

    update_req(req_id=state.req_id, approval=approval)

    return {
        "current_calendar_flow": None,
    }


def send_email(state: State):
    print("sending email")
    SMTP_SERVER = "mail.spacemail.com"
    SMTP_PORT = 465
    SENDER_EMAIL = SPACE_MAIL
    SENDER_PASSWORD = SPACE_PASSWORD
    RECIPIENT_EMAIL = state.email

    msg = EmailMessage()

    llm = init_chat_model(model=email_model, api_key=GROQ_API_KEY)

    email = llm.invoke(
        f"""
    Write a professional email to {state.name} regarding their
    meeting request with Kasturi.

    Meeting details:
    - Request type: {state.intent}
    - Date and time: {state.start_time}

    Instructions:
    - Write in a polite, professional, and concise tone.
    - Clearly explain the outcome of their meeting request.
    - Include the meeting date and time when relevant.
    - Use an appropriate subject line and email body.
    - Do not invent details that are not provided.
    - Sign off as Kasturi.

    Return only the email, including the subject line.
    """
    )
    msg["Subject"] = "Your Request confirmation with Kasturi"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg.set_content(email.content)

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(PRIMARY_SPACE_MAIL, SENDER_PASSWORD)
            server.send_message(msg)
            print("Email sent  ")
            return {"email_status": "sent"}
    except smtplib.SMTPAuthenticationError:
        print("Send Email failed Auth error")
        return {
            "email_status": "failed",
            "error": "Authentication failed: Check your username or password.",
        }
    except smtplib.SMTPConnectError:
        print("Send Email failed connection error")
        return {
            "email_status": "failed",
            "error": "Connection failed: Check your SMTP host and port.",
        }
    except smtplib.SMTPException as e:
        print(f"Send email faile {e}")
        return {"email_status": "failed", "error": str(e)}


def edit_event_extractor(state: State):
    llm = init_chat_model(
        model=classifier_model, model_provider="deepseek"
    ).with_structured_output(EditGetInfo)

    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    result = cast(
        EditGetInfo,
        llm.invoke(
            f"""
            Extract information from the user's latest message.

            User message:
            {state.user_input}

            Selected event:
            {state.selected_event}

            Recent conversation:
            {state.messages[-4:]}

            Current date and time: {now.isoformat()} (Asia/Kolkata)

            Allowed update fields: name, event_description, start_date_time, end_date_time.
            For date/time fields, return ISO 8601 with the +05:30 offset.

            Example: "move it to tomorrow 3pm" ->
            updates=[{{"field": "start_date_time", "value": "<ISO datetime>"}}]

            Extract:
            - email: the email associated with the user's existing meeting
            - selection: the numbered event option the user selected
            - updates: the fields the user wants updated

            If the user replies "one", "first one", or "option 1"
            in response to a numbered event list, extract selection=1.
            If either value is not provided, return null for that field.
            """
        ),
    )

    selected_event: dict | None = None
    selected_event_id: str | None = None

    if result.selection is not None and state.retrived_events:
        index = result.selection - 1
        if 0 <= index < len(state.retrived_events):
            selected_event = state.retrived_events[index]

            if selected_event is not None:
                selected_event_id = (
                    selected_event.get("req_id")
                    if isinstance(selected_event, dict)
                    else selected_event.req_id
                )

        return {
            "update_email": result.email or state.update_email,
            "selected_event": selected_event or state.selected_event,
            "selected_event_id": selected_event_id or state.selected_event_id,
            "update_values": result.updates or state.update_values,
        }


def update_get_info(state: State):
    llm = init_chat_model(model=chat_model, api_key=GROQ_API_KEY)
    missing_fields = []
    if not state.selected_event:
        missing_fields.append("email")
    elif state.selected_event_id and not state.update_values:
        missing_fields.append("items to update in the meeting")

    response = llm.invoke(
        f"""
        The user wants to update an existing meeting but hasn't provided the {missing_fields} associated with the meeting.
        Keep the response short, conversational, and focused only on the {missing_fields}.
        Update fields relates to fields that the user wants to change here i.e, description, time or date.
        Do not suggest creating a new meeting."""
    )

    return {"messages": [response], "response": response.content}


def check_event_info(state: State):
    llm = init_chat_model(model=chat_model, api_key=GROQ_API_KEY)
    events = get_events_email(state.update_email)
    event_details = "\n".join(
        f"Option {index}: {event.name} | "
        f"Description: {event.event_description} | "
        f"Start: {event.start_date_time} | "
        f"End: {event.end_date_time}"
        for index, event in enumerate(events, start=1)
    )

    if len(events) == 1:
        response = llm.invoke(f"""  The calendar events that was created by the user with email{state.update_email} are:
            {event_details} ask them only which field they would like to edit donot mention any fields.
        Example:
            Option 1: Pranav | Description: Interview at Meta | Start: 2026-11-10 11:30:00+00:00 | End: 2026-11-10 12:30:00+00:00
            Then: I found one event {{option 1}} what would you like to edit?""")

        return {
            "messages": [response],
            "response": response.content,
            "selected_event_id": events[0].req_id,
            "selected_event": events[0],
            "retrived_events": events,
        }

    elif events:
        response = llm.invoke(f"""
        The calendar events that was created by the user with email{state.update_email} are:
            {event_details}
        Events are not on the users calendar, mention "Here are the results that i found for your email".
        Ask the user which event they'd want to edit based on the event name nothing else if there is more than 1 entry.
        Exclude displaying id and status to the user display any time in ist(Asia/Kolkata).
        Example:
        Option 1: Pranav | Description: Interview at Meta | Start: 2026-11-10 11:30:00+00:00 | End: 2026-11-10 12:30:00+00:00
        Option 2: Pranav | Description: Interview at Meta | Start: 2026-11-10 11:30:00+00:00 | End: 2026-11-10 12:30:00+00:00
        Then: I found {{Number of events}} event {{option number}} events which would you like to edit you can say {{number}}/options {{number}}/{{number in words}} and what wouold you like to edit?
        """)

        return {
            "retrived_events": [
                {
                    "req_id": event.req_id,
                    "name": event.name,
                    "event_description": event.event_description,
                    "start_date_time": event.start_date_time.isoformat(),
                    "end_date_time": event.end_date_time.isoformat(),
                }
                for event in events
            ],
            "messages": [response],
            "response": response.content,
        }

    else:
        response = llm.invoke(
            "The was no events scheduled on Kasturi's calendar ask the user to confirm their email or create a event"
        )

        return {"response": response.content, "messages": [response]}
