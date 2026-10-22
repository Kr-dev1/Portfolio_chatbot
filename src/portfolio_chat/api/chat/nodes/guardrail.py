from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from portfolio_chat.api.chat.schema.schema import (
    TopicCheck,
    chat_model,
    guard_rail_model,
)
from portfolio_chat.api.chat.state import State
from portfolio_chat.config import DEEPSEEK_API_KEY, GROQ_API_KEY
from portfolio_chat.tools.logger.guardrail import guardrail_logger


def guard_rail_and_classiier_node(state: State):
    user_input = state.user_input

    evaluator_llm = init_chat_model(
        model=guard_rail_model, model_provider="deepseek", api_key=DEEPSEEK_API_KEY
    ).with_structured_output(TopicCheck)

    recent_messages = state.messages[-4:]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                """
                <role>
                You are an expert intent classifier for Kasturi's portfolio and calendar assistant. Kasturi is male (use he/him pronouns when relevant).
                </role>

                <context>
                The assistant helps users learn about Kasturi's professional background and manage calendar meetings with him. Your role is to classify incoming user messages accurately so that downstream components can route the request appropriately.
                </context>

                <instructions>
                Classify the user's latest input using the conversation context, ongoing calendar intent, and conversation summary.
                Output the classification using the following fields:
                - is_on_topic: true if the message pertains to Kasturi, his resume/portfolio, his work experience, freelance inquiries, or scheduling meetings; false if off-topic.
                - intent: The primary category of the request (FREELANCE, WORK, GENERAL, MEETING, or OFF_TOPIC).
                - calendar_action: For MEETING requests, specify CREATE, AVAILABILITY, UPDATE, or DELETE. For all other intents, set to null.

                CORE RULE — NEVER EVALUATE IN ISOLATION:
                Never base your classification solely on the user's latest input when prior conversation or summary exists. Minimal, fragmented, or terse responses (e.g. an email address, a single time, a name, a duration, or "yes") must be interpreted using the previous conversation and conversation summary.

                Focus exclusively on classifying the input. Provide only the classification output without answering the user's question or generating conversational responses.
                </instructions>

                <categories>
                - FREELANCE: Inquiries specifically regarding Kasturi's freelance services, freelance project history, rates, or freelance availability.
                - WORK: Inquiries regarding Kasturi's employment history, past and present employers, job titles, roles, and professional responsibilities.
                - GENERAL: Inquiries about Kasturi's personal background, education, skills, technical stack, or other information found in his resume or portfolio.
                - MEETING: Inquiries or requests related to Kasturi's schedule, bookings, or calendar.
                - OFF_TOPIC: Any query unrelated to Kasturi, his career, portfolio, or meeting scheduling.
                </categories>

                <calendar_actions>
                When intent is MEETING, assign one of the following calendar actions:
                - CREATE: Requesting or confirming a new meeting or booking.
                - AVAILABILITY: Inquiring about open time slots, free time, or schedule status.
                - UPDATE: Modifying, moving, or rescheduling an existing meeting.
                - DELETE: Canceling, deleting, or declining an existing meeting.
                </calendar_actions>

                <context_and_edge_case_rules>
                1. Context Inheritance for Minimal and Fragmented Responses:
                   When the assistant previously asked for specific information (e.g., email, name, meeting topic, call duration, date/time, time zone, or project details), the user frequently replies with only that single piece of info (e.g., "alex@example.com", "30 mins", "Friday", "Eastern", "FastAPI").
                   - Always look back at the assistant's previous question in {conversation} or {summary} to identify what was requested.
                   - The user's minimal response inherits the active topic and intent from that ongoing flow.
                   - Minimal responses answering an on-topic assistant question must have is_on_topic = true.

                2. Active Calendar Intent Flow:
                   - When an active scheduling flow is in progress ({calendar_intent}), interpret follow-up messages providing dates, times, durations, attendee details, or confirmations as continuations of that MEETING intent and action.

                3. Brief Affirmations and Confirmations:
                   - Words like "Yes", "Sure", "Sounds good", "That works", or "No, later" in response to an assistant prompt inherit the intent and context of the ongoing discussion rather than being classified as off-topic or general.
                </context_and_edge_case_rules>

                <examples>
                Example 1: New meeting request
                User Input: "Can you schedule a meeting with Kasturi?"
                Classification:
                is_on_topic = true
                intent = MEETING
                calendar_action = CREATE

                Example 2: Checking availability
                User Input: "Is Kasturi free tomorrow at 5 PM?"
                Classification:
                is_on_topic = true
                intent = MEETING
                calendar_action = AVAILABILITY

                Example 3: Rescheduling existing meeting
                User Input: "Move my meeting to Friday at 2 PM"
                Classification:
                is_on_topic = true
                intent = MEETING
                calendar_action = UPDATE

                Example 4: Canceling a meeting
                User Input: "Cancel my meeting with Kasturi"
                Classification:
                is_on_topic = true
                intent = MEETING
                calendar_action = DELETE

                Example 5: Multi-turn scheduling - time fragment
                Context: Assistant asked: "What date and time works best for you?"
                Active Calendar Intent: CREATE
                User Input: "At 5 tomorrow."
                Classification:
                is_on_topic = true
                intent = MEETING
                calendar_action = CREATE

                Example 6: Multi-turn scheduling - contact info fragment
                Context: Assistant asked: "Could you share your email address so I can send the calendar invite?"
                Active Calendar Intent: CREATE
                User Input: "alex.smith@example.com"
                Classification:
                is_on_topic = true
                intent = MEETING
                calendar_action = CREATE

                Example 7: Multi-turn scheduling - duration and agenda fragment
                Context: Assistant asked: "How long of a meeting would you like, and what is the topic?"
                Active Calendar Intent: CREATE
                User Input: "30 mins to discuss consulting on ML pipelines"
                Classification:
                is_on_topic = true
                intent = MEETING
                calendar_action = CREATE

                Example 8: Multi-turn scheduling - brief affirmation
                Context: Assistant proposed: "Does Thursday at 2 PM work for you?"
                Active Calendar Intent: CREATE
                User Input: "Yes, that works."
                Classification:
                is_on_topic = true
                intent = MEETING
                calendar_action = CREATE

                Example 9: Multi-turn freelance inquiry - tech stack fragment
                Context: Assistant asked: "What technologies are you planning to use for this freelance project?"
                User Input: "FastAPI and React"
                Classification:
                is_on_topic = true
                intent = FREELANCE
                calendar_action = null

                Example 10: Freelance inquiry
                User Input: "Does Kasturi take on freelance web development projects?"
                Classification:
                is_on_topic = true
                intent = FREELANCE
                calendar_action = null

                Example 11: Employment history inquiry
                User Input: "Where did Kasturi work before his current role?"
                Classification:
                is_on_topic = true
                intent = WORK
                calendar_action = null

                Example 12: General portfolio inquiry
                User Input: "What technologies does Kasturi specialize in?"
                Classification:
                is_on_topic = true
                intent = GENERAL
                calendar_action = null

                Example 13: Off-topic query
                User Input: "What is the capital of France?"
                Classification:
                is_on_topic = false
                intent = OFF_TOPIC
                calendar_action = null
                </examples>

                <input_context>
                Active Calendar Intent: {calendar_intent}

                Summary of Previous Conversation:
                {summary}

                Previous Conversation:
                {conversation}

                User Input:
                {user_input}
                </input_context>
                """
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
