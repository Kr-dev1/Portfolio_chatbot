from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

guard_rail_model = "deepseek-chat"
email_model = "groq:openai/gpt-oss-20b"
chat_model = "groq:openai/gpt-oss-120b"
classifier_model = "deepseek-chat"
calendar_id = "primary"


class IntentEnum(str, Enum):
    FREELANCE = "freelance"
    WORK = "work"
    GENERAL = "general"
    MEETING = "meeting"
    OFF_TOPIC = "off_topic"


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
    done_editing: bool = Field(
        default=False,
        description=(
            "True when the user explicitly says they have no more changes "
            "to add or want to proceed with the current changes."
        ),
    )
    cancel_edit: bool = Field(
        default=False,
        description=(
            "True when the user explicitly wants to abandon the edit flow "
            "without applying any changes."
        ),
    )

    delete_confirmed: bool = Field(
        default = False,
        description=(
            'True when the user explicity says they want to delete an event'
            "and want to proceed with the current selected event"
        )
    )
