import smtplib
from email.message import EmailMessage

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage

from portfolio_chat.api.chat.schema.schema import email_model
from portfolio_chat.api.chat.state import State
from portfolio_chat.config import (
    GROQ_API_KEY,
    PRIMARY_SPACE_MAIL,
    SPACE_MAIL,
    SPACE_PASSWORD,
)


def send_email(state: State):
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


def send_contact_email(state: State):
    SMTP_SERVER = "mail.spacemail.com"
    SMTP_PORT = 465
    SENDER_EMAIL = SPACE_MAIL
    SENDER_PASSWORD = SPACE_PASSWORD
    RECIPIENT_EMAIL = PRIMARY_SPACE_MAIL, "kasturirangan20@gmail.com"

    msg = EmailMessage()

    email = f"""
        Hey Kasturi,

        Someone requested changes to a meeting that has already been approved.

        Original meeting:
        {state.selected_event}

        Requester's email:
        {state.update_email}

        Requested changes:
        {state.update_values or "No specific changes provided."}

        Please follow up with the requester to handle the change manually.

        Regards,
        Kasturi's Assistant
    """
    msg["Subject"] = "Your Request confirmation with Kasturi"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg.set_content(email)

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(PRIMARY_SPACE_MAIL, SENDER_PASSWORD)
            server.send_message(msg)
            print("Email sent  ")
            response = "The event was already approved by Kasturi, I cannot edit events that have been approved already. I've sent an email to kasturi with the event details and the contact info Kasturi will contact you shortly"
            return {"email_status": "sent", "messages": [AIMessage(content=response)], "response": response}
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
