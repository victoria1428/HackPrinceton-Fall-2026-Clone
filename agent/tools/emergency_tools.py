import json
import os
from langchain_core.tools import tool

# Local contact store (can also be loaded from a JSON file on the Pi)
CONTACTS_DB = {
    "default_resident": {
        "name": "Alex Rivera",
        "medical_notes": "History of syncope (fainting), allergic to penicillin",
        "contacts": [
            {"name": "Jordan Rivera (Spouse)", "phone": "+16673695850", "priority": 1},
            {"name": "Princeton Campus EMS (Demo)", "phone": "+15550109911", "priority": 2},
        ],
    }
}


@tool
def lookup_emergency_contacts(resident_id: str = "default_resident") -> str:
    """
    Searches the resident's profile for verified emergency contacts, phone numbers,
    and critical medical notes to relay to first responders.
    """
    profile = CONTACTS_DB.get(resident_id, CONTACTS_DB["default_resident"])
    return json.dumps(profile, indent=2)


@tool
def dispatch_emergency_call(phone_number: str, contact_name: str, incident_summary: str) -> str:
    """
    Places an automated one-way emergency voice call and SMS via Twilio to the
    specified contact with the incident summary and medical notes.
    """
    print(f"[TOOL EXECUTED] Calling {contact_name} ({phone_number}) -> {incident_summary}")

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_PHONE_NUMBER")

    # If Twilio credentials are configured, place the real live demo call & SMS
    if account_sid and auth_token and from_number:
        try:
            from twilio.rest import Client

            client = Client(account_sid, auth_token)
            twiml = (
                "<Response>"
                "<Pause length='1'/>"
                f"<Say voice='alice'>Emergency Alert for {contact_name}. {incident_summary}</Say>"
                "<Pause length='1'/>"
                f"<Say voice='alice'>Repeating: {incident_summary}</Say>"
                "</Response>"
            )
            call = client.calls.create(to=phone_number, from_=from_number, twiml=twiml)
            client.messages.create(
                to=phone_number,
                from_=from_number,
                body=f"EMERGENCY ALERT ({contact_name}): {incident_summary}",
            )
            return f"Twilio call (SID: {call.sid}) and SMS dispatched to {contact_name} at {phone_number}."
        except Exception as e:
            print(f"[Twilio Error] {e}")
            return f"Logged emergency alert for {contact_name}, but Twilio failed: {e}"

    return f"Simulated emergency call & SMS dispatched to {contact_name} at {phone_number}."