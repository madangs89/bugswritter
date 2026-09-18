import asyncio
import audioop
import json
import base64
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

# IST = UTC+5:30 — used everywhere we pass current time to Gemini
IST = timezone(timedelta(hours=5, minutes=30))

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, WebSocket, Request, Response
from google import genai
from google.genai.types import (
    LiveConnectConfig, PrebuiltVoiceConfig, VoiceConfig,
    SpeechConfig, Blob, Tool, GoogleSearch, Content, Part,
    FunctionDeclaration, Schema, Type as GType, FunctionResponse,
)
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client as TwilioClient
from dotenv import load_dotenv
import os

load_dotenv()

# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------
scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    print("⏰ Scheduler started")
    yield
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)

MODEL_ID = "gemini-live-2.5-flash-native-audio"

# ---------------------------------------------------------------------------
# Pending reminder store  {task: str}
# When a reminder call fires this is set so the greeting uses the reminder text
# ---------------------------------------------------------------------------
pending_reminder: dict = {}

# ---------------------------------------------------------------------------
# Reminder registry  { reminder_name -> {"job_id": str, "task": str, "remind_at": str} }
# Lets us look up scheduled jobs by a friendly name for cancel/reschedule
# ---------------------------------------------------------------------------
reminders: dict[str, dict] = {}



# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def mulaw_to_pcm16k(mulaw_bytes: bytes) -> bytes:
    pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
    return pcm_16k

def pcm24k_to_mulaw8k(pcm_24k: bytes) -> bytes:
    pcm_8k, _ = audioop.ratecv(pcm_24k, 2, 1, 24000, 8000, None)
    return audioop.lin2ulaw(pcm_8k, 2)


# ---------------------------------------------------------------------------
# Scheduler job — fired by APScheduler at the reminder time
# ---------------------------------------------------------------------------
async def fire_reminder(task: str):
    print(f"\n⏰ Reminder fired: {task}")
    pending_reminder["task"] = task
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post("http://localhost:8000/make-call")
            print(f"📲 Reminder call triggered: {resp.json()}")
        except Exception as e:
            print(f"Failed to trigger reminder call: {e}")


# ---------------------------------------------------------------------------
# Gemini function tools: schedule / cancel / reschedule / list reminders
# ---------------------------------------------------------------------------
SCHEDULE_REMINDER_DECL = FunctionDeclaration(
    name="schedule_reminder",
    description=(
        "Schedule a reminder call at a specific date and time. "
        "If the user has not specified a time, suggest a sensible default "
        "(e.g. 8:00 AM tomorrow for medicine, or in 1 hour for urgent tasks) "
        "and confirm with the user before scheduling."
    ),
    parameters=Schema(
        type=GType.OBJECT,
        properties={
            "task": Schema(
                type=GType.STRING,
                description="Short description of what to remind the user about.",
            ),
            "remind_at": Schema(
                type=GType.STRING,
                description=(
                    "ISO 8601 datetime string for when to send the reminder, "
                    "e.g. '2025-04-11T08:00:00'. Use the user's local timezone."
                ),
            ),
        },
        required=["task", "remind_at"],
    ),
)

CANCEL_REMINDER_DECL = FunctionDeclaration(
    name="cancel_reminder",
    description=(
        "Cancel a previously scheduled reminder. "
        "Call list_reminders first if you are unsure of the exact reminder name."
    ),
    parameters=Schema(
        type=GType.OBJECT,
        properties={
            "reminder_name": Schema(
                type=GType.STRING,
                description="The name/key of the reminder to cancel (as returned by list_reminders or schedule_reminder).",
            ),
        },
        required=["reminder_name"],
    ),
)

RESCHEDULE_REMINDER_DECL = FunctionDeclaration(
    name="reschedule_reminder",
    description=(
        "Change the time of an existing scheduled reminder. "
        "Use list_reminders first if unsure of the exact name."
    ),
    parameters=Schema(
        type=GType.OBJECT,
        properties={
            "reminder_name": Schema(
                type=GType.STRING,
                description="The name/key of the reminder to reschedule.",
            ),
            "new_remind_at": Schema(
                type=GType.STRING,
                description="New ISO 8601 datetime string, e.g. '2025-04-12T09:00:00'.",
            ),
        },
        required=["reminder_name", "new_remind_at"],
    ),
)

LIST_REMINDERS_DECL = FunctionDeclaration(
    name="list_reminders",
    description="Return all currently scheduled reminders with their names and times.",
    parameters=Schema(
        type=GType.OBJECT,
        properties={},
    ),
)

ALL_REMINDER_TOOLS = Tool(
    function_declarations=[
        SCHEDULE_REMINDER_DECL,
        CANCEL_REMINDER_DECL,
        RESCHEDULE_REMINDER_DECL,
        LIST_REMINDERS_DECL,
    ]
)


# ---------------------------------------------------------------------------
# Tool handler functions
# ---------------------------------------------------------------------------
def handle_schedule_reminder(task: str, remind_at: str) -> dict:
    """Register the reminder with APScheduler and store in registry."""
    try:
        run_time = datetime.fromisoformat(remind_at)
        # Build a friendly name from task (slug-style)
        name = task.lower().replace(" ", "_")[:40]
        # Make unique if name already taken
        if name in reminders:
            name = f"{name}_{int(run_time.timestamp())}"
        job_id = f"reminder_{name}_{int(run_time.timestamp())}"

        scheduler.add_job(
            fire_reminder,
            trigger="date",
            run_date=run_time,
            args=[task],
            id=job_id,
            replace_existing=True,
        )
        reminders[name] = {"job_id": job_id, "task": task, "remind_at": remind_at}
        print(f"✅ Reminder scheduled: '{name}' → '{task}' at {run_time}")
        return {
            "status": "scheduled",
            "reminder_name": name,
            "task": task,
            "remind_at": remind_at,
            "message": (
                f"Scheduled '{task}' at {run_time.strftime('%I:%M %p on %B %d')}. "
                f"Reminder name: '{name}'."
            ),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def handle_cancel_reminder(reminder_name: str) -> dict:
    """Cancel a reminder by its registry name."""
    entry = reminders.pop(reminder_name, None)
    if not entry:
        available = list(reminders.keys())
        return {
            "status": "not_found",
            "message": f"No reminder named '{reminder_name}'. Existing reminders: {available}",
        }
    try:
        scheduler.remove_job(entry["job_id"])
        print(f"🗑️  Reminder cancelled: '{reminder_name}'")
        return {
            "status": "cancelled",
            "reminder_name": reminder_name,
            "message": f"Your reminder '{entry['task']}' has been cancelled.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def handle_reschedule_reminder(reminder_name: str, new_remind_at: str) -> dict:
    """Reschedule an existing reminder to a new time."""
    entry = reminders.get(reminder_name)
    if not entry:
        available = list(reminders.keys())
        return {
            "status": "not_found",
            "message": f"No reminder named '{reminder_name}'. Existing reminders: {available}",
        }
    try:
        new_time = datetime.fromisoformat(new_remind_at)
        scheduler.reschedule_job(
            entry["job_id"],
            trigger="date",
            run_date=new_time,
        )
        old_time = entry["remind_at"]
        entry["remind_at"] = new_remind_at
        print(f"🔄 Reminder rescheduled: '{reminder_name}' → {new_time}")
        return {
            "status": "rescheduled",
            "reminder_name": reminder_name,
            "task": entry["task"],
            "old_remind_at": old_time,
            "new_remind_at": new_remind_at,
            "message": (
                f"Rescheduled '{entry['task']}' to "
                f"{new_time.strftime('%I:%M %p on %B %d')}."
            ),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def handle_list_reminders() -> dict:
    """Return all currently scheduled reminders."""
    if not reminders:
        return {"status": "ok", "reminders": [], "message": "You have no scheduled reminders."}
    items = [
        {"name": name, "task": v["task"], "remind_at": v["remind_at"]}
        for name, v in reminders.items()
    ]
    print(f"📋 Listed {len(items)} reminder(s)")
    return {"status": "ok", "reminders": items}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/make-call")
async def make_call(request: Request):
    # Allow optional JSON body to inject a reminder task
    try:
        body = await request.json()
        if body.get("task"):
            pending_reminder["task"] = body["task"]
    except Exception:
        pass  # no body or not JSON — normal call

    twilio_client = TwilioClient(
        os.getenv("TWILIO_ACCOUNT_SID"),
        os.getenv("TWILIO_AUTH_TOKEN"),
    )
    ngrok_url = os.getenv("NGROK_URL")
    call = twilio_client.calls.create(
        to=os.getenv("MY_NUMBER"),
        from_=os.getenv("TWILIO_FROM_NUMBER"),
        url=f"{ngrok_url}/incoming-call",
    )
    print(f"📲 Outbound call initiated: {call.sid}")
    return {"status": "calling", "call_sid": call.sid}


@app.post("/incoming-call")
async def incoming_call(request: Request):
    response = VoiceResponse()
    connect = Connect()
    ngrok_url = os.getenv("NGROK_URL")
    ws_url = ngrok_url.replace("https://", "wss://") + "/twilio-stream"
    connect.stream(url=ws_url)
    response.append(connect)
    return Response(content=str(response), media_type="application/xml")


@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    print("\n📞 Call connected!")

    # Pop the pending reminder (if any) so this call uses it once
    reminder_task = pending_reminder.pop("task", None)
    print(f"📣 Call type: {'REMINDER — ' + reminder_task if reminder_task else 'NORMAL'}")

    client = genai.Client()

    if reminder_task:
        # ── Reminder call: no tools, just speak the announcement ──────────
        SYSTEM_PROMPT = (
            "You are a warm, natural Indian multilingual female voice assistant making a scheduled reminder call. "
            "Identity: You are female. Use a warm, approachable, and confident feminine tone that is empathetic and naturally conversational. "
            "Language: Detect the user's language from their response and always reply in the same language. "
            "You speak Kannada, Hindi, Telugu, Marathi, and Indian English fluently. "
            "In Kannada use a soft, rounded tone and gentle pacing (e.g., 'haudu', 'sari', 'gotthu' naturally). "
            "In Hindi use clear consonants with a slightly rhythmic neutral Indian intonation. "
            "In Telugu use a smooth, slightly musical cadence with open vowel sounds. "
            "In Marathi use a balanced tone with slightly crisp consonants. "
            "In English use Indian English with regional influence based on detected context. "
            "Code-switch naturally if the user mixes languages (Hinglish, Kanglish, etc.). "
            "When the call connects, immediately and warmly deliver the reminder announcement in the same Language as the announcement language given., "
            "then stay on the line to answer any follow-up questions. Do not use any tools."
        )
        config = LiveConnectConfig(
            response_modalities=["audio"],
            system_instruction=Content(parts=[Part(text=SYSTEM_PROMPT)]),
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Aoede")
                )
            ),
        )
        opening_text = (
            f"Hello! This is your scheduled reminder. "
            f"Here is your alert: {reminder_task}. "
            f"Have a wonderful day!"
        )
    else:
        # ── Normal call: full assistant with all reminder tools ────────────
        SYSTEM_PROMPT = (
            "You are a warm, natural Indian multilingual female voice assistant. "
            "Identity: You are female. Use a warm, approachable, and confident feminine tone that is empathetic and naturally conversational. "
            "\n\nLanguage Adaptation:\n"
            "- Detect the user's language, accent, and regional cues from their input (words, grammar, place names).\n"
            "- Always respond in the same language the user speaks. Never switch languages unless the user does first.\n"
            "- Kannada: soft, rounded tone, gentle pacing (use 'haudu', 'sari', 'gotthu' naturally).\n"
            "- Hindi: clear consonants, slightly rhythmic flow, neutral Indian intonation.\n"
            "- Telugu: smooth, slightly musical cadence, open vowel sounds.\n"
            "- Marathi: balanced tone, slightly crisp consonants, steady pace.\n"
            "- English: Indian English with regional influence based on detected context.\n"
            "- Code-switch naturally if the user mixes languages (Hinglish, Kanglish, Tanglish, etc.).\n"
            "\n\nReminder Tools — use these tools ONLY when the user explicitly asks about reminders:\n"
            "- ONLY call a reminder tool if the user directly asks to set, reschedule, cancel, or list reminders.\n"
            "- Do NOT call any tool for general conversation, greetings, or unrelated questions.\n"
            "1. schedule_reminder — call ONLY when user explicitly asks to set or add a reminder.\n"
            "2. reschedule_reminder — call ONLY when user explicitly asks to change/move a reminder time. "
            "   If you need the reminder name, call list_reminders ONCE first, then immediately call reschedule_reminder.\n"
            "3. cancel_reminder — call ONLY when user explicitly asks to cancel or delete a reminder. "
            "   If you need the reminder name, call list_reminders ONCE first.\n"
            "4. list_reminders — call ONLY when user explicitly asks to see/list their reminders, or when needed to find a name for reschedule/cancel.\n"
            "\nRecognize reminder-related intent in any language:\n"
            "- Schedule: Kannada 'reminder haki', Hindi 'reminder lagao', Telugu 'reminder petto'\n"
            "- Reschedule: Kannada 'time change maadi/samaya badlisi', Hindi 'time badlo', Telugu 'time marchadam'\n"
            "- Cancel: Kannada 'cancel maadi/beda', Hindi 'cancel karo', Telugu 'cancel cheyyi'\n"
            "- List: Kannada 'yaavaella reminders ide', Hindi 'kya reminders hai', Telugu 'reminders chupinchu'\n"
            "If the user does not specify a time, suggest a sensible default and confirm before scheduling.\n"
            f"\nThe current date and time is: {datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}. "
            "Always interpret times the user gives as IST (Indian Standard Time, UTC+5:30) "
            "and use IST when constructing remind_at values."
        )
        config = LiveConnectConfig(
            response_modalities=["audio"],
            system_instruction=Content(parts=[Part(text=SYSTEM_PROMPT)]),
            tools=[
                Tool(google_search=GoogleSearch()),
                ALL_REMINDER_TOOLS,
            ],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Aoede")
                )
            ),
        )
        opening_text = (
            "Hello! I'm your AI assistant. I can speak Kannada, Hindi, Telugu, and English. "
            "I can help you with reminders. How can I help you today?"
        )

    async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
        stream_sid = None

        # Send opening announcement
        await session.send_client_content(
            turns=Content(role="user", parts=[Part(text=f"Please say the following aloud: {opening_text}")]),
            turn_complete=True,
        )

        # ----------------------------------------------------------------
        # Task 1: Receive from Twilio → send mic audio to Gemini
        # ----------------------------------------------------------------
        async def recv_from_twilio():
            nonlocal stream_sid
            try:
                async for raw in ws.iter_text():
                    msg = json.loads(raw)
                    event = msg.get("event")

                    if event == "start":
                        stream_sid = msg["start"]["streamSid"]
                        print(f"✅ Stream started: {stream_sid}")

                    elif event == "media":
                        mulaw = base64.b64decode(msg["media"]["payload"])
                        pcm = mulaw_to_pcm16k(mulaw)
                        await session.send_realtime_input(
                            media=Blob(data=pcm, mime_type="audio/pcm;rate=16000")
                        )

                    elif event == "stop":
                        print("📵 Caller hung up.")
                        break
            except Exception as e:
                print(f"recv error: {e}")

        # ----------------------------------------------------------------
        # Task 2: Receive from Gemini → send audio to Twilio
        #         Also handles tool calls (schedule_reminder)
        # ----------------------------------------------------------------
        async def send_to_twilio():
            # Deduplicate tool calls — same fc.id should never be processed twice
            seen_call_ids: set[str] = set()
            try:
                # session.receive() exhausts after each turn_complete signal.
                # while True keeps the conversation alive across multiple turns.
                # seen_call_ids prevents the same tool call executing twice.
                while True:
                    async for message in session.receive():

                        # --- Handle function tool calls ---
                        if message.tool_call:
                            for fc in message.tool_call.function_calls:
                                # Skip any call we have already responded to
                                if fc.id in seen_call_ids:
                                    print(f"[SKIP duplicate tool call {fc.id} — {fc.name}]")
                                    continue
                                seen_call_ids.add(fc.id)

                                args = dict(fc.args)
                                print(f"🔧 Tool call: {fc.name}({args})")

                                # GoogleSearch is a grounding tool — Gemini resolves
                                # it internally. Do NOT send tool_response for it;
                                # doing so causes a 1007 protocol error.
                                if fc.name == "search":
                                    print("🔍 Google Search handled internally by Gemini — skipping tool_response")
                                    continue

                                if fc.name == "schedule_reminder":
                                    result = handle_schedule_reminder(
                                        task=args["task"],
                                        remind_at=args["remind_at"],
                                    )
                                elif fc.name == "cancel_reminder":
                                    result = handle_cancel_reminder(
                                        reminder_name=args["reminder_name"],
                                    )
                                elif fc.name == "reschedule_reminder":
                                    result = handle_reschedule_reminder(
                                        reminder_name=args["reminder_name"],
                                        new_remind_at=args["new_remind_at"],
                                    )
                                elif fc.name == "list_reminders":
                                    result = handle_list_reminders()
                                else:
                                    result = {"status": "error", "message": f"Unknown tool: {fc.name}"}

                                await session.send_tool_response(
                                    function_responses=[
                                        FunctionResponse(
                                            name=fc.name,
                                            id=fc.id,
                                            response=result,
                                        )
                                    ]
                                )

                        # --- Handle audio output ---
                        sc = message.server_content
                        if sc:
                            if sc.interrupted:
                                print("[Interrupted]")
                            if sc.model_turn and sc.model_turn.parts:
                                for part in sc.model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        mulaw = pcm24k_to_mulaw8k(part.inline_data.data)
                                        payload = base64.b64encode(mulaw).decode()
                                        if stream_sid:
                                            await ws.send_json({
                                                "event": "media",
                                                "streamSid": stream_sid,
                                                "media": {"payload": payload},
                                            })
                                        print(".", end="", flush=True)
            except Exception as e:
                print(f"send error: {e}")

        await asyncio.gather(recv_from_twilio(), send_to_twilio())
