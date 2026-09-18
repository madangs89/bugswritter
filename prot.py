import asyncio
import audioop
import json
import base64
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

# IST = UTC+5:30
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
# Scheduler
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
# Global state
# ---------------------------------------------------------------------------

# { task: str } — set just before a reminder call fires
pending_reminder: dict = {}

# { reminder_name -> { job_id, task, remind_at } }
reminders: dict[str, dict] = {}

# Per-WebSocket cancel guard — prevents mass-cancel in one model turn
# { ws_id -> { batch_id: str, count: int } }
_cancel_guards: dict[int, dict] = {}


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
# APScheduler job — fires the reminder call
# ---------------------------------------------------------------------------
async def fire_reminder(task: str):
    print(f"\n⏰ Reminder fired: '{task}'")
    pending_reminder["task"] = task
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post("http://localhost:8000/make-call")
            print(f"📲 Reminder call result: {resp.json()}")
        except Exception as e:
            print(f"❌ Failed to trigger reminder call: {e}")


# ---------------------------------------------------------------------------
# Tool declarations
# ---------------------------------------------------------------------------
SCHEDULE_REMINDER_DECL = FunctionDeclaration(
    name="schedule_reminder",
    description=(
        "Schedule a reminder call at a specific future date and time. "
        "ALWAYS call get_current_time first to know the real current time before scheduling. "
        "If the user has not specified a time, suggest a sensible default and confirm before scheduling. "
        "Never schedule in the past."
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
                    "ISO 8601 datetime WITH timezone offset, e.g. '2025-04-11T08:00:00+05:30'. "
                    "Always include +05:30 for IST."
                ),
            ),
        },
        required=["task", "remind_at"],
    ),
)

CANCEL_REMINDER_DECL = FunctionDeclaration(
    name="cancel_reminder",
    description=(
        "Cancel ONE specific previously scheduled reminder by its exact name. "
        "Call list_reminders first if you do not know the exact reminder_name. "
        "NEVER cancel more than one reminder per user request without explicit per-reminder confirmation."
    ),
    parameters=Schema(
        type=GType.OBJECT,
        properties={
            "reminder_name": Schema(
                type=GType.STRING,
                description="Exact reminder name key as returned by list_reminders or schedule_reminder.",
            ),
        },
        required=["reminder_name"],
    ),
)

RESCHEDULE_REMINDER_DECL = FunctionDeclaration(
    name="reschedule_reminder",
    description=(
        "Change the time of ONE existing reminder. "
        "Call get_current_time first to verify the new time is in the future. "
        "Call list_reminders first if you are unsure of the reminder_name. "
        "NEVER follow a successful reschedule with cancel_reminder + schedule_reminder."
    ),
    parameters=Schema(
        type=GType.OBJECT,
        properties={
            "reminder_name": Schema(
                type=GType.STRING,
                description="Exact name of the reminder to reschedule.",
            ),
            "new_remind_at": Schema(
                type=GType.STRING,
                description="New ISO 8601 datetime with +05:30 offset, e.g. '2025-04-12T09:00:00+05:30'.",
            ),
        },
        required=["reminder_name", "new_remind_at"],
    ),
)

LIST_REMINDERS_DECL = FunctionDeclaration(
    name="list_reminders",
    description=(
        "Return all currently scheduled reminders with their names and times. "
        "Call this ONLY when the user explicitly asks to see reminders, "
        "or when you need the exact reminder_name for a cancel/reschedule. "
        "NEVER call this to verify after a successful operation."
    ),
    parameters=Schema(type=GType.OBJECT, properties={}),
)

GET_CURRENT_TIME_DECL = FunctionDeclaration(
    name="get_current_time",
    description=(
        "Returns the REAL current date and time in IST (Indian Standard Time, UTC+5:30). "
        "ALWAYS call this before scheduling or rescheduling any reminder. "
        "Use it to resolve relative times like 'in 10 minutes', 'tomorrow 9am', or bare times like '3 o clock'."
    ),
    parameters=Schema(type=GType.OBJECT, properties={}),
)

ALL_REMINDER_TOOLS = Tool(
    function_declarations=[
        GET_CURRENT_TIME_DECL,          # listed first — model calls it first
        SCHEDULE_REMINDER_DECL,
        CANCEL_REMINDER_DECL,
        RESCHEDULE_REMINDER_DECL,
        LIST_REMINDERS_DECL,
    ]
)

# GoogleSearch grounding tool — real Google results handled internally by Gemini
GOOGLE_SEARCH_TOOL = Tool(google_search=GoogleSearch())

# Known function tool names — everything else gets an ack response to avoid 1007
KNOWN_FUNCTION_TOOLS = {
    "get_current_time",
    "schedule_reminder",
    "cancel_reminder",
    "reschedule_reminder",
    "list_reminders",
}


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _now_ist() -> datetime:
    return datetime.now(IST)

def _make_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt

def _eta_string(delta_seconds: float) -> str:
    total_mins = int(delta_seconds // 60)
    if total_mins < 1:
        return "in less than a minute"
    elif total_mins < 60:
        return f"in {total_mins} minute(s)"
    elif total_mins < 1440:
        h, m = divmod(total_mins, 60)
        return f"in {h}h {m}m"
    else:
        return f"in {total_mins // 1440} day(s)"


def handle_get_current_time() -> dict:
    now = _now_ist()
    result = {
        "status": "ok",
        "datetime_ist": now.strftime("%Y-%m-%d %H:%M:%S IST"),
        "date": now.strftime("%A, %d %B %Y"),
        "time_12h": now.strftime("%I:%M %p IST"),
        "time_24h": now.strftime("%H:%M IST"),
        "iso": now.isoformat(),
        "note": "All times are IST (UTC+5:30). Use this as reference for all scheduling.",
    }
    print(f"🕒 get_current_time → {result['datetime_ist']}")
    return result


def handle_schedule_reminder(task: str, remind_at: str) -> dict:
    try:
        run_time = _make_aware(datetime.fromisoformat(remind_at))
        now = _now_ist()

        if run_time <= now:
            diff_mins = int((now - run_time).total_seconds() // 60)
            return {
                "status": "error",
                "code": "PAST_TIME",
                "message": (
                    f"Cannot schedule '{task}' at {run_time.strftime('%I:%M %p on %d %B')} — "
                    f"that was {diff_mins} minute(s) ago. "
                    f"Current IST time is {now.strftime('%I:%M %p')}. "
                    f"Ask the user for a FUTURE time."
                ),
            }

        if (run_time - now).days > 365:
            return {
                "status": "error",
                "code": "TOO_FAR",
                "message": (
                    f"The time {run_time.strftime('%I:%M %p on %d %B %Y')} is more than a year away. "
                    f"Confirm with the user if this is intentional."
                ),
            }

        # Build unique slug name
        base_name = task.lower().replace(" ", "_")[:40]
        name = base_name
        counter = 1
        while name in reminders:
            name = f"{base_name}_{counter}"
            counter += 1

        job_id = f"reminder_{name}_{int(run_time.timestamp())}"
        scheduler.add_job(
            fire_reminder,
            trigger="date",
            run_date=run_time,
            args=[task],
            id=job_id,
            replace_existing=True,
        )
        reminders[name] = {"job_id": job_id, "task": task, "remind_at": run_time.isoformat()}

        eta = _eta_string((run_time - now).total_seconds())
        print(f"✅ Scheduled: '{name}' → '{task}' at {run_time} ({eta})")
        return {
            "status": "scheduled",
            "reminder_name": name,
            "task": task,
            "remind_at_human": run_time.strftime("%I:%M %p on %A, %d %B %Y"),
            "eta": eta,
            "message": (
                f"✅ Scheduled '{task}' at {run_time.strftime('%I:%M %p on %d %B')} ({eta}). "
                f"Reminder name: '{name}'. "
                f"STOP — do NOT call schedule_reminder again for this task."
            ),
        }
    except ValueError as e:
        return {"status": "error", "code": "PARSE_ERROR", "message": f"Could not parse datetime '{remind_at}': {e}"}
    except Exception as e:
        return {"status": "error", "code": "UNKNOWN", "message": str(e)}


def handle_cancel_reminder(reminder_name: str, ws_id: int, batch_id: str) -> dict:
    # Per-session, per-batch cancel guard (max 1 cancel per model turn)
    guard = _cancel_guards.setdefault(ws_id, {"batch_id": None, "count": 0})
    if guard["batch_id"] == batch_id and guard["count"] >= 1:
        return {
            "status": "blocked",
            "code": "MULTI_CANCEL_BLOCKED",
            "message": (
                "Only ONE cancellation is allowed per user request. "
                "Tell the user which reminder was cancelled and ask them "
                "to confirm each additional cancellation separately."
            ),
        }
    guard["batch_id"] = batch_id
    guard["count"] += 1

    # Fuzzy name match — handle minor model hallucinations
    actual_name = reminder_name
    if reminder_name not in reminders:
        matches = [k for k in reminders if k.lower() == reminder_name.lower()]
        if len(matches) == 1:
            actual_name = matches[0]
        else:
            return {
                "status": "not_found",
                "code": "NOT_FOUND",
                "message": (
                    f"No reminder named '{reminder_name}'. "
                    f"Available: {list(reminders.keys())}. "
                    f"Call list_reminders to see exact names."
                ),
            }

    entry = reminders.pop(actual_name)
    try:
        scheduler.remove_job(entry["job_id"])
    except Exception:
        pass  # job may have already fired

    print(f"🗑️  Cancelled: '{actual_name}' ('{entry['task']}')")
    return {
        "status": "cancelled",
        "reminder_name": actual_name,
        "task": entry["task"],
        "message": (
            f"✅ Cancelled reminder '{entry['task']}'. "
            f"STOP — do NOT cancel any other reminders unless the user explicitly asks again."
        ),
    }


def handle_reschedule_reminder(reminder_name: str, new_remind_at: str) -> dict:
    # Fuzzy match
    actual_name = reminder_name
    if reminder_name not in reminders:
        matches = [k for k in reminders if k.lower() == reminder_name.lower()]
        if len(matches) == 1:
            actual_name = matches[0]
        else:
            return {
                "status": "not_found",
                "code": "NOT_FOUND",
                "message": f"No reminder named '{reminder_name}'. Available: {list(reminders.keys())}.",
            }

    entry = reminders[actual_name]
    try:
        new_time = _make_aware(datetime.fromisoformat(new_remind_at))
        now = _now_ist()

        if new_time <= now:
            diff_mins = int((now - new_time).total_seconds() // 60)
            return {
                "status": "error",
                "code": "PAST_TIME",
                "message": (
                    f"Cannot reschedule to {new_time.strftime('%I:%M %p')} — "
                    f"that was {diff_mins} minute(s) ago. "
                    f"Current time: {now.strftime('%I:%M %p IST')}. "
                    f"Ask the user for a FUTURE time. "
                    f"Do NOT cancel and reschedule manually."
                ),
            }

        old_time_str = entry["remind_at"]
        scheduler.reschedule_job(entry["job_id"], trigger="date", run_date=new_time)
        entry["remind_at"] = new_time.isoformat()

        eta = _eta_string((new_time - now).total_seconds())
        print(f"🔄 Rescheduled: '{actual_name}' → {new_time} ({eta})")
        return {
            "status": "rescheduled",
            "reminder_name": actual_name,
            "task": entry["task"],
            "old_remind_at": old_time_str,
            "new_remind_at_human": new_time.strftime("%I:%M %p on %A, %d %B %Y"),
            "eta": eta,
            "message": (
                f"✅ Rescheduled '{entry['task']}' to "
                f"{new_time.strftime('%I:%M %p on %d %B')} ({eta}). "
                f"STOP — do NOT call cancel_reminder or schedule_reminder."
            ),
        }
    except ValueError as e:
        return {"status": "error", "code": "PARSE_ERROR", "message": f"Could not parse '{new_remind_at}': {e}"}
    except Exception as e:
        return {"status": "error", "code": "UNKNOWN", "message": str(e)}


def handle_list_reminders() -> dict:
    if not reminders:
        return {
            "status": "ok",
            "count": 0,
            "reminders": [],
            "message": "You have no scheduled reminders.",
        }
    now = _now_ist()
    items = []
    for name, v in reminders.items():
        try:
            remind_dt = _make_aware(datetime.fromisoformat(v["remind_at"]))
            delta = (remind_dt - now).total_seconds()
            eta = "overdue" if delta < 0 else _eta_string(delta)
            human = remind_dt.strftime("%I:%M %p on %A, %d %B")
        except Exception:
            eta = ""
            human = v["remind_at"]
        items.append({"name": name, "task": v["task"], "remind_at": human, "eta": eta})

    print(f"📋 Listed {len(items)} reminder(s)")
    return {
        "status": "ok",
        "count": len(items),
        "reminders": items,
        "message": f"You have {len(items)} reminder(s). STOP — do NOT call list_reminders again.",
    }


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
REMINDER_SYSTEM_PROMPT = (
    "You are a warm, natural Indian multilingual female voice assistant making a scheduled reminder call. "
    "Identity: You are female. Use a warm, approachable, confident feminine tone. "
    "Language: Detect language from the user and always reply in the same language. "
    "You speak Kannada, Hindi, Telugu, Marathi, and Indian English fluently. "
    "Code-switch naturally for Hinglish, Kanglish, Tanglish, etc. "
    "When the call connects, immediately and warmly deliver the reminder, then answer follow-up questions. "
    "Do NOT use any tools."
)

NORMAL_SYSTEM_PROMPT = (
    "You are a warm, natural Indian multilingual female voice assistant. "
    "Use a warm, approachable, confident feminine tone.\n\n"

    "LANGUAGE RULES:\n"
    "- Detect language from first words spoken. Reply in the SAME language always.\n"
    "- Kannada: soft rounded tone ('haudu', 'sari', 'gotthu').\n"
    "- Hindi: clear consonants, rhythmic flow.\n"
    "- Telugu: smooth musical cadence, open vowels.\n"
    "- Marathi: balanced, slightly crisp consonants.\n"
    "- English: Indian English with regional warmth.\n"
    "- Code-switch naturally (Hinglish, Kanglish, Tanglish).\n\n"

    "══════════════════════════════════════════════\n"
    "TOOL DISCIPLINE — MANDATORY, NO EXCEPTIONS\n"
    "══════════════════════════════════════════════\n\n"

    "BEFORE ANY SCHEDULING/RESCHEDULING:\n"
    "  → ALWAYS call get_current_time FIRST. Never assume the time.\n"
    "  → Use the returned 'iso' field as your now-reference.\n"
    "  → Resolve relative times ('in 10 min', 'tomorrow 9am', '3 o clock') using the real current time.\n"
    "  → When user says a bare time like '3' or '1:20', pick the nearest future occurrence.\n"
    "  → Always confirm time with user in 12-hour format with AM/PM before calling schedule_reminder.\n"
    "  → Always include +05:30 timezone offset in remind_at ISO strings.\n\n"

    "SCHEDULING RULES:\n"
    "  1. Call schedule_reminder EXACTLY ONCE per reminder task.\n"
    "  2. If status=scheduled → STOP. Tell user the reminder is set. Done.\n"
    "  3. If status=error/PAST_TIME → tell user the time passed, ask for a new time.\n"
    "  4. NEVER call schedule_reminder twice for the same task.\n\n"

    "CANCELLING RULES:\n"
    "  1. Cancel EXACTLY ONE reminder per user request.\n"
    "  2. If user says 'cancel all' or refers to multiple → ask them to confirm each one by name separately.\n"
    "  3. If status=cancelled → STOP. Do NOT cancel more in the same turn.\n"
    "  4. If status=blocked → tell user one was cancelled, ask which else they want cancelled.\n"
    "  5. NEVER loop cancel_reminder across multiple reminders in one turn.\n\n"

    "RESCHEDULING RULES:\n"
    "  1. Call get_current_time, then reschedule_reminder ONCE.\n"
    "  2. If status=rescheduled → STOP. Never follow with cancel+schedule.\n"
    "  3. If status=error/PAST_TIME → ask user for a future time.\n\n"

    "LISTING RULES:\n"
    "  1. Call list_reminders ONLY when user explicitly asks to see reminders,\n"
    "     or when you need a reminder_name you don't know.\n"
    "  2. NEVER call list_reminders to verify after a successful operation.\n"
    "  3. After list_reminders returns → STOP. Do not call it again.\n\n"

    "GOOGLE SEARCH & DEPTH RULES:\n"
    "  - Google Search runs internally. You do NOT call it manually — it is triggered automatically.\n"
    "  - NEVER read URLs aloud.\n\n"

    "  WHEN TO USE GOOGLE SEARCH — MANDATORY:\n"
    "  Always trigger Google Search (do NOT answer from memory alone) when:\n"
    "    ✦ You are UNSURE or your knowledge might be OUTDATED — search first, then answer.\n"
    "    ✦ The topic is CURRENT EVENTS, NEWS, SPORTS SCORES, STOCK PRICES, WEATHER, recent releases.\n"
    "    ✦ The user asks about PEOPLE, COMPANIES, PLACES, PRODUCTS you don't have full info on.\n"
    "    ✦ The user asks 'what is happening', 'latest', 'recent', 'today', 'this week/month/year'.\n"
    "    ✦ The answer requires UP-TO-DATE facts that may have changed since your training.\n"
    "  NEVER say 'I don't know', 'I'm not sure', 'my knowledge is limited', or 'I can't answer that'.\n"
    "  Instead: use Google Search, get the real answer, and present it confidently.\n"
    "  If Google Search also returns no result, say: 'I searched but couldn't find reliable info on that.'\n\n"

    "  DEFAULT (casual questions): Give a clear 2-4 sentence answer.\n\n"

    "  DETAILED MODE — activate when user explicitly asks for depth:\n"
    "    Trigger words (any language):\n"
    "      English  : 'in detail', 'tell me more', 'full story', 'explain', 'elaborate', 'deep dive'\n"
    "      Kannada  : 'visheshavagi', 'poora heḷi', 'sariyaagi heḷi', 'full story heḷi'\n"
    "      Hindi    : 'detail mein batao', 'poora batao', 'samjhao', 'vistaar se'\n"
    "      Telugu   : 'vistaaramga cheppu', 'anni cheppu', 'full ga cheppu'\n"
    "      Marathi  : 'saangaa sarva', 'vistaraat saangaa'\n\n"

    "  When DETAILED MODE is triggered for NEWS or EVENTS, structure your answer like a journalist:\n"
    "    1. HEADLINE — What happened? (1-2 sentences)\n"
    "    2. WHO & WHERE — Key people, organizations, locations involved.\n"
    "    3. WHAT & WHY — Full explanation of what exactly happened and why it matters.\n"
    "    4. WHEN — Timeline of events if relevant.\n"
    "    5. OUTCOME / IMPACT — What are the consequences or results?\n"
    "    6. WHAT'S NEXT — Any upcoming developments or open questions.\n"
    "  Speak each section naturally as flowing conversation — no bullet reads, no 'section 1' announcements.\n"
    "  Aim for a thorough 1-2 minute verbal response for detailed requests. Do NOT cut short.\n\n"

    "  When DETAILED MODE is triggered for GENERAL KNOWLEDGE or EXPLANATIONS:\n"
    "    - Cover background/context, main facts, why it matters, and any nuances.\n"
    "    - Use analogies if they help understanding.\n"
    "    - Speak naturally across multiple sentences — like an informed friend explaining to you.\n\n"

    "  AFTER any detailed answer, ask: 'Would you like me to go deeper on any specific part?'\n\n"

    "GENERAL:\n"
    "  - Trust tool responses. status=scheduled/rescheduled/cancelled means success.\n"
    "  - Never call tools for normal conversation or greetings.\n\n"

    "LANGUAGE REMINDER KEYWORDS:\n"
    "  Schedule: 'reminder haki' (Kn), 'reminder lagao' (Hi), 'reminder petto' (Te)\n"
    "  Reschedule: 'time badlisi' (Kn), 'time badlo' (Hi), 'time marchadam' (Te)\n"
    "  Cancel: 'cancel maadi/beda' (Kn), 'cancel karo' (Hi), 'cancel cheyyi' (Te)\n"
    "  List: 'yaavaella reminders ide' (Kn), 'kya reminders hai' (Hi), 'reminders chupinchu' (Te)\n"
)
# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/make-call")
async def make_call(request: Request):
    try:
        body = await request.json()
        if body.get("task"):
            pending_reminder["task"] = body["task"]
    except Exception:
        pass

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
    ws_id = id(ws)
    print(f"\n📞 Call connected! (ws_id={ws_id})")

    reminder_task = pending_reminder.pop("task", None)
    print(f"📣 Call type: {'REMINDER — ' + reminder_task if reminder_task else 'NORMAL'}")

    client = genai.Client()
    session_lock = asyncio.Lock()

    if reminder_task:
        config = LiveConnectConfig(
            response_modalities=["audio"],
            system_instruction=Content(parts=[Part(text=REMINDER_SYSTEM_PROMPT)]),
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Aoede")
                )
            ),
        )
        opening_text = (
            f"Hello! This is your scheduled reminder: {reminder_task}. "
            f"Have a wonderful day! Let me know if you need anything."
        )
    else:
        config = LiveConnectConfig(
            response_modalities=["audio"],
            system_instruction=Content(parts=[Part(text=NORMAL_SYSTEM_PROMPT)]),
            tools=[
                GOOGLE_SEARCH_TOOL,     # real Google grounding — Gemini handles internally
                ALL_REMINDER_TOOLS,     # reminder function tools — we handle these
            ],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Aoede")
                )
            ),
        )
        opening_text = (
            "Hello! I'm your AI assistant. I can speak Kannada, Hindi, Telugu, English and Marathi. "
            "I can help you set reminders and answer questions. How can I help you today?"
        )

    try:
        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            stream_sid = None

            # Send opening message inside the lock for safety
            async with session_lock:
                await session.send_client_content(
                    turns=Content(
                        role="user",
                        parts=[Part(text=f"Please say the following aloud: {opening_text}")]
                    ),
                    turn_complete=True,
                )

            # ----------------------------------------------------------------
            # Task 1 — Twilio mic audio → Gemini
            # ----------------------------------------------------------------
            async def recv_from_twilio():
                nonlocal stream_sid
                try:
                    async for raw in ws.iter_text():
                        msg = json.loads(raw)
                        event = msg.get("event")

                        if event == "start":
                            stream_sid = msg["start"]["streamSid"]
                            _cancel_guards[ws_id] = {"batch_id": None, "count": 0}
                            print(f"✅ Stream started: {stream_sid}")

                        elif event == "media":
                            mulaw = base64.b64decode(msg["media"]["payload"])
                            pcm = mulaw_to_pcm16k(mulaw)
                            async with session_lock:
                                await session.send_realtime_input(
                                    media=Blob(data=pcm, mime_type="audio/pcm;rate=16000")
                                )

                        elif event == "stop":
                            print("📵 Caller hung up.")
                            break

                except Exception as e:
                    print(f"recv_from_twilio error: {e}")

            # ----------------------------------------------------------------
            # Task 2 — Gemini → Twilio audio + all tool handling
            # ----------------------------------------------------------------
            async def send_to_twilio():
                seen_call_ids: set[str] = set()
                try:
                    while True:
                        async for message in session.receive():

                            # ── Tool calls ───────────────────────────────
                            if message.tool_call:
                                fcs = message.tool_call.function_calls
                                batch_id = fcs[0].id if fcs else "unknown"

                                # Reset cancel guard for every new model turn batch
                                if ws_id in _cancel_guards:
                                    _cancel_guards[ws_id]["batch_id"] = None
                                    _cancel_guards[ws_id]["count"] = 0

                                for fc in fcs:
                                    if fc.id in seen_call_ids:
                                        print(f"[SKIP duplicate {fc.id} — {fc.name}]")
                                        continue
                                    seen_call_ids.add(fc.id)

                                    args = dict(fc.args)
                                    print(f"🔧 Tool: {fc.name}({args})")

                                    # ── Google Search grounding ──────────
                                    # Gemini resolves it internally but still
                                    # emits a function_call event. We MUST
                                    # send a tool_response or the session dies
                                    # with 1007 after a few minutes.
                                    if fc.name not in KNOWN_FUNCTION_TOOLS:
                                        print(f"🔍 Grounding/unknown tool '{fc.name}' — sending ack to keep session alive")
                                        async with session_lock:
                                            await session.send_tool_response(
                                                function_responses=[
                                                    FunctionResponse(
                                                        name=fc.name,
                                                        id=fc.id,
                                                        response={"status": "ok"},
                                                    )
                                                ]
                                            )
                                        continue

                                    # ── Reminder function tools ──────────
                                    if fc.name == "get_current_time":
                                        result = handle_get_current_time()

                                    elif fc.name == "schedule_reminder":
                                        result = handle_schedule_reminder(
                                            task=args.get("task", ""),
                                            remind_at=args.get("remind_at", ""),
                                        )

                                    elif fc.name == "cancel_reminder":
                                        result = handle_cancel_reminder(
                                            reminder_name=args.get("reminder_name", ""),
                                            ws_id=ws_id,
                                            batch_id=batch_id,
                                        )

                                    elif fc.name == "reschedule_reminder":
                                        result = handle_reschedule_reminder(
                                            reminder_name=args.get("reminder_name", ""),
                                            new_remind_at=args.get("new_remind_at", ""),
                                        )

                                    elif fc.name == "list_reminders":
                                        result = handle_list_reminders()

                                    else:
                                        result = {
                                            "status": "error",
                                            "message": f"Unknown tool: {fc.name}",
                                        }

                                    async with session_lock:
                                        await session.send_tool_response(
                                            function_responses=[
                                                FunctionResponse(
                                                    name=fc.name,
                                                    id=fc.id,
                                                    response=result,
                                                )
                                            ]
                                        )

                            # ── Audio output ──────────────────────────────
                            sc = message.server_content
                            if sc:
                                if sc.interrupted:
                                    print("\n[User interrupted model]")
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
                    print(f"\nsend_to_twilio error: {e}")

            # 9-min timeout — Gemini Live sessions max out around 10 min
            try:
                await asyncio.wait_for(
                    asyncio.gather(recv_from_twilio(), send_to_twilio()),
                    timeout=540,
                )
            except asyncio.TimeoutError:
                print("\n⏰ 9-min session limit reached — hanging up gracefully.")

    except Exception as e:
        print(f"Session error: {e}")
    finally:
        _cancel_guards.pop(ws_id, None)
        print(f"\n🔚 Session closed (ws_id={ws_id})")
