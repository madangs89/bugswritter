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

GET_CURRENT_TIME_DECL = FunctionDeclaration(
    name="get_current_time",
    description=(
        "Returns the current date and time in IST (Indian Standard Time). "
        "Call this whenever the user asks what time or date it is, "
        "or whenever you need the current time to compute a reminder offset "
        "(e.g. 'remind me in 10 minutes', 'tomorrow morning')."
    ),
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
        GET_CURRENT_TIME_DECL,
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


def handle_get_current_time() -> dict:
    """Return the current IST date and time — always fresh, never stale."""
    now = datetime.now(IST)
    result = {
        "status": "ok",
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S IST"),
        "date": now.strftime("%A, %d %B %Y"),
        "time": now.strftime("%I:%M %p IST"),
        "iso": now.isoformat(),
    }
    print(f"🕒 get_current_time → {result['datetime']}")
    return result

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

    # 🛡️ CRITICAL FIX: Lock to prevent Twilio WebSocket 1007 crashes
    session_lock = asyncio.Lock()

    if reminder_task:
        # ── Reminder call: no tools — just speak the announcement ──────────
        # ⚠️ Keep this prompt simple. Security/tool blocks previously caused
        #    Gemini to refuse "Please say the following aloud" with "I can't do that."
        SYSTEM_PROMPT = (
            f"You are a warm, natural Indian multilingual female voice assistant "
            f"making a scheduled reminder call.\n\n"

            f"YOUR ONLY JOB ON THIS CALL:\n"
            f"When the call connects, immediately and warmly deliver this reminder:\n"
            f"  \"{reminder_task}\"\n"
            f"Then stay on the line and answer any follow-up questions the user has.\n\n"

            f"LANGUAGE:\n"
            f"- Deliver the reminder in the same language as the reminder text.\n"
            f"- Detect the user's language from their response and always reply in that language.\n"
            f"- You speak Kannada, Hindi, Telugu, Marathi, and Indian English fluently.\n"
            f"- Code-switch naturally (Hinglish, Kanglish, Tanglish, etc.).\n\n"

            f"TONE:\n"
            f"- Warm, approachable, confident, feminine tone.\n"
            f"- Be conversational and empathetic.\n"
            f"- Do NOT use any tools during this call.\n"
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
        # Send the reminder as a direct assistant-style prompt so Gemini just speaks it
        # — no "please say this" framing that could be misread as a jailbreak attempt
        opening_text = (
            f"Hello! This is your scheduled reminder: {reminder_task}. "
            f"Have a good day"
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
            "\n⚠️ MANDATORY — Before scheduling any reminder you MUST call get_current_time first.\n"
            "Use the returned current hour to correctly determine AM or PM when the user does not say it:\n"
            "  - If current hour is 0–11 (midnight to noon) and user says a bare time like '1:20' or '2 o clock',\n"
            "    interpret it as the nearest upcoming time in that half of the day.\n"
            "  - Example: current time 01:19 AM, user says '1:20' → schedule 01:20 AM (NOT 13:20 PM).\n"
            "  - Example: current time 02:00 PM, user says '3' → schedule 03:00 PM.\n"
            "  - Always tell the user the scheduled time in 12-hour format with AM/PM before confirming.\n"
            "\nTool usage:\n"
            "1. schedule_reminder — call ONLY when user explicitly asks to set or add a reminder.\n"
            "   Always call get_current_time first to get the real current time.\n"
            "2. reschedule_reminder — call ONLY when user explicitly asks to change/move a reminder time.\n"
            "   If you need the reminder name, call list_reminders ONCE first, then call reschedule_reminder.\n"
            "3. cancel_reminder — call ONLY when user explicitly asks to cancel or delete a reminder.\n"
            "   If you need the reminder name, call list_reminders ONCE first.\n"
            "4. list_reminders — call ONLY when user explicitly asks to see their reminders,\n"
            "   or to find a reminder name for reschedule/cancel.\n"
            "5. get_current_time — returns the REAL current date and time in IST.\n"
            "   Call before scheduling to infer AM/PM and compute offsets like 'in 10 minutes', 'tomorrow 9am'.\n"
            "\nRecognize reminder intent in any language:\n"
            "- Schedule: Kannada 'reminder haki', Hindi 'reminder lagao', Telugu 'reminder petto'\n"
            "- Reschedule: Kannada 'time change maadi/samaya badlisi', Hindi 'time badlo', Telugu 'time marchadam'\n"
            "- Cancel: Kannada 'cancel maadi/beda', Hindi 'cancel karo', Telugu 'cancel cheyyi'\n"
            "- List: Kannada 'yaavaella reminders ide', Hindi 'kya reminders hai', Telugu 'reminders chupinchu'\n"
            "Always interpret times as IST (UTC+5:30) and use IST when constructing remind_at ISO values.\n\n"
            "GOOGLE SEARCH & KNOWLEDGE EXPLANATIONS (CRITICAL OVERRIDE):\n"
            "  - Google Search is enabled. Use it for facts, news, people, and things you aren't sure about.\n"
            "  - IMPORTANT: As a native voice model, you naturally default to very short, brief answers. YOU MUST OVERRIDE THIS BEHAVIOR.\n"
            "  - When explaining a topic, answering a question, or summarizing search results, YOU MUST BE HIGHLY VERBOSE AND DETAILED.\n"
            "  - MINIMUM LENGTH: You MUST speak for AT LEAST 4 to 6 full sentences when answering a knowledge question.\n"
            "  - Act like a knowledgeable tutor or podcast host giving a thorough deep-dive.\n"
            "  - NEVER give a 1-sentence summary under any circumstance when the user asks a question.\n"
            "  - NEVER read URLs aloud."
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

        # Send opening message as a model turn (not a user command).
        # Using role='model' means Gemini treats it as speech it already initiated,
        # never as a user asking it to "repeat" something — avoids refusal.
        async with session_lock:
            await session.send_client_content(
                turns=Content(role="model", parts=[Part(text=opening_text)]),
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
                        async with session_lock:
                            await session.send_realtime_input(
                                media=Blob(data=pcm, mime_type="audio/pcm;rate=16000")
                            )

                    elif event == "mark":
                        # Twilio echoes a mark only AFTER playing all audio up to that point.
                        # When we get "reminder_done" back, the announcement has fully played.
                        mark_name = msg.get("mark", {}).get("name", "")
                        if mark_name == "reminder_done":
                            print("\n📵 Announcement fully played — hanging up.")
                            await ws.close()
                            break

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

                                # GoogleSearch is a grounding tool — Gemini resolves it internally.
                                # If it hallucinates a function call named "search", we MUST respond
                                # with a safe payload, otherwise the WebSocket halts waiting for an answer.
                                if fc.name == "search":
                                    print("🔍 Grounding tool invoked — sending safe ack to keep session alive")
                                    async with session_lock:
                                        await session.send_tool_response(
                                            function_responses=[
                                                FunctionResponse(
                                                    name=fc.name,
                                                    id=fc.id,
                                                    response={"status": "handled internally"}
                                                )
                                            ]
                                        )
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
                                elif fc.name == "get_current_time":
                                    result = handle_get_current_time()
                                else:
                                    result = {"status": "error", "message": f"Unknown tool: {fc.name}"}

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

                            # For reminder calls: send a Twilio mark after all audio chunks.
                            # Twilio echoes the mark back ONLY after it has played all
                            # buffered audio — recv_from_twilio then does the actual close.
                            # This prevents hang-up mid-playback.
                            if reminder_task and sc.turn_complete:
                                print("\n✅ Gemini done — sending mark to sync hang-up")
                                if stream_sid:
                                    await ws.send_json({
                                        "event": "mark",
                                        "streamSid": stream_sid,
                                        "mark": {"name": "reminder_done"},
                                    })
                                return  # stop this task; recv_from_twilio handles close

            except Exception as e:
                print(f"send error: {e}")

        # Try block to catch timeout and cleanly close
        try:
            await asyncio.wait_for(
                asyncio.gather(recv_from_twilio(), send_to_twilio()),
                timeout=540,
            )
        except asyncio.TimeoutError:
            print("\n⏰ Session timed out. Closing gracefully.")
