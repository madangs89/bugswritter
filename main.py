import asyncio
import pyaudio
import sys
import os
from dotenv import load_dotenv

from google import genai
from google.genai.types import (
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    VoiceConfig,
    SpeechConfig,
    Blob,
)

# Load variables from .env
load_dotenv()

# --- Audio Configuration ---
# Gemini processes input at 16kHz but outputs at 24kHz natively.
# We configure two separate sample rates for the mic and speaker.
FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK = 1024

async def main():
    # 1. Initialize PyAudio Streams
    p = pyaudio.PyAudio()

    print("Initializing audio streams...")
    mic_stream = p.open(format=FORMAT, channels=CHANNELS, rate=INPUT_RATE,
                        input=True, frames_per_buffer=CHUNK)
    speaker_stream = p.open(format=FORMAT, channels=CHANNELS, rate=OUTPUT_RATE,
                            output=True, frames_per_buffer=CHUNK)

    # 2. Initialize the Client
    # Depending on your .env, this automatically picks up either the Vertex AI
    # credentials (gcloud) or standard Google AI Studio (GEMINI_API_KEY).
    client = genai.Client()

    # We use the standard 2.0 flash model or the one you enabled in your project
    model_id = "gemini-live-2.5-flash-native-audio"

    # 3. Configure the Live Session (matching the reference file)
    config = LiveConnectConfig(
        response_modalities=["audio"],
        speech_config=SpeechConfig(
            voice_config=VoiceConfig(
                prebuilt_voice_config=PrebuiltVoiceConfig(
                    voice_name="Aoede", # Options: Aoede, Puck, Charon, Kore, Fenrir
                )
            )
        ),
    )

    print("Connecting to Gemini Live API...")

    # Connect to the Live API WebSocket
    async with client.aio.live.connect(model=model_id, config=config) as session:
        print("\n✅ Connected! Start speaking into your microphone.")
        print("   (Press Ctrl+C to stop the script)\n")

        # --- Task 1: Stream Mic to Gemini ---
        async def send_mic_audio():
            while True:
                try:
                    # Read from microphone
                    data = mic_stream.read(CHUNK, exception_on_overflow=False)

                    # 🚀 FIX: Use the exact syntax from the reference notebook
                    # Wrap the raw bytes in a Blob and pass it to the 'media' keyword
                    await session.send_realtime_input(
                        media=Blob(
                            data=data,
                            mime_type=f"audio/pcm;rate={INPUT_RATE}"
                        )
                    )

                    await asyncio.sleep(0.001)
                except Exception as e:
                    print(f"\nError sending audio: {e}")
                    break

        # --- Task 2: Receive Audio from Gemini ---
        async def receive_model_audio():
            loop = asyncio.get_event_loop()
            try:
                # session.receive() exhausts after each turn_complete signal.
                # Wrap in while True so we keep listening across multiple turns.
                while True:
                    async for message in session.receive():
                        server_content = message.server_content
                        if server_content is not None:

                            # Handle interruptions (Voice Activity Detection triggered)
                            if server_content.interrupted:
                                print("\n[User Interrupted Model]")

                            # Handle incoming audio chunks
                            model_turn = server_content.model_turn
                            if model_turn is not None and model_turn.parts:
                                for part in model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        # speaker_stream.write() is a blocking call — offload it to
                                        # a thread so the event loop stays free for send_mic_audio()
                                        await loop.run_in_executor(
                                            None, speaker_stream.write, part.inline_data.data
                                        )
                                        sys.stdout.write(".")
                                        sys.stdout.flush()

            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"\nError receiving audio: {e}")

        # Run both streaming tasks concurrently
        send_task = asyncio.create_task(send_mic_audio())
        receive_task = asyncio.create_task(receive_model_audio())

        await asyncio.gather(send_task, receive_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nSession ended by user. Goodbye!")
