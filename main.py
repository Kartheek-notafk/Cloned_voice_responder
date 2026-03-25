"""
Voice Assistant — Groq (LLaMA3) + Coqui XTTS v2 (local, unlimited, CUDA)
Type your question → get an answer in a cloned voice (100% offline TTS)
Responses are saved as numbered WAV files in the 'responses/' folder.

Usage:
    python main.py --sample path/to/voice_sample.wav
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import torch
import pygame
from groq import Groq
from TTS.api import TTS


# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "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")
GROQ_MODEL     = "llama-3.3-70b-versatile"

SYSTEM_PROMPT  = (
    "You are a helpful, conversational assistant. "
    "Keep your answers concise — 2 to 4 sentences max — "
    "because your response will be spoken aloud."
)

XTTS_MODEL     = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE       = "en"        # change if needed: "hi", "es", "fr", "te" etc.
RESPONSES_DIR  = Path("responses")   # folder where WAV files are saved
# ──────────────────────────────────────────────────────────────────────────────


def detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU detected: {name} ({vram:.1f} GB VRAM) — using CUDA")
        return "cuda"
    print("⚠️  No CUDA GPU found — falling back to CPU (slower)")
    return "cpu"


def next_response_path() -> Path:
    """Return the next numbered save path, e.g. responses/response_003.wav"""
    RESPONSES_DIR.mkdir(exist_ok=True)
    existing = sorted(RESPONSES_DIR.glob("response_*.wav"))
    next_num = len(existing) + 1
    return RESPONSES_DIR / f"response_{next_num:03d}.wav"


def init_clients(voice_sample: str):
    device = detect_device()

    print("\n🔧 Loading Groq client...")
    groq_client = Groq(api_key=GROQ_API_KEY)

    print("🔧 Loading Coqui XTTS v2 model (first run downloads ~1.8 GB)...")
    tts = TTS(XTTS_MODEL).to(device)

    # Warm up — eliminates the delay on first real request
    print("🔥 Warming up TTS engine...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        warmup_path = tmp.name
    tts.tts_to_file(
        text="Warming up.",
        speaker_wav=voice_sample,
        language=LANGUAGE,
        file_path=warmup_path,
    )
    Path(warmup_path).unlink(missing_ok=True)

    RESPONSES_DIR.mkdir(exist_ok=True)
    print(f"✅ Ready!  |  voice sample : {voice_sample}")
    print(f"💾 Saving responses to    : {RESPONSES_DIR.resolve()}\n")

    pygame.mixer.init()
    return groq_client, tts


def ask_groq(client: Groq, history: list, user_text: str) -> str:
    """Send user message to Groq and return the assistant's text reply."""
    history.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
    )

    reply = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": reply})
    return reply


def speak(tts: TTS, voice_sample: str, text: str):
    """Synthesize speech, save it as a numbered WAV, then play it."""
    print(f"🔊 Speaking: {text}\n")

    # Generate to a temp file first
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    tts.tts_to_file(
        text=text,
        speaker_wav=voice_sample,
        language=LANGUAGE,
        file_path=str(tmp_path),
    )

    # Save a permanent numbered copy
    save_path = next_response_path()
    shutil.copy2(tmp_path, save_path)
    print(f"💾 Saved → {save_path}")

    # Play the audio
    pygame.mixer.music.load(str(tmp_path))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    tmp_path.unlink(missing_ok=True)


def run(voice_sample: str):
    print("=" * 60)
    print("  🎙️  Voice Assistant  |  Groq + Coqui XTTS v2 + CUDA")
    print("  Type your question. Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    if not Path(voice_sample).exists():
        print(f"\n❌ Voice sample not found: {voice_sample}")
        print("   Pass a valid audio file with --sample\n")
        return

    groq_client, tts = init_clients(voice_sample)
    history = []

    while True:
        try:
            user_input = input("\nYou › ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Goodbye!")
            break

        print("⏳ Thinking...")
        reply = ask_groq(groq_client, history, user_input)
        speak(tts, voice_sample, reply)


def main():
    parser = argparse.ArgumentParser(
        description="Voice assistant with cloned voice — Groq + XTTS v2"
    )
    parser.add_argument(
        "--sample",
        required=True,
        help="Path to audio sample of the target person (6 sec min, 30 sec+ recommended)",
    )
    args = parser.parse_args()
    run(args.sample)


if __name__ == "__main__":
    main()