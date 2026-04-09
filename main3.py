import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

import torch
import pygame
from groq import Groq
from TTS.api import TTS
from num2words import num2words
from pydub import AudioSegment
import librosa
import soundfile as sf
import numpy as np
from langdetect import detect

# ── gTTS for Telugu ───────────────────────────────────────────────────────────
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("⚠️  gTTS not installed. Run: pip install gTTS")

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "(git should auto-del this for me when i upload to github, but set it in your env vars to avoid the default key rate limit. But it doesnt)")
GROQ_MODEL    = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are a helpful, conversational assistant. "
    "Keep your answers concise — 2 to 4 sentences max — "
    "because your response will be spoken aloud."
)

XTTS_MODEL    = "tts_models/multilingual/multi-dataset/xtts_v2"
RESPONSES_DIR = Path("responses")


# ── Device ────────────────────────────────────────────────────────────────────
def detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU detected: {name} ({vram:.1f} GB VRAM) — using CUDA")
        return "cuda"
    print("⚠️  No CUDA GPU found — falling back to CPU (slower)")
    return "cpu"


# ── File helpers ──────────────────────────────────────────────────────────────
def next_response_path() -> Path:
    RESPONSES_DIR.mkdir(exist_ok=True)
    existing = sorted(RESPONSES_DIR.glob("response_*.wav"))
    next_num = len(existing) + 1
    return RESPONSES_DIR / f"response_{next_num:03d}.wav"


# ── Language detection ────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    """Returns 'te' for Telugu, 'en' for everything else."""
    try:
        lang = detect(text)
        return "te" if lang == "te" else "en"
    except Exception:
        return "en"


# ── Text preprocessing ────────────────────────────────────────────────────────
def preprocess_text(text: str, lang: str = "en") -> str:
    """Convert numbers to words and expand abbreviations."""
    def replace_number(match):
        try:
            return num2words(int(match.group()))
        except Exception:
            return match.group()

    text = re.sub(r'\b\d+\b', replace_number, text)

    if lang == "en":
        abbreviations = {
            'Dr.': 'Doctor', 'Mr.': 'Mister', 'Mrs.': 'Misses',
            'Ms.': 'Ms', 'St.': 'Saint', 'etc.': 'et cetera',
            'e.g.': 'for example', 'i.e.': 'that is', 'vs.': 'versus',
            'Prof.': 'Professor', 'Ave.': 'Avenue', 'Rd.': 'Road',
            'Blvd.': 'Boulevard',
        }
        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)

    return ' '.join(text.split())


# ── Voice sample preprocessing ────────────────────────────────────────────────
def preprocess_voice_sample(sample_path: str) -> str:
    """
    Resample to 22050 Hz mono WAV — XTTS v2's native format.
    Cleaner input = better speaker embedding = higher voice accuracy.
    Returns path to a temp file.
    """
    print("🎛️  Preprocessing voice sample (22050 Hz mono)...")
    y, sr = librosa.load(sample_path, sr=None, mono=True)
    if sr != 22050:
        print(f"   Resampling {sr} Hz → 22050 Hz")
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)

    # Normalize amplitude of the sample itself
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y)) * 0.95

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, 22050)
    return tmp.name


def validate_voice_sample(sample_path: str):
    """Warn if sample is too short for good cloning."""
    try:
        duration = librosa.get_duration(path=sample_path)
        if duration < 6:
            print(f"⚠️  Sample too short ({duration:.1f}s). Minimum 6s, ideal 20–30s.")
        elif duration < 20:
            print(f"⚠️  Sample is {duration:.1f}s. 20–30s gives much better accuracy.")
        else:
            print(f"✅ Voice sample: {duration:.1f}s (great)")
    except Exception as e:
        print(f"⚠️  Could not check sample: {e}")


# ── Audio post-processing ─────────────────────────────────────────────────────
def enhance_audio(file_path: Path):
    """Normalize volume and remove low-frequency rumble."""
    try:
        audio = AudioSegment.from_wav(file_path)
        audio = audio.normalize(headroom=2.0)
        audio = audio.high_pass_filter(80)
        audio.export(file_path, format="wav")
    except Exception as e:
        print(f"⚠️  Audio enhancement failed: {e}")


# ── Telugu via gTTS ───────────────────────────────────────────────────────────
def speak_telugu_gtts(text: str, out_path: Path):
    """Google TTS for Telugu — correct pronunciation, completely free."""
    if not GTTS_AVAILABLE:
        raise RuntimeError("Install gTTS: pip install gTTS")
    import io
    tts_obj = gTTS(text=text, lang="te", slow=False)
    mp3_buf = io.BytesIO()
    tts_obj.write_to_fp(mp3_buf)
    mp3_buf.seek(0)
    AudioSegment.from_mp3(mp3_buf).export(str(out_path), format="wav")
    print("   [gTTS] Telugu audio generated")


# ── XTTS v2 speaker embedding extraction ─────────────────────────────────────
def extract_embeddings(tts: TTS, processed_sample: str):
    """
    Extract speaker embeddings ONCE at startup and reuse every call.
    This is the #1 fix for low voice accuracy — the original code called
    tts_to_file() each time which silently re-extracted embeddings,
    causing inconsistency and quality loss.
    """
    model = tts.synthesizer.tts_model
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[processed_sample],
        gpt_cond_len=30,       # use up to 30s of audio for conditioning
        gpt_cond_chunk_len=6,  # chunk size for long samples
    )
    return gpt_cond_latent, speaker_embedding


# ── XTTS v2 direct inference ──────────────────────────────────────────────────
def synthesize_xtts(
    tts: TTS,
    text: str,
    gpt_cond_latent,
    speaker_embedding,
    out_path: Path,
    speed: float = 1.0,
):
    """
    Direct model.inference() with pre-extracted embeddings.
    Key tuning for higher accuracy vs defaults:
      temperature     0.65  (default 0.85) — less hallucination, more stable tone
      rep_penalty    10.0  (default 5.0)  — fewer repeated/stuttered phonemes
      top_p           0.85               — tighter sampling, more consistent voice
    """
    model = tts.synthesizer.tts_model

    out = model.inference(
        text=text,
        language="en",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.65,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        speed=speed,
        enable_text_splitting=True,
    )

    wav = out["wav"]
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    sf.write(str(out_path), wav.astype(np.float32), 24000)


# ── Init ──────────────────────────────────────────────────────────────────────
def init_clients(voice_sample: str, speed: float = 1.0):
    device = detect_device()

    print("\n🔧 Loading Groq client...")
    groq_client = Groq(api_key=GROQ_API_KEY)

    print("🔧 Loading XTTS v2 model (first run downloads ~1.8 GB)...")
    tts = TTS(XTTS_MODEL).to(device)

    processed_sample = preprocess_voice_sample(voice_sample)

    print("🎤 Extracting speaker embeddings (once)...")
    gpt_cond_latent, speaker_embedding = extract_embeddings(tts, processed_sample)

    print("🔥 Warming up XTTS engine...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        warmup_path = Path(tmp.name)
    try:
        synthesize_xtts(
            tts, "Hello, warming up.",
            gpt_cond_latent, speaker_embedding, warmup_path, speed
        )
    except Exception as e:
        print(f"⚠️  Warmup failed (non-critical): {e}")
    finally:
        warmup_path.unlink(missing_ok=True)

    RESPONSES_DIR.mkdir(exist_ok=True)
    print(f"✅ Ready!  |  voice sample : {voice_sample}")
    print(f"💾 Saving responses to    : {RESPONSES_DIR.resolve()}\n")

    pygame.mixer.init()
    return groq_client, tts, gpt_cond_latent, speaker_embedding, processed_sample


# ── Groq LLM ──────────────────────────────────────────────────────────────────
def ask_groq(client: Groq, history: list, user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
    )
    reply = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": reply})
    return reply


# ── Speak (language router) ───────────────────────────────────────────────────
def speak(
    tts: TTS,
    text: str,
    gpt_cond_latent,
    speaker_embedding,
    speed: float = 1.0,
):
    print(f"🔊 Speaking: {text}\n")
    lang = detect_language(text)
    clean_text = preprocess_text(text, lang)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        if lang == "te":
            print("🌐 Telugu detected → gTTS (free, correct pronunciation)")
            speak_telugu_gtts(clean_text, tmp_path)
        else:
            print("🎤 English detected → XTTS v2 (voice cloned)")
            synthesize_xtts(tts, clean_text, gpt_cond_latent, speaker_embedding, tmp_path, speed)
    except Exception as e:
        print(f"❌ TTS failed: {e}")
        tmp_path.unlink(missing_ok=True)
        return

    enhance_audio(tmp_path)

    save_path = next_response_path()
    shutil.copy2(tmp_path, save_path)
    print(f"💾 Saved → {save_path}")

    pygame.mixer.music.load(str(tmp_path))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    tmp_path.unlink(missing_ok=True)


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(voice_sample: str, speed: float = 1.0):
    print("=" * 60)
    print("  🎙️  Voice Assistant  |  Groq + XTTS v2 + gTTS (Telugu)")
    print("  Type your question. Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    if not Path(voice_sample).exists():
        print(f"\n❌ Voice sample not found: {voice_sample}")
        print("   Pass a valid audio file with --sample\n")
        return

    validate_voice_sample(voice_sample)
    groq_client, tts, gpt_cond_latent, speaker_embedding, processed_sample = \
        init_clients(voice_sample, speed)

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
        speak(tts, reply, gpt_cond_latent, speaker_embedding, speed)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Voice assistant — Groq LLM + XTTS v2 (English) + gTTS (Telugu)"
    )
    parser.add_argument(
        "--sample", required=True,
        help="Path to voice sample WAV/MP3 (20–30s recommended)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Speech speed for XTTS (0.5–2.0, default=1.0)",
    )
    args = parser.parse_args()
    run(args.sample, args.speed)


if __name__ == "__main__":
    main()