"""Warm a set of Telugu phrases into DragonTTS (Cartesia, te-IN, μ-law 8kHz).

Run while the server is up:  uv run uvicorn app.main:app --port 8000
Then:  uv run python scripts/warm_telugu.py
"""

import re

import httpx

HOST = "http://localhost:8000"
VOICE = "3b554273-4299-48b9-9aaf-eefd438e3941"

# Phrases verbatim from the spec, separated by "." (newlines also treated as
# separators since they came in as separate paste blocks).
RAW = """మీ టైమ్ ఇచ్చినందుథ్యాంక్యూ.హావ్ ఎ నైస్ డే.
మీ ఇంట్రెస్ట్కి థ్యాంక్యూ.మరి-న్ని వివరాలతో మా డెవలపర్స్ మిమ్మల్ని కాంటాక్ట్అవుతారుహావ్ ఎ నైస్ డే.
నేనుమీ. డెంటల్క్లినిక్ ని ఆన్‌లైన్లో చూసానుమీకుమంచిరేటింగ్కూడాఉంది.కానీ వెబ్‌సైట్లేదు.మీకోసంమేముతక్కువకాస్ట్ లో వెబ్‌సైట్చేసిస్తాము
హలో నమస్తే. నా పేరుస్వాతి అండి."""

PHRASES = [p.strip() for p in re.split(r"[.\n]+", RAW) if p.strip()]

# Tuning params — MUST match the template's tts_configuration so the warmed
# entries share cache keys with clairvoyance's requests (params are in the key).
PARAMS = {
    "speed": 0.9,
    "volume": 1.1,
    "emotion": "positivity:high,curiosity:medium",
}

BODY = {
    "model_id": "cartesia:sonic-3.5",
    "voice": {"id": VOICE},
    "language": "te-IN",
    "output_format": {"container": "raw", "encoding": "mulaw", "sample_rate": 8000},
    "params": PARAMS,
}


def main() -> None:
    print(f"warming {len(PHRASES)} phrases (cartesia, voice={VOICE[:8]}…, te-IN, mulaw@8k)\n")
    ok = 0
    with httpx.Client(timeout=60.0) as client:
        for i, phrase in enumerate(PHRASES, 1):
            r = client.post(f"{HOST}/tts/create", json={**BODY, "transcript": phrase})
            if r.status_code == 200:
                j = r.json()
                ok += 1
                print(f"  [{i:>2}/{len(PHRASES)}] {j['status']:9} {j['size_bytes']:6}B "
                      f"key={j['key'][:10]}…  «{phrase[:42]}»")
            else:
                print(f"  [{i:>2}/{len(PHRASES)}] HTTP {r.status_code}: {r.text[:140]}")
    print(f"\n{ok}/{len(PHRASES)} cached.")


if __name__ == "__main__":
    main()
