import pytest
from fastapi.testclient import TestClient
import sys
import os
import numpy as np
import soundfile as sf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_transcribe_short():
    # Generate 1 second of 440 Hz tone
    sr = 16000
    t = np.linspace(0, 1, int(sr*1), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * 440 * t)

    sf.write("dummy_short.wav", x, sr)

    with open("dummy_short.wav", "rb") as f:
        response = client.post("/transcribe", files={"audio": ("dummy_short.wav", f, "audio/wav")})

    os.remove("dummy_short.wav")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "response" in data

def test_transcribe_long():
    # Generate 35 seconds of 440 Hz tone to test chunking
    sr = 16000
    t = np.linspace(0, 35, int(sr*35), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * 440 * t)

    sf.write("dummy_long.wav", x, sr)

    with open("dummy_long.wav", "rb") as f:
        response = client.post("/transcribe", files={"audio": ("dummy_long.wav", f, "audio/wav")})

    os.remove("dummy_long.wav")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "response" in data
