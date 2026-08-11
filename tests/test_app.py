import pytest
from fastapi.testclient import TestClient
import sys
import os
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock the pipeline globally before importing app so it doesn't download models during tests
with patch("transformers.AutoProcessor.from_pretrained") as mock_processor, \
     patch("optimum.intel.openvino.OVModelForSpeechSeq2Seq.from_pretrained") as mock_model, \
     patch("transformers.pipeline") as mock_pipeline:

    # Configure the mocked pipeline to return a fake transcription result
    mock_pipe_instance = MagicMock()
    mock_pipe_instance.return_value = {"text": "mocked response"}
    mock_pipeline.return_value = mock_pipe_instance

    from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_transcribe_short():
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
