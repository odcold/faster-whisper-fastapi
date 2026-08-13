import pytest
from fastapi.testclient import TestClient
import sys
import os
import numpy as np
import soundfile as sf
import torch
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock the processor and model
with patch("transformers.AutoProcessor.from_pretrained") as mock_processor, \
     patch("optimum.intel.openvino.OVModelForSpeechSeq2Seq.from_pretrained") as mock_model:

    # Configure the mocked model to return a dummy sequence of ids
    mock_model_instance = MagicMock()
    mock_model_instance.generate.return_value = [[1, 2, 3]]
    mock_model.return_value = mock_model_instance

    # Configure the mocked processor
    mock_processor_instance = MagicMock()
    # When processor is called, return dummy input features (as a torch Tensor so ones_like works)
    mock_inputs = MagicMock()
    mock_inputs.input_features = torch.tensor([[1.0, 2.0]])
    mock_processor_instance.return_value = mock_inputs
    # Decode to text
    mock_processor_instance.batch_decode.return_value = ["mocked response"]

    mock_processor.return_value = mock_processor_instance

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
    assert data["response"] == "mocked response"

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
    assert data["response"] == "mocked response mocked response"
