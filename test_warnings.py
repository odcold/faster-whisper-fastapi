import librosa
import numpy as np
import torch
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
import warnings

model_id = "OpenVINO/whisper-tiny-int8-ov"
processor = AutoProcessor.from_pretrained(model_id)
model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, device="CPU")

model.generation_config.num_beams = 5
model.generation_config.task = "transcribe"

# Fix the SuppressTokens warnings by clearing the built-in properties
if hasattr(model.generation_config, "suppress_tokens"):
    model.generation_config.suppress_tokens = None
if hasattr(model.generation_config, "begin_suppress_tokens"):
    model.generation_config.begin_suppress_tokens = None

# Generate 1 sec of audio
sr = 16000
t = np.linspace(0, 1, int(sr*1), endpoint=False)
chunk = 0.5 * np.sin(2 * np.pi * 440 * t)

inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")

# Fix attention mask warning
attention_mask = torch.ones_like(inputs.input_features)

print("--- GENERATING ---")
predicted_ids = model.generate(
    inputs.input_features,
    attention_mask=attention_mask
)
print("--- DONE ---")
