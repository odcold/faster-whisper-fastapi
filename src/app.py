import os
import shutil
import tempfile
from typing import Literal

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import Response
import uvicorn

from transformers import AutoProcessor, pipeline
from optimum.intel.openvino import OVModelForSpeechSeq2Seq

# The environment variable MODEL_SIZE is mapped to pre-exported OpenVINO INT8 models
# to prevent OOM errors during on-the-fly export and to drastically reduce memory usage.
# Existing official Intel OpenVINO int8 optimized models:
# OpenVINO/whisper-tiny-int8-ov
# OpenVINO/whisper-base-int8-ov
# OpenVINO/whisper-small-int8-ov
# OpenVINO/whisper-medium-int8-ov
# OpenVINO/whisper-large-v2-int8-ov
# OpenVINO/whisper-large-v3-int8-ov
model_size = os.getenv("MODEL_SIZE", "tiny")
model_id = f"OpenVINO/whisper-{model_size}-int8-ov"

models_dir = os.path.join(os.path.dirname(__file__), "whisper_models/")
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

device = os.getenv("OPENVINO_DEVICE", "CPU")

# Initialize OpenVINO model and processor directly from the pre-exported INT8 weights
print(f"Loading processor for {model_id}...")
processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_dir)

print(f"Loading OpenVINO INT8 model {model_id} on {device} (No OOM during load!)...")
# export=False is the default, which avoids the massive RAM spike during boot.
model = OVModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    cache_dir=models_dir,
    device=device
)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30, # required to support audio longer than 30s
)

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    response: Response, audio: UploadFile
) -> dict[Literal["response", "status"], str]:
    try:
        # Determine the file extension safely
        ext = os.path.splitext(audio.filename)[1] if audio.filename else ".wav"
        if not ext:
            ext = ".wav"

        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(audio.file, tmp)
            tmp_path = tmp.name

        try:
            # Transcribe using the pipeline
            # Passing generate_kwargs to ensure parity with the previous implementation's accuracy
            result = pipe(tmp_path, generate_kwargs={"num_beams": 5})
            text = result.get("text", "")
            return {
                "status": "ok",
                "response": text.strip(),
            }
        finally:
            # Clean up the temporary file
            os.remove(tmp_path)

    except Exception as e:
        response.status_code = 500
        return {
            "status": "error",
            "response": f"error: {e}",
        }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
