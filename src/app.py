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

# The environment variable MODEL_SIZE is mapped to Hugging Face model IDs.
# Defaulting to tiny, matching original behavior.
model_size = os.getenv("MODEL_SIZE", "tiny")
model_id = f"openai/whisper-{model_size}"

models_dir = os.path.join(os.path.dirname(__file__), "whisper_models/")
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

device = os.getenv("OPENVINO_DEVICE", "CPU")

# Initialize OpenVINO model and processor
print(f"Loading processor for {model_id}...")
processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_dir)

print(f"Loading and exporting OpenVINO model for {model_id} on {device}...")
model = OVModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    export=True,
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
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename or ".wav")[1]) as tmp:
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
