FROM python:3.12-slim-bullseye

# Install ffmpeg for audio processing and OpenCL drivers for Intel GPU support in OpenVINO
RUN apt-get update && apt-get install -y ffmpeg intel-opencl-icd ocl-icd-libopencl1 && rm -rf /var/lib/apt/lists/*

ENV APP_HOME=/home/app/app/

RUN mkdir -p ${APP_HOME}

WORKDIR ${APP_HOME}

RUN pip install torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .
