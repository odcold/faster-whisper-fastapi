FROM python:3.12-slim-bullseye

RUN apt-get update && apt-get install ffmpeg -y

ENV APP_HOME=/home/app/app/

RUN mkdir -p ${APP_HOME}

WORKDIR ${APP_HOME}

RUN pip install torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .
