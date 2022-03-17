# syntax = docker/dockerfile:experimental
FROM python:3.9.7

WORKDIR /app

# Python dependencies
COPY requirements-blocks.txt ./
RUN pip3 --no-cache-dir install -r requirements-blocks.txt

COPY . ./

EXPOSE 4446

ENTRYPOINT ["python3", "-u", "dsp-server.py"]
