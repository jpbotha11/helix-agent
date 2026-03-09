FROM python:3.11-slim

RUN useradd -m sandbox
USER sandbox

WORKDIR /sandbox
ENV PYTHONUNBUFFERED=1