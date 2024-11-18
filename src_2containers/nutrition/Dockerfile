# Use the official Debian-hosted Python image
FROM python:3.11-slim-buster

# Tell pipenv where the shell is. 
# This allows us to use "pipenv shell" as a container entry point.
ENV PYENV_SHELL=/bin/bash

EXPOSE 8080

ENV GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/llm-service-account.json

# Ensure we have an up to date baseline, install dependencies
RUN set -ex; \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential git && \
    apt-get install -y python3-opencv && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv && \
    pip install flask && \
    mkdir -p /app

WORKDIR /app

# Add Pipfile, Pipfile.lock
ADD . /app/

RUN chmod -R 755 /app/secrets

RUN pipenv sync

# Entry point
ENTRYPOINT ["/bin/bash"]


# Get into the pipenv shell
CMD ["-c", "pipenv run sh run_scripts.sh"]

