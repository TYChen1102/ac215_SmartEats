# Use the official Ubuntu image as the base
FROM ubuntu:20.04

# Set the environment variable for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYENV_SHELL=/bin/bash
ENV PYTHONUNBUFFERED=1

# Install required dependencies
RUN apt-get update && \
    apt-get install -y curl apt-transport-https ca-certificates gnupg lsb-release openssh-client

# Add the Google Cloud SDK repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

# Install Google Cloud SDK and additional tools
RUN apt-get update && \
    apt-get install -y google-cloud-sdk google-cloud-sdk-gke-gcloud-auth-plugin jq

# Python and required system libraries for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.9 python3-pip libgl1-mesa-glx libglib2.0-0 && \
    pip install pipenv

# Add user and set up directories
RUN useradd -ms /bin/bash app -d /home/app -u 1000 -p "$(openssl passwd -1 Passw0rd)" && \
    mkdir -p /app && \
    mkdir -p /persistent && \
    chown app:app /persistent && \
    chown app:app /app

# Switch to the new user
USER app
WORKDIR /app

# Install Python packages
ADD --chown=app:app Pipfile Pipfile.lock /app/
RUN pipenv sync

# Add the rest of the source code
ADD --chown=app:app . /app

# Entry point
ENTRYPOINT ["/bin/bash", "./docker-entrypoint.sh"]
