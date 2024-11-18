# Use the official Debian-hosted Python image
FROM python:3.11-slim-buster

# Tell pipenv where the shell is.
# This allows us to use "pipenv shell" as a container entry point.
ENV PYENV_SHELL=/bin/bash

ENV GOOGLE_APPLICATION_CREDENTIALS=secrets/ac215-smarteat-437821-7bf746ea309f.json

# Ensure we have an up to date baseline, install dependencies
RUN set -ex; \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential git && \
    apt-get install -y python3-opencv && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv

# Add a user app
RUN useradd -ms /bin/bash app && \
    mkdir -p /app && && chmod -R 777 /app \
    chown app:app /app

# Switch to the app user
USER app

# Set the working directory to /app
WORKDIR /app

# Install python packages
#RUN pipenv --python 3.11 && \
#    pipenv install pandas numpy scikit-learn matplotlib seaborn opencv-python tensorflow keras && \
#    pipenv install google-cloud-storage

# Add Pipfile, Pipfile.lock + python code
ADD . /app/

RUN pipenv sync

# Entry point
ENTRYPOINT ["/bin/bash"]

# Get into the pipenv shell
CMD ["-c", "pipenv run python predict_food.py"]
