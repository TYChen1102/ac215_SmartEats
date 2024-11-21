#!/bin/bash

pip install pytest-cov

pytest --version

pytest tests.py

pytest --cov=. --cov-report=html tests.py
