#!/bin/bash

pip install pytest-cov

pytest --version

pytest tests.py
