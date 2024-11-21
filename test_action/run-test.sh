#!/bin/bash

pip install pytest-cov

pytest --version

#pytest --cov=. --cov-report=html:./htmlcov tests.py
pytest --cov=/test_action/htmlcov --cov-report=html tests.py
