#!/bin/bash

pip install pytest-cov

pytest --version

#pytest --cov=. --cov-report=html:./htmlcov tests.py
pytest --cov=/test_action --cov-report=html:/test_action/htmlcov tests.py
