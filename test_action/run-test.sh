#!/bin/bash

pip install pytest-cov

pytest --version

#pytest --cov=. --cov-report=html:./htmlcov tests.py
mkdir -p /test_action/htmlcov
chmod 777 /test_action/htmlcov
pytest --cov=. --cov-report=html:/test_action/htmlcov tests.py test_chat_utils.py test_llm_rag_chat.py test_llm_rag_utils.py test_service.py
