#!/bin/bash

echo "Container is running!!!"


if [ "${DEV}" = 1 ]; then
  pipenv shell
else
  pipenv run python cli.py --download --load --chunk_type recursive-split
fi
