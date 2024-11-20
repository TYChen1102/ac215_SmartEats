#!/bin/bash

echo "Container is running!!!"

args="$@"
echo "hihihi"
echo $args

if [[ -z ${args} ]]; 
then
    pipenv shell
else
  pipenv run $args
fi
