#!/bin/bash
USERNAME=$1
PROJECT_NAME=$2

for RATE in 1 1e-1 1e-2; do
  for HIDDEN in "4,8" "32,64"; do
    JOB_NAME=lr-$RATE-hiddenunits-$HIDDEN
    JOB_NAME=${JOB_NAME//,/_}
    just create job single \
        --project $USERNAME/$PROJECT_NAME \
        --name $JOB_NAME \
        --command "python -m mnist --learning_rate $RATE --hidden_units $HIDDEN" \
        --setup-command "pip install -r requirements.txt" \
        --docker-image tensorflow-1.11.0-cpu-py35 \
        --time-limit 1h \
        --instance-type c5.xlarge
    just start job $PROJECT_NAME/$JOB_NAME
    echo "Job $JOB_NAME started!"
  done
done
