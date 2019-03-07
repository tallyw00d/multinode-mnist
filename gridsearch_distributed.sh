#!/bin/bash
USERNAME=$1
PROJECT_NAME=$2

for WORKERS in 2 4; do
  for BATCH in 256 512; do
    JOB_NAME=batchsize-$BATCH-workers-$WORKERS
    just create job distributed \
        --project $USERNAME/$PROJECT_NAME \
        --name $JOB_NAME \
        --command "python -m mnist --batch_size $BATCH" \
        --setup-command "pip install -r requirements.txt" \
        --ps-docker-image tensorflow-1.11.0-cpu-py35 \
        --docker-image tensorflow-1.11.0-cpu-py35 \
        --time-limit 1h \
        --ps-type t2.small \
        --worker-type c5.xlarge \
        --ps-replicas 1 \
        --worker-replicas $WORKERS
    just start job $PROJECT_NAME/$JOB_NAME
    echo "Job $JOB_NAME started!"
  done
done
