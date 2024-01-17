#!/bin/bash
existing_tags=$(gcloud container images list --repository=gcr.io/elite-totem-410916 --filter="name:base_image" --format=json)
if [[ "$existing_tags" == "[]" ]]; then
    gcloud builds submit --config cloud_yaml/cloudbuild_base_image.yaml
else
    echo "Base image already exists. Now checking for potential changes to requirements."
fi

# Check if requirements.txt has changed in current commit
changes=$(git diff HEAD^ HEAD -- requirements.txt)
#changes=1

if [ "$changes" != "" ]; then
    num_logs=1    
else
    num_logs=2
fi

echo $num_logs

commit_hash=$(git log -n $num_logs --format="%H" -- requirements.txt)
changes=$(git diff $commit_hash -- requirements.txt)
if "$changes" != "" ]]; then
    gcloud builds submit --config cloud_yaml/cloudbuild_base_image.yaml
else
    echo "No changes in requirements.txt and Docker image already exists. Skipping build."
fi