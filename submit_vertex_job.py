import argparse
from google.cloud import aiplatform
from google.protobuf.struct_pb2 import Value
import json

# Initialize argument parser
parser = argparse.ArgumentParser(description="Run a training job on Vertex AI with wandb key.")
parser.add_argument("--wandb_key", type=str, required=True, help="Your wandb API key")
args = parser.parse_args()

# Initialize the Vertex AI client
project = "elite-totem-410916"
location = "europe-west1"  # change as needed
aiplatform.init(project=project, location=location, staging_bucket=None)

# Define your worker pool specifications
worker_pool_specs = [
    {
        "machine_spec": {
            "machine_type": "n1-standard-4",
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "gcr.io/elite-totem-410916/trainer:latest",
            "env": [
                {
                    "name": "WANDB_API_KEY",
                    "value": args.wandb_key,  # Use the key passed from the command line
                }
            ],
        },
    }
]

# Create a CustomJob
custom_job = aiplatform.CustomJob(
    display_name="Training_Job",
    staging_bucket="gs://detect_ai_mlops",
    worker_pool_specs=worker_pool_specs,
)

# Run the CustomJob
custom_job.submit()  # Set sync=False if you don't want to wait for the job to finish
