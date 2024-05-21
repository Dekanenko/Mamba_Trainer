import sagemaker
from sagemaker.estimator import Estimator

# Set the IAM role with necessary permissions
role = 'arn:aws:iam::851725251342:role/SageMakerExecutionRole'

# Define the Docker image URI
image_uri = 'public.ecr.aws/p5s4i0x3/mamba-trainer'

# Set the instance type and count
instance_type = 'ml.m4.xlarge'
instance_count = 1

# Define the output path in S3
output_path = 's3://mambabucket/model'

# Create the SageMaker Estimator
estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=instance_count,
    instance_type=instance_type,
    volume_size=50,  # Size in GB
    max_run=1000,    # Max run time in seconds
    output_path=output_path,
    base_job_name='mamba-training'
)

# Start the training job
estimator.fit()
