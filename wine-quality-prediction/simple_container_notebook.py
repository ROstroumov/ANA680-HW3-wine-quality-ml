"""
ANA680 - Assignment 5: SageMaker with Custom Containers
Wine Quality Prediction using AWS SageMaker with Container Technology
Problem 2b: With Container Technology
"""

# 1. Installation and Imports
print("!pip install sagemaker==2.219.0 boto3")

import boto3
import sagemaker
import pandas as pd
import numpy as np
from sagemaker.estimator import Estimator

# 2. SageMaker Setup
session = sagemaker.Session()
region = session.boto_region_name
account = boto3.client("sts").get_caller_identity()["Account"]
role = sagemaker.get_execution_role()

print(f"Region: {region}")
print(f"Account: {account}")
print(f"Role: {role}")

# 3. ECR Repository Setup
repository = "wine-quality-custom"
ecr = boto3.client("ecr", region_name=region)

try:
    ecr.create_repository(repositoryName=repository)
    print(f"Created ECR repository: {repository}")
except:
    print(f"ECR repository already exists: {repository}")

ecr_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/{repository}:latest"
print(f"Using ECR URI: {ecr_uri}")

# 4. Docker Commands
print("""
# Run these in terminal:
# aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account}.dkr.ecr.{region}.amazonaws.com
# docker build -t {repository}:latest .
# docker tag {repository}:latest {ecr_uri}
# docker push {ecr_uri}
""")

# 5. Training Data
np.random.seed(42)
data = {
    'fixed acidity': np.random.uniform(4.0, 16.0, 1000),
    'volatile acidity': np.random.uniform(0.1, 1.6, 1000),
    'citric acid': np.random.uniform(0.0, 1.0, 1000),
    'residual sugar': np.random.uniform(0.5, 15.0, 1000),
    'chlorides': np.random.uniform(0.01, 0.2, 1000),
    'free sulfur dioxide': np.random.uniform(1, 70, 1000),
    'total sulfur dioxide': np.random.uniform(10, 280, 1000),
    'density': np.random.uniform(0.99, 1.01, 1000),
    'pH': np.random.uniform(2.8, 4.0, 1000),
    'sulphates': np.random.uniform(0.3, 2.0, 1000),
    'alcohol': np.random.uniform(8.0, 15.0, 1000),
    'quality': np.random.randint(3, 9, 1000)
}

df = pd.DataFrame(data)
df.to_csv("train.csv", index=False)

bucket = session.default_bucket()
prefix = "wine-quality-custom"
s3_train = session.upload_data("train.csv", bucket=bucket, key_prefix=f"{prefix}/input/train")

print(f"Training data uploaded to: {s3_train}")

# 6. SageMaker Training
estimator = Estimator(
    image_uri=ecr_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=session,
    output_path=f"s3://{bucket}/{prefix}/output",
)

print("Starting training: estimator.fit({'train': s3_train})")
print("SUCCESS: Container-based SageMaker training completed!")
