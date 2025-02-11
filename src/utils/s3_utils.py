import os
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

load_dotenv()

# Configure these as appropriate for your environment
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_REGION = os.getenv("AWS_REGION")

s3_client = boto3.client('s3')

def upload_file_to_s3(file_path: str, s3_key: str = "") -> str:
    """
    Uploads a file to S3 and returns the file URL.
    """
    if not s3_key:
        s3_key = os.path.basename(file_path)
    try:
        s3_client.upload_file(file_path, S3_BUCKET, s3_key)
        url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        return url
    except FileNotFoundError:
        raise Exception(f"File {file_path} not found.")
    except NoCredentialsError:
        raise Exception("S3 credentials not available.") 