import os
import boto3
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)

def get_s3_client():
    """
    Creates and returns a boto3 S3 client using credentials from environment variables.
    """
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region_name = os.getenv("AWS_REGION")

    if not all([aws_access_key_id, aws_secret_access_key, region_name]):
        logger.error("Missing AWS S3 configuration in environment variables")
        raise ValueError("Missing AWS S3 configuration in environment variables")

    return boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

def upload_file_to_s3(file_path: str, s3_key: str, bucket_name: str = None) -> bool:
    """
    Uploads a file to an S3 bucket.
    
    :param file_path: Local path to the file
    :param s3_key: S3 key (path/name) under which to store the file
    :param bucket_name: S3 bucket name; if not provided, uses S3_BUCKET_NAME env variable.
    :return: True if upload succeeded, False otherwise.
    """
    bucket = bucket_name or os.getenv("S3_BUCKET_NAME") or os.getenv("AWS_S3_BUCKET")
    if not bucket:
        logger.error("S3 bucket name not provided in environment variables")
        raise ValueError("S3 bucket name not provided in environment variables")
    
    s3 = get_s3_client()
    try:
        s3.upload_file(file_path, bucket, s3_key)
        logger.info(f"Uploaded {file_path} to s3://{bucket}/{s3_key}")
        return True
    except ClientError as e:
        logger.error(f"Error uploading file to S3: {e}")
        return False

def download_file_from_s3(s3_key: str, dest_path: str, bucket_name: str = None) -> bool:
    """
    Downloads a file from S3.
    
    :param s3_key: S3 key (path/name) of the file
    :param dest_path: Local destination path
    :param bucket_name: S3 bucket name; if not provided, uses S3_BUCKET_NAME env variable.
    :return: True if download succeeded, False otherwise.
    """
    bucket = bucket_name or os.getenv("S3_BUCKET_NAME") or os.getenv("AWS_S3_BUCKET")
    if not bucket:
        logger.error("S3 bucket name not provided in environment variables")
        raise ValueError("S3 bucket name not provided in environment variables")
    
    s3 = get_s3_client()
    try:
        s3.download_file(bucket, s3_key, dest_path)
        logger.info(f"Downloaded s3://{bucket}/{s3_key} to {dest_path}")
        return True
    except ClientError as e:
        logger.error(f"Error downloading file from S3: {e}")
        return False

def list_s3_objects(bucket_name: str = None, prefix: str = "") -> list:
    """
    Lists objects in an S3 bucket under an optional prefix.
    
    :param bucket_name: S3 bucket name; if not provided, uses S3_BUCKET_NAME env variable.
    :param prefix: Filter objects by prefix.
    :return: List of objects (each represented as a dictionary).
    """
    bucket = bucket_name or os.getenv("S3_BUCKET_NAME") or os.getenv("AWS_S3_BUCKET")
    if not bucket:
        logger.error("S3 bucket name not provided in environment variables")
        raise ValueError("S3 bucket name not provided in environment variables")
    
    s3 = get_s3_client()
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        objects = response.get("Contents", [])
        logger.info(f"Found {len(objects)} object(s) in s3://{bucket}/{prefix}")
        return objects
    except ClientError as e:
        logger.error(f"Error listing S3 objects: {e}")
        return [] 