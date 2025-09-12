# Utilities for retrieving secrets from AWS Secrets Manager.
# The module checks environment variables first and falls back to
# AWS Secrets Manager at runtime.

from __future__ import annotations

import os
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError
except Exception:  # pragma: no cover - optional dependency
    boto3 = None
    ClientError = Exception


def _fetch_secret(secret_name: str, region_name: Optional[str] = None) -> Optional[str]:
    """Retrieve a secret value from AWS Secrets Manager."""
    if boto3 is None:
        return None
    region = region_name or os.getenv("AWS_REGION", "us-east-1")
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region)
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return response.get("SecretString")
    except ClientError:
        return None


def get_env_or_secret(env_var: str, secret_name_env: Optional[str] = None) -> Optional[str]:
    """Get a configuration value from an environment variable or Secrets Manager."""
    value = os.getenv(env_var)
    if value:
        return value

    secret_name = os.getenv(secret_name_env) if secret_name_env else env_var
    if not secret_name:
        return None
    return _fetch_secret(secret_name)
