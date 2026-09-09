"""S3 sync helpers for Colab (boto3 — no AWS CLI required)."""

from __future__ import annotations

import os
from pathlib import Path


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, key_prefix) from s3://bucket/path/to/ or bucket/path."""
    uri = uri.strip()
    if uri.startswith("s3://"):
        uri = uri[5:]
    parts = uri.split("/", 1)
    bucket = parts[0]
    key_prefix = parts[1].strip("/") if len(parts) > 1 else ""
    return bucket, key_prefix


def _client():
    import boto3

    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def sync_s3_prefix(bucket: str, prefix: str, local_dir: Path, *, quiet: bool = False) -> int:
    """Download all objects under s3://bucket/prefix/ into local_dir. Returns file count."""
    prefix = prefix.strip("/") + "/"
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    s3 = _client()
    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix) :]
            if not rel:
                continue
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            n += 1
    if not quiet:
        print(f"Synced s3://{bucket}/{prefix} -> {local_dir} ({n} files)", flush=True)
    return n


def download_s3_file(bucket: str, key: str, local_path: Path, *, quiet: bool = False) -> None:
    """Download a single S3 object."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(bucket, key.lstrip("/"), str(local_path))
    if not quiet:
        print(f"Downloaded s3://{bucket}/{key} -> {local_path}", flush=True)


def upload_dir_to_s3(local_dir: Path, bucket: str, prefix: str, *, quiet: bool = False) -> int:
    """Upload all files under local_dir to s3://bucket/prefix/. Returns file count."""
    local_dir = Path(local_dir)
    prefix = prefix.strip("/") + "/"
    s3 = _client()
    n = 0
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        key = prefix + rel
        s3.upload_file(str(path), bucket, key)
        n += 1
    if not quiet:
        print(f"Uploaded {local_dir} -> s3://{bucket}/{prefix} ({n} files)", flush=True)
    return n


def upload_file_to_s3(local_path: Path, bucket: str, key: str) -> None:
    """Upload one file (used for progress.json streaming)."""
    _client().upload_file(str(local_path), bucket, key.lstrip("/"))
