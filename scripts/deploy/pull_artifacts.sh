#!/usr/bin/env bash
# Partner command: pull experiment artifacts from S3 to local scratch/.
# Does NOT start EC2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

INCLUDE_CSV=false
for arg in "$@"; do
  case "$arg" in
    --include-csv) INCLUDE_CSV=true ;;
    -h|--help)
      echo "Usage: $0 [--include-csv]"
      exit 0
      ;;
  esac
done

if [[ ! -f "$AWS_ENV_FILE" ]]; then
  msg "Copy deploy/aws.env.example to deploy/aws.env and fill in bucket names."
  msg "Daniel can share deploy/aws.env values (not the file itself if it has secrets — buckets are fine)."
  die "Missing deploy/aws.env"
fi

load_aws_env
require_aws_cli

mkdir -p "${ROOT}/scratch"

msg "Pulling s3://${S3_BUCKET_ARTIFACTS}/scratch/ -> ${ROOT}/scratch/"
aws s3 sync "s3://${S3_BUCKET_ARTIFACTS}/scratch/" "${ROOT}/scratch/" --region "$AWS_REGION"

if [[ "$INCLUDE_CSV" == true ]]; then
  aws s3 sync "s3://${S3_BUCKET_ARTIFACTS}/scratch/EDA/" "${ROOT}/scratch/EDA/" \
    --region "$AWS_REGION" --exclude "*" --include "*.csv"
fi

if aws s3 ls "s3://${S3_BUCKET_ARTIFACTS}/mvp_web/cache/" --region "$AWS_REGION" 2>/dev/null | grep -q .; then
  mkdir -p "${ROOT}/mvp_web/cache"
  aws s3 sync "s3://${S3_BUCKET_ARTIFACTS}/mvp_web/cache/" \
    "${ROOT}/mvp_web/cache/" --region "$AWS_REGION"
fi

msg "Done. Artifacts in ${ROOT}/scratch/"
