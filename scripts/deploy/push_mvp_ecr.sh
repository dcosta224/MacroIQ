#!/usr/bin/env bash
# Build the recipe-opt agent Docker image (CPU-only PyTorch) and push to ECR.
#
# Usage:
#   ./scripts/deploy/push_mvp_ecr.sh
#   ECR_REPO=macroiq AWS_REGION=us-east-1 ./scripts/deploy/push_mvp_ecr.sh
#   ./scripts/deploy/push_mvp_ecr.sh --no-cache
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO="${ECR_REPO:-macroiq}"
LOCAL_IMAGE="${LOCAL_IMAGE:-capstone-agent:local}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
NO_CACHE=false

for arg in "$@"; do
  case "$arg" in
    --no-cache) NO_CACHE=true ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *) die "Unknown argument: $arg" ;;
  esac
done

require_cmd docker
require_aws_cli

AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

msg "=== Build + push recipe-opt agent image to ECR ==="
msg "Account:  ${AWS_ACCOUNT_ID}"
msg "Region:   ${AWS_REGION}"
msg "Repo:     ${ECR_REPO}"
msg "Image:    ${LOCAL_IMAGE} -> ${IMAGE_URI}:deployment"
msg "App:      recipe_opt_web.server:app :8000"
msg "Platform: ${DOCKER_PLATFORM}"
msg "Local:    ${LOCAL_IMAGE}"
msg "Remote:   ${IMAGE_URI}:latest"
msg ""

if ! aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1; then
  msg "Creating ECR repository ${ECR_REPO}..."
  aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" >/dev/null
fi

BUILD_ARGS=(--platform "$DOCKER_PLATFORM" -t "$LOCAL_IMAGE" "$ROOT")
if [[ "$NO_CACHE" == true ]]; then
  BUILD_ARGS=(--no-cache "${BUILD_ARGS[@]}")
fi

msg "Building image..."
docker build "${BUILD_ARGS[@]}"

msg "Logging in to ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin \
  "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

msg "Tagging ${IMAGE_URI}:latest and ${IMAGE_URI}:deployment"
docker tag "$LOCAL_IMAGE" "${IMAGE_URI}:latest"
docker tag "$LOCAL_IMAGE" "${IMAGE_URI}:deployment"

msg "Pushing ${IMAGE_URI}:latest"
docker push "${IMAGE_URI}:latest"
msg "Pushing ${IMAGE_URI}:deployment"
docker push "${IMAGE_URI}:deployment"

msg ""
msg "Done: ${IMAGE_URI}:deployment (and :latest)"
