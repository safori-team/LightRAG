#!/usr/bin/env bash
# Build the Lambda container image from the repository root context.
#
#   LAMBDA_PLATFORM=linux/amd64 ./deploy/lambda_native/build-local.sh
#
# Defaults to linux/arm64 (Lambda Graviton). The Lambda function architecture
# must match whatever is built here.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
platform="${LAMBDA_PLATFORM:-linux/arm64}"
image_tag="${IMAGE_TAG:-safori-emotion-native:local-${platform##*/}}"

# The graph artifact is not tracked by Git (it embeds speech transcripts and
# this fork is public). Stage it into the build context first.
run="${ARTIFACT_RUN:-split455_train273_case_w1_gemini_1536}"
if [[ ! -d "$repo_root/deploy/emotion_artifact/$run" ]]; then
  "$repo_root/deploy/lambda_native/prepare-artifact.sh"
fi

docker buildx build \
  --platform "$platform" \
  --provenance=false \
  --load \
  --file "$repo_root/deploy/lambda_native/Dockerfile" \
  --tag "$image_tag" \
  "$repo_root"

echo
echo "Built: $image_tag ($platform)"
docker image inspect "$image_tag" \
  --format 'id={{.Id}} size={{.Size}} arch={{.Architecture}}'
