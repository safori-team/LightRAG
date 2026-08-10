#!/usr/bin/env bash
# Tag and push the locally tested image to ECR.
#
#   AWS_ACCOUNT_ID=123456789012 AWS_REGION=ap-northeast-2 \
#     ./deploy/lambda_native/push-ecr.sh
#
# Does not create or update the Lambda function; see
# docs/LambdaEmotionDeployment.md for that step.
set -euo pipefail

: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}"
: "${AWS_REGION:?Set AWS_REGION}"

repository="${ECR_REPOSITORY:-safori-emotion-native}"
source_image="${IMAGE_TAG:-safori-emotion-native:local-arm64}"
artifact="$(python3 -c "
import json,pathlib
p = pathlib.Path('deploy/emotion_artifact')
print(json.loads(next(p.glob('*/build_info.json')).read_text())['run'])
")"
# The tag records both the code commit and the graph artifact, so a rollback
# target is unambiguous when only one of the two changed.
release_tag="${RELEASE_TAG:-$(git rev-parse --short HEAD)-${artifact}-arm64}"
registry="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
target_image="$registry/$repository:$release_tag"

aws ecr describe-repositories --repository-names "$repository" \
  --region "$AWS_REGION" >/dev/null 2>&1 ||
  aws ecr create-repository \
    --repository-name "$repository" \
    --image-scanning-configuration scanOnPush=true \
    --image-tag-mutability IMMUTABLE \
    --region "$AWS_REGION" >/dev/null

aws ecr get-login-password --region "$AWS_REGION" |
  docker login --username AWS --password-stdin "$registry"

docker tag "$source_image" "$target_image"
docker push "$target_image"

echo
echo "Pushed: $target_image"
echo "Update the function with:"
echo "  aws lambda update-function-code --function-name safori-emotion-select \\"
echo "    --image-uri $target_image --region $AWS_REGION"
