# Lambda image — native LightRAG emotion selection

Builds `emotion_api/` + LightRAG core + the emotion graph artifact into an AWS
Lambda container image.

```bash
./build-local.sh          # or: LAMBDA_PLATFORM=linux/amd64 ./build-local.sh
./test-local.sh           # offline checks, no Gemini calls
GEMINI_API_KEY=... ./test-local.sh --with-gemini   # billable: 2 embeddings + 2 generations
AWS_ACCOUNT_ID=... AWS_REGION=... ./push-ecr.sh
```

Run the scripts from the repository root — the build context is the repo root
and exclusions come from `Dockerfile.dockerignore`.

Full deployment, IAM, exposure, and rollback instructions:
[docs/LambdaEmotionDeployment.md](../../docs/LambdaEmotionDeployment.md).

This directory is the native-query production path. The older
`deploy/lambda/` scaffold (custom vector/graph retrieval plus a separate
`judge.py`) is not used by this service.
