# Emotion Selection Lambda — Build, Deploy, Operate

The service takes an **already produced** Gemini voice-analysis JSON and uses a
native LightRAG `aquery(mode="hybrid")` over a bundled emotion knowledge graph
to choose the final official minor-category emotions.

It does **not** re-run emotion detection, and it does **not** call the
experimental `judge.py`. The only outbound API calls per request are one query
embedding and one native-query generation.

```text
EC2 Spring server ──HTTPS──> Lambda Function URL ──> emotion_api.handler
                                                       │
                                                       ├─ Gemini embedding  (query vector)
                                                       └─ Gemini 2.5 Flash  (native query answer)
                                                          over the graph artifact in /tmp
```

## Contents

| Path | Role |
|---|---|
| `emotion_api/` | Production package: config, schemas, prompts, service, handler |
| `deploy/lambda_native/Dockerfile` | Lambda container image (arm64 by default) |
| `deploy/lambda_native/Dockerfile.dockerignore` | Build-context exclusions |
| `deploy/lambda_native/*.sh` | Build / local test / ECR push |
| `deploy/lambda_native/events/` | Request fixtures used by the local test |
| `deploy/emotion_artifact/split455_train273_case_w1_gemini_1536/` | Graph artifact baked into the image |
| `tests/emotion_api/` | Unit, contract, and query-config regression tests |

## API contract

Request:

```json
{
  "request_id": "231436",
  "gemini_result": {
    "transcript": "입력 발화",
    "summary": "선택 사항",
    "prosody": "선택 사항",
    "major": ["HAPPY"],
    "detected": [{"code": "JOY", "confidence": 0.75}]
  }
}
```

Response:

```json
{
  "request_id": "231436",
  "minor_categories": [
    {"code": "JOY", "confidence": 0.75},
    {"code": "SATISFACTION", "confidence": 0.63}
  ],
  "meta": {"engine": "lightrag_native_query", "mode": "hybrid", "model": "gemini-2.5-flash"}
}
```

`major_category` / `minor_categories` are accepted as request aliases for
`major` / `detected`.

Guarantees: only official taxonomy codes are returned, duplicates are merged at
the highest confidence, confidences are clamped to `[0, 1]`, and the count is
not forced to 3–5. `request_id` is echoed unchanged.

### Status codes

| Status | `error.code` | Cause |
|---|---|---|
| 400 | `MISSING_REQUEST_ID`, `INVALID_GEMINI_RESULT`, `MISSING_TRANSCRIPT`, `INVALID_MAJOR`, `INVALID_DETECTED`, `MALFORMED_JSON` | Input validation |
| 401 | `UNAUTHORIZED` | `EMOTION_API_TOKEN` set and header missing/wrong |
| 500 | `ARTIFACT_ERROR`, `CONFIG_ERROR`, `INTERNAL_ERROR` | Bad image, missing key, unexpected failure |
| 502 | `UPSTREAM_ERROR` | Gemini API or native query failure |

Error bodies contain only `code` and a generic message — never prompts,
retrieval context, file paths, or API keys.

When the query succeeds but the answer is unusable, the service returns **200**
with the caller's own detected emotions and a `meta.fallback` reason:

| `meta.fallback` | Meaning |
|---|---|
| `parse_failed` | The reply was not parseable JSON |
| `no_official_code` | The reply parsed but held no official taxonomy code |
| `no_context` | Retrieval ran but found no usable context (`[no-context]`) |

Treat any present `meta.fallback` as a degraded answer worth alerting on.

A provider outage is **not** a fallback. `LightRAG.aquery` swallows provider
errors and returns `None` rather than raising, so the service maps `None` and
empty replies to `502 UPSTREAM_ERROR`. Without that mapping a Gemini outage
would be indistinguishable from a model that answered in prose, and every
request during the outage would return `200` while quietly echoing the caller's
own input back as the answer.

## Local build and test

```bash
./deploy/lambda_native/build-local.sh                 # arm64 (default)
LAMBDA_PLATFORM=linux/amd64 ./deploy/lambda_native/build-local.sh

./deploy/lambda_native/test-local.sh                  # offline: no Gemini calls
GEMINI_API_KEY=... ./deploy/lambda_native/test-local.sh --with-gemini
```

`--with-gemini` is billable: two invocations, each costing one
`gemini-embedding-001` embedding plus one `gemini-2.5-flash` generation.

Unit tests (no network, no Docker — all marked `offline`):

```bash
./scripts/test.sh tests/emotion_api
```

The image is ~763 MB. `pandas` is excluded from `requirements.txt` because
LightRAG imports it lazily only in the Excel export branch, which this service
never reaches; the Dockerfile's build-time check imports the whole query path
so a wrongly pruned dependency fails the build rather than the first request.

### Architecture choice

arm64 (Graviton) is the default: every runtime dependency — numpy, pandas,
tiktoken, cryptography, pydantic-core — publishes `manylinux aarch64` wheels, so
nothing builds from source, it is native to Apple Silicon build hosts, and
Graviton Lambda is roughly 20% cheaper per GB-second. **The Lambda function's
architecture setting must match the image you built.**

## AWS deployment

### 1. Create the ECR repository and push

```bash
export AWS_ACCOUNT_ID=123456789012
export AWS_REGION=ap-northeast-2

aws ecr create-repository \
  --repository-name safori-emotion-native \
  --image-scanning-configuration scanOnPush=true \
  --image-tag-mutability IMMUTABLE \
  --region "$AWS_REGION"

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin \
    "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

./deploy/lambda_native/push-ecr.sh
```

`push-ecr.sh` tags as `<git-short-sha>-<artifact-run>-arm64`, so the image name
identifies both the code and the graph data it carries.

### 2. IAM execution role

```bash
aws iam create-role --role-name safori-emotion-lambda-role \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
    "Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam attach-role-policy --role-name safori-emotion-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

Add `secretsmanager:GetSecretValue` for the Gemini secret only if you read it
from Secrets Manager at runtime. No VPC attachment is needed — the function only
calls the public Gemini API, and putting it in a VPC would require a NAT gateway.

### 3. Create the function

```bash
aws lambda create-function \
  --function-name safori-emotion-select \
  --package-type Image \
  --code ImageUri="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/safori-emotion-native:<tag>" \
  --role "arn:aws:iam::$AWS_ACCOUNT_ID:role/safori-emotion-lambda-role" \
  --architectures arm64 \
  --memory-size 3008 \
  --timeout 120 \
  --ephemeral-storage Size=1024 \
  --environment "Variables={GEMINI_API_KEY=<secret>,LOG_LEVEL=INFO}" \
  --region "$AWS_REGION"
```

Recommended settings and why:

| Setting | Value | Reason |
|---|---|---|
| Memory | 3008 MB | The cold start loads a 26 MB relationship vector store plus numpy/pandas; memory also scales CPU, which shortens cold start |
| Timeout | 120 s | Cold start (artifact copy + storage init) plus one Gemini generation |
| Ephemeral storage | 1024 MB | The 35 MB artifact is copied to `/tmp`, and LightRAG writes status/cache files there; 512 MB works but leaves little headroom |
| Architecture | arm64 | Must match the built image |

### 4. Secrets

Never put `GEMINI_API_KEY` in the image, in Git, or in `.env.example`. Either:

```bash
# (a) Lambda environment variable, encrypted at rest with a KMS key
aws lambda update-function-configuration \
  --function-name safori-emotion-select \
  --environment "Variables={GEMINI_API_KEY=<secret>,LOG_LEVEL=INFO}" \
  --kms-key-arn arn:aws:kms:...:key/... --region "$AWS_REGION"

# (b) Secrets Manager, injected by the AWS Parameters and Secrets Lambda
#     extension so the key is never stored in the function configuration
aws secretsmanager create-secret --name safori/gemini-api-key \
  --secret-string '{"GEMINI_API_KEY":"..."}' --region "$AWS_REGION"
```

Option (a) is enough for a single-team MVP; move to (b) when the key needs
rotation or audit.

### 5. Exposure: how the Spring server calls it

**Recommended: Lambda Function URL with `AWS_IAM` auth.**

| Option | Verdict |
|---|---|
| Function URL + `AWS_IAM` | **Recommended.** No extra service, no per-request API Gateway cost, SigV4-signed by the EC2 instance role, 15-minute timeout ceiling |
| Function URL + `NONE` | Only with `EMOTION_API_TOKEN`; a public unauthenticated endpoint on a paid API is not acceptable for production |
| API Gateway (HTTP API) | Add later if you need usage plans, WAF, a custom domain, or request throttling — it adds cost and a 29-second timeout ceiling |
| Direct `lambda:InvokeFunction` | Fine too; requires the AWS SDK in Spring rather than a plain HTTP client |

```bash
aws lambda create-function-url-config \
  --function-name safori-emotion-select \
  --auth-type AWS_IAM --region "$AWS_REGION"
```

Grant the EC2 instance role `lambda:InvokeFunctionUrl` on the function, and sign
requests with SigV4 (`aws-crt` or the AWS SDK's HTTP signer in Spring). Because
the timeout is 120 s, set the Spring client's read timeout to at least 130 s, or
call it asynchronously.

If you must use `NONE` auth, set `EMOTION_API_TOKEN` on the function and send it
as the `X-Emotion-Api-Token` header, and restrict egress to the EC2 elastic IP.

### 6. Logs and monitoring

```bash
aws logs tail /aws/lambda/safori-emotion-select --follow --region "$AWS_REGION"
```

Watch for: `staged graph artifact to ...` (cold start), `LightRAG storages
initialized`, `native query failed`, and any response carrying `meta.fallback`.
Alarm on the `Errors`, `Throttles`, and `Duration` p99 metrics.

### 7. Releasing a new image and rolling back

```bash
./deploy/lambda_native/build-local.sh
./deploy/lambda_native/test-local.sh              # gate on the offline checks
./deploy/lambda_native/push-ecr.sh

aws lambda update-function-code \
  --function-name safori-emotion-select \
  --image-uri "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/safori-emotion-native:<new-tag>" \
  --publish --region "$AWS_REGION"
```

Use `--publish` to create a numbered version, and point a `live` alias at it so
rollback is a one-line alias move rather than a rebuild:

```bash
aws lambda update-alias --function-name safori-emotion-select \
  --name live --function-version <previous-version> --region "$AWS_REGION"
```

Because the ECR repository is immutable-tagged, the previous image is always
still available by tag.

### 8. Replacing the graph artifact

The artifact is baked into the image, so a new graph means a new image.

1. Build the new artifact with the emotion experiment pipeline.
2. Copy it to `deploy/emotion_artifact/<run-name>/`, keeping `build_info.json`
   and renaming the pipeline's `rag_storage/` directory to `graph_storage/`.
   The rename is required: the repository `.gitignore` excludes every
   `rag_storage/` directory, so the artifact could not otherwise be committed.
3. Update the `COPY` path in `deploy/lambda_native/Dockerfile`.
4. Update the expected `artifact_version` in `deploy/lambda_native/test-local.sh`.
5. Rebuild, run the local tests, push, and deploy as above.

The release tag from `push-ecr.sh` includes the artifact run name, so which
graph is live is visible from the image tag alone.

**The embedding model and dimension must match the artifact.** The bundled
artifact was built with `gemini-embedding-001` at 1536 dimensions; querying it
with a different embedding space produces silently wrong retrieval, not an
error.

## Reproducing the evaluated behaviour

The query configuration is pinned to the 91-record evaluation recorded in
`examples/emotion_graphrag/results/split455_native_query/summary.json`
(macro Jaccard 0.2610, precision 0.4057, 0 parse failures):

`mode="hybrid"`, `top_k=8`, `chunk_top_k=8`, `enable_rerank=False`,
`response_type="JSON object"`, `hl_keywords` = major categories,
`ll_keywords` = detected minor categories, and the taxonomy-restricted
`user_prompt`.

`tests/emotion_api/test_query_config.py` fails if any of these drift, including
a byte-for-byte comparison of the prompt against the experiment script. Passing
the keywords in explicitly is also what avoids a third Gemini call for keyword
extraction on every request.
