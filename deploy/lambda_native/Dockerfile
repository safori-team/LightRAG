# AWS Lambda container image for the native LightRAG emotion selection service.
#
# Build context is the repository root. Exclusions live in the sibling
# Dockerfile.dockerignore, so nothing has to be deleted from the fork to keep
# the context small.
#
# Target architecture: linux/arm64 (Lambda Graviton). Every runtime dependency
# ships a manylinux aarch64 wheel, so no source builds are needed and arm64 is
# both cheaper and native to Apple Silicon build hosts. For an x86_64 function,
# build with --platform linux/amd64 and set the Lambda architecture to match.

# ---- stage 1: resolve dependencies into a clean site-packages tree ----------
FROM public.ecr.aws/lambda/python:3.12 AS deps

COPY deploy/lambda_native/requirements.txt /tmp/requirements.txt
# Bytecode caches are kept on purpose: they are written once here instead of on
# every cold start. The base image is minimal (no coreutils), so any pruning
# would have to be done in Python.
RUN pip install --no-cache-dir --target /opt/python -r /tmp/requirements.txt

# ---- stage 2: runtime image -------------------------------------------------
FROM public.ecr.aws/lambda/python:3.12

COPY --from=deps /opt/python ${LAMBDA_TASK_ROOT}

# LightRAG core, imported directly from the fork rather than pip-installed so
# the deployed code is exactly what this branch contains.
COPY lightrag ${LAMBDA_TASK_ROOT}/lightrag
COPY emotion_api ${LAMBDA_TASK_ROOT}/emotion_api

# Read-only graph artifact. Copied to /tmp at cold start (see service.py).
COPY deploy/emotion_artifact/split455_train273_case_w1_gemini_1536 \
     ${LAMBDA_TASK_ROOT}/emotion_artifact

ENV LIGHTRAG_ARTIFACT_DIR=${LAMBDA_TASK_ROOT}/emotion_artifact \
    LIGHTRAG_WORK_DIR=/tmp/emotion-artifact \
    GEMINI_LLM_MODEL=gemini-2.5-flash \
    GEMINI_EMBEDDING_MODEL=gemini-embedding-001 \
    EMBEDDING_DIM=1536 \
    LIGHTRAG_MODE=hybrid \
    LIGHTRAG_TOP_K=8 \
    LIGHTRAG_CHUNK_TOP_K=8 \
    LIGHTRAG_KV_STORAGE=JsonKVStorage \
    LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage \
    LIGHTRAG_GRAPH_STORAGE=NetworkXStorage \
    LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage \
    SUMMARY_LANGUAGE=English \
    MAX_ASYNC=4 \
    LOG_LEVEL=INFO \
    PYTHONDONTWRITEBYTECODE=1

# Build-time health check. Fails the build if the handler cannot import, if the
# bundled artifact is incomplete, or if anything on the query path is missing a
# dependency. service.py imports LightRAG lazily, so those imports are forced
# here explicitly — otherwise a pruned dependency would only surface on the
# first production request. Makes no network call and needs no API key.
RUN python -c "\
import json; \
from lightrag import LightRAG, QueryParam; \
from lightrag.llm.gemini import gemini_model_complete, gemini_embed; \
from lightrag.utils import EmbeddingFunc; \
from emotion_api.handler import lambda_handler; \
from emotion_api import service, prompts; \
assert service.verify_artifact(); \
print(json.dumps(lambda_handler({'action': 'health'}, None)))"

CMD ["emotion_api.handler.lambda_handler"]
