#!/usr/bin/env bash
# Local Lambda Runtime Interface Emulator test.
#
#   ./deploy/lambda_native/test-local.sh              # offline checks only
#   ./deploy/lambda_native/test-local.sh --with-gemini # adds one billable call
#
# The offline pass covers health, artifact loading, and every input-validation
# failure — none of it calls Gemini. --with-gemini sends ONE real request that
# costs one gemini-embedding-001 embedding plus one gemini-2.5-flash generation.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image_tag="${IMAGE_TAG:-safori-emotion-native:local-arm64}"
platform="${LAMBDA_PLATFORM:-linux/arm64}"
container="safori-emotion-native-local"
port="${LAMBDA_LOCAL_PORT:-9010}"
url="http://127.0.0.1:$port/2015-03-31/functions/function/invocations"
with_gemini="${1:-}"

cleanup() { docker rm -f "$container" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

run_args=(--detach --name "$container" --platform "$platform"
          --publish "$port:8080" --env LOG_LEVEL=INFO)
if [[ "$with_gemini" == "--with-gemini" ]]; then
  : "${GEMINI_API_KEY:?Set GEMINI_API_KEY for --with-gemini}"
  run_args+=(--env "GEMINI_API_KEY=$GEMINI_API_KEY")
fi
docker run "${run_args[@]}" "$image_tag" >/dev/null

invoke() { curl --fail --silent --max-time 180 -XPOST "$url" \
             -H 'Content-Type: application/json' --data-binary "@$1"; }

for _ in {1..40}; do
  curl --fail --silent -XPOST "$url" -H 'Content-Type: application/json' \
    --data '{"action":"health"}' >/dev/null 2>&1 && break
  sleep 1
done

fail=0
check() { # name expected_jq_expr event_file
  local name="$1" expr="$2" file="$3" body
  body="$(invoke "$here/events/$file")"
  if python3 -c "
import json,sys
payload = json.loads(sys.argv[1])
assert $expr, payload
" "$body" 2>/dev/null; then
    echo "  PASS  $name"
  else
    echo "  FAIL  $name -> $body"; fail=1
  fi
}

echo "== offline checks (no Gemini call) =="
check "health / artifact loads" \
  "payload['ok'] and payload['artifact_version']=='split455_train273_case_w1_gemini_1536'" \
  health.json
check "missing transcript -> 400" \
  "payload['status']==400 and payload['error']['code']=='MISSING_TRANSCRIPT'" \
  invalid_missing_transcript.json
check "empty major/detected -> 400" \
  "payload['status']==400 and payload['error']['code']=='INVALID_MAJOR'" \
  invalid_empty_major.json
check "unknown emotion code -> 400" \
  "payload['status']==400 and payload['error']['code']=='INVALID_DETECTED'" \
  invalid_bad_code.json
check "missing request_id -> 400" \
  "payload['status']==400 and payload['error']['code']=='MISSING_REQUEST_ID'" \
  invalid_missing_request_id.json

# Only meaningful without a key, and it would otherwise consume the cold start
# that the --with-gemini timing below is trying to measure.
if [[ "$with_gemini" != "--with-gemini" ]]; then
  check "no API key -> 5xx, no leak" \
    "payload['status']>=500 and 'AIza' not in json.dumps(payload)" \
    select_happy.json
fi

# Failure modes that need their own container because they depend on the
# environment rather than on the request payload.
echo "== environment failure modes (separate containers, no Gemini call) =="

probe() { # name env_flag expected_expr
  local name="$1" env_flag="$2" expr="$3" probe_port=$((port + 1)) body
  docker rm -f "$container-probe" >/dev/null 2>&1 || true
  # shellcheck disable=SC2086
  docker run --detach --name "$container-probe" --platform "$platform" \
    --publish "$probe_port:8080" $env_flag "$image_tag" >/dev/null
  local probe_url="http://127.0.0.1:$probe_port/2015-03-31/functions/function/invocations"
  for _ in {1..40}; do
    curl --fail --silent -XPOST "$probe_url" -H 'Content-Type: application/json' \
      --data '{"action":"health"}' >/dev/null 2>&1 && break
    sleep 1
  done
  body="$(curl --silent --max-time 120 -XPOST "$probe_url" \
    -H 'Content-Type: application/json' \
    --data-binary "@$here/events/select_happy.json")"
  docker rm -f "$container-probe" >/dev/null 2>&1 || true
  if python3 -c "
import json,sys
payload = json.loads(sys.argv[1])
assert $expr, payload
" "$body" 2>/dev/null; then
    echo "  PASS  $name"
  else
    echo "  FAIL  $name -> $body"; fail=1
  fi
}

probe "bad artifact path -> ARTIFACT_ERROR" \
  "--env LIGHTRAG_ARTIFACT_DIR=/var/task/does-not-exist" \
  "payload['status']==500 and payload['error']['code']=='ARTIFACT_ERROR'"

probe "invalid Gemini key -> UPSTREAM_ERROR" \
  "--env GEMINI_API_KEY=invalid-key-for-testing" \
  "payload['status']==502 and payload['error']['code']=='UPSTREAM_ERROR' and 'invalid-key-for-testing' not in json.dumps(payload)"

if [[ "$with_gemini" == "--with-gemini" ]]; then
  echo "== live native query (1 embedding + 1 generation, billable) =="
  start=$(python3 -c 'import time;print(time.time())')
  body="$(invoke "$here/events/select_happy.json")"
  elapsed=$(python3 -c "import time;print(f'{time.time()-$start:.2f}')")
  echo "$body" | python3 -m json.tool
  echo "cold-start invocation: ${elapsed}s"
  python3 -c "
import json,sys
p = json.loads(sys.argv[1])
assert p['request_id'] == '231436', p
assert p['meta']['engine'] == 'lightrag_native_query', p
assert p['meta']['mode'] == 'hybrid', p
assert p['minor_categories'], p
" "$body" && echo "  PASS  live selection" || fail=1

  start=$(python3 -c 'import time;print(time.time())')
  invoke "$here/events/select_happy.json" >/dev/null
  echo "warm invocation: $(python3 -c "import time;print(f'{time.time()-$start:.2f}')")s"
fi

docker logs "$container" 2>&1 | tail -5
exit "$fail"
