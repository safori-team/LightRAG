#!/usr/bin/env bash
# Stage the emotion graph artifact into the build context.
#
#   ./deploy/lambda_native/prepare-artifact.sh [source_dir]
#
# The artifact is deliberately NOT tracked by Git: it embeds verbatim speech
# transcripts, and this repository is a public fork. It reaches production only
# through the private ECR image. Keep the source on a controlled location — the
# machine that built it, or a private S3 bucket:
#
#   aws s3 sync s3://<private-bucket>/emotion-artifacts/<run>/ <source_dir>/
#
# The pipeline emits the storage files under "rag_storage/"; they are staged as
# "graph_storage/" because the repository .gitignore excludes every
# "rag_storage/" directory.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
run="${ARTIFACT_RUN:-split455_train273_case_w1_gemini_1536}"

default_source="$repo_root/../lightRAG/examples/emotion_graphrag/local_experiments/2026-08-09-455/artifacts/$run"
source_dir="${1:-${ARTIFACT_SOURCE:-$default_source}}"
target_dir="$repo_root/deploy/emotion_artifact/$run"

storage_files=(
  graph_chunk_entity_relation.graphml
  kv_store_text_chunks.json
  vdb_chunks.json
  vdb_entities.json
  vdb_relationships.json
)

if [[ ! -d "$source_dir" ]]; then
  echo "Artifact source not found: $source_dir" >&2
  echo "Pass the directory as the first argument or set ARTIFACT_SOURCE." >&2
  exit 1
fi

# The source may use either layout depending on where it came from.
src_storage="$source_dir/rag_storage"
[[ -d "$src_storage" ]] || src_storage="$source_dir/graph_storage"
if [[ ! -d "$src_storage" ]]; then
  echo "No rag_storage/ or graph_storage/ under $source_dir" >&2
  exit 1
fi

for name in "${storage_files[@]}"; do
  if [[ ! -f "$src_storage/$name" ]]; then
    echo "Incomplete artifact, missing: $src_storage/$name" >&2
    exit 1
  fi
done

# Stage beside the target and rename, so an interrupted copy cannot leave a
# half-populated directory that a later build would treat as complete.
staging="$target_dir.staging"
rm -rf "$staging"
mkdir -p "$staging/graph_storage"
cp "$source_dir/build_info.json" "$staging/build_info.json"
for name in "${storage_files[@]}"; do
  cp "$src_storage/$name" "$staging/graph_storage/$name"
done
rm -rf "$target_dir"
mv "$staging" "$target_dir"

echo "Staged artifact: $target_dir"
python3 -c "
import json, pathlib
info = json.loads(pathlib.Path('$target_dir/build_info.json').read_text())
backend = info.get('backend', {})
print(f\"  run={info.get('run')} chunks={info.get('chunks')} \"
      f\"embedding={backend.get('embedding_model')}@{backend.get('embedding_dim')}\")
"
du -sh "$target_dir"
