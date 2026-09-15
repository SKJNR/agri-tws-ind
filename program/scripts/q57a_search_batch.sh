#!/bin/bash
# Q57-A adversarial verification search batch (Task 57, GLM round)
# Reads q57a_queries.tsv (id<TAB>query), runs z-ai web_search per query,
# saves raw JSON per query, logs pass/fail, retries failures once.
OUT=/home/z/my-project/scripts/q57a_results
QTSV=/home/z/my-project/scripts/q57a_queries.tsv
mkdir -p "$OUT"
LOG="$OUT/batch_run.log"
: > "$LOG"

run_one() {
  local id="$1" query="$2" attempt="$3"
  local args
  args=$(python3 -c "import json,sys; print(json.dumps({'query': sys.argv[1], 'num': 6}))" "$query")
  if z-ai function -n web_search -a "$args" -o "$OUT/${id}.json" >> "$LOG" 2>&1; then
    # verify non-empty array
    if python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if isinstance(d,list) and len(d)>0 else 1)" "$OUT/${id}.json" 2>>"$LOG"; then
      echo "OK   $id (attempt $attempt)"
      return 0
    fi
  fi
  return 1
}

declare -A FAILED
while IFS=$'\t' read -r id query; do
  [ -z "$id" ] && continue
  if ! run_one "$id" "$query" 1; then
    sleep 2
    if ! run_one "$id" "$query" 2; then
      echo "FAIL $id"
      FAILED[$id]="$query"
    fi
  fi
done < "$QTSV"

echo "----"
echo "Failed: ${#FAILED[@]}"
for k in "${!FAILED[@]}"; do echo "  $k: ${FAILED[$k]}"; done
