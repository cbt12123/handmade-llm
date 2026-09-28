#!/usr/bin/env bash
# =============================================================================
# 一键冒烟测试：按顺序执行 script/ 下所有章节脚本，报告 PASS / FAIL / SKIP
#
#   bash script/run_all.sh            # 跑全部章节
#   bash script/run_all.sh 01         # 只跑第 1 章
#   bash script/run_all.sh 07         # 只跑第 7 章（无 GPU 时会自动 SKIP）
# =============================================================================
set -u
cd "$(dirname "$0")/.." || exit 1

PY=${PYTHON:-python3}
if command -v timeout >/dev/null 2>&1; then RUNNER="timeout 600"; else RUNNER=""; fi

PASS=0; FAIL=0; SKIP=0
FAILED_LIST=()

FILTER="${1:-}"
for f in $(find script -name "*.py" -not -name "common_utils.py" -not -name "_*.py" | sort); do
  if [ -n "$FILTER" ] && [[ "$f" != *"/${FILTER}_"* ]]; then continue; fi
  printf "▶ %-58s" "$f"
  out=$($RUNNER "$PY" "$f" 2>&1)
  code=$?
  if [ $code -ne 0 ]; then
    echo "FAIL(exit=$code)"
    FAIL=$((FAIL+1)); FAILED_LIST+=("$f")
    echo "$out" | tail -n 12 | sed 's/^/     /'
  elif echo "$out" | grep -q "\[SKIP\]"; then
    echo "SKIP"
    SKIP=$((SKIP+1))
  else
    echo "PASS"
    PASS=$((PASS+1))
  fi
done

echo
echo "================ 汇总 ================"
echo "PASS: $PASS   SKIP: $SKIP   FAIL: $FAIL"
if [ ${#FAILED_LIST[@]} -gt 0 ]; then
  echo "失败脚本："
  for f in "${FAILED_LIST[@]}"; do echo "  - $f"; done
  exit 1
fi
