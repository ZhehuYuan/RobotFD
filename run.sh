#!/usr/bin/env bash
set -euo pipefail

# defaults
EXE="./fd"
DIM=3
NUM_PAIRS=32
EXT_PATTERN="*"
FO=""

usage() {
  echo "Usage: $0 <seed_int> <data_dir> [exe_path] [num_pairs] [ext_pattern]"
  echo "Example:"
  echo "  $0 12345 ../../DROID/1000"
  echo "  $0 12345 ../../DROID/1000 ./fd 32 '*.txt'"
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 1
fi

SEED="$1"
DATA_DIR="$2"
FO="${3:-$FO}"
NUM_PAIRS="${4:-$NUM_PAIRS}"
EXT_PATTERN="${5:-$EXT_PATTERN}"

if ! [[ "$SEED" =~ ^-?[0-9]+$ ]]; then
  echo "Error: seed must be an integer, got: $SEED" >&2
  exit 1
fi
if ! [[ "$NUM_PAIRS" =~ ^[0-9]+$ ]] || (( NUM_PAIRS <= 0 )); then
  echo "Error: num_pairs must be a positive integer, got: $NUM_PAIRS" >&2
  exit 1
fi

# ensure ./fd relative path works
cd "$(dirname "$0")"

[[ -x "$EXE" ]] || { echo "Error: not executable: $EXE" >&2; exit 1; }
[[ -d "$DATA_DIR" ]] || { echo "Error: missing dir: $DATA_DIR" >&2; exit 1; }

# collect files (null-delimited)
mapfile -d '' files < <(find "$DATA_DIR" -type f -name "$EXT_PATTERN" -print0)
N=${#files[@]}

if (( N < 2 )); then
  echo "Error: need at least 2 files in $DATA_DIR" >&2
  exit 1
fi

max_pairs=$(( N*(N-1)/2 ))   # unordered pairs without self-pairs
if (( NUM_PAIRS > max_pairs )); then
  echo "Error: requested $NUM_PAIRS unique unordered pairs, but only $max_pairs exist for N=$N files." >&2
  echo "Hint: reduce num_pairs or add more files." >&2
  exit 1
fi

echo "Seed      = $SEED"
echo "Data dir  = $DATA_DIR"
echo "Exe       = $EXE"
echo "Pairs     = $NUM_PAIRS (unique unordered; files may repeat across pairs)"
echo "Dim       = $DIM"
echo "Pattern   = $EXT_PATTERN"
echo "Files     = $N"
echo

# Generate pairs in python (reproducible, unique unordered)
# Output: f1\0f2\0f1\0f2\0...
mapfile -d '' chosen < <(
  printf '%s\0' "${files[@]}" | \
  SEED="$SEED" NUM_PAIRS="$NUM_PAIRS" python3 -c '
import os, random, sys

seed = int(os.environ["SEED"])
k = int(os.environ["NUM_PAIRS"])

data = sys.stdin.buffer.read().split(b"\0")
if data and data[-1] == b"": data.pop()

n = len(data)
if n < 2:
    raise SystemExit(f"Error: need at least 2 files, got {n}")

rnd = random.Random(seed)
seen = set()
out = []

max_attempts = k * 5000
attempts = 0

while len(out) < k:
    attempts += 1
    if attempts > max_attempts:
        raise SystemExit(f"Error: too many attempts (generated {len(out)}/{k}); n={n}")

    i = rnd.randrange(n)
    j = rnd.randrange(n)

    if i == j:
        continue  # disallow (A,A); delete this line if you want to allow self-pairs

    a, b = (i, j) if i < j else (j, i)  # unordered
    key = (a, b)
    if key in seen:
        continue
    seen.add(key)
    out.append((data[a], data[b]))

for f1, f2 in out:
    sys.stdout.buffer.write(f1 + b"\0" + f2 + b"\0")
'
)

need=$((NUM_PAIRS * 2))
if (( ${#chosen[@]} != need )); then
  echo "Error: expected $need chosen items but got ${#chosen[@]}." >&2
  exit 1
fi

for ((i=0; i<need; i+=2)); do
  f1="${chosen[i]}"
  f2="${chosen[i+1]}"
  echo "[$((i/2+1))/$NUM_PAIRS] $f1  |  $f2"
  "$EXE" "$f1" "$f2" "$DIM" "$FO"
done

