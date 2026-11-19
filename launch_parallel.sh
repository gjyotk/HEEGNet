#!/usr/bin/env bash
# Run one held-out-subject fold per GPU, in parallel.
# Each GPU works through its own queue of subjects, so 10 subjects on 6 GPUs
# means GPUs 0-3 run two subjects one after the other.
#
# Usage:
#   bash launch_parallel.sh                         # defaults below
#   TAG=euclid SUBJECTS="1 2 3" bash launch_parallel.sh
#
# Every setting is forwarded as an environment variable to run_things_eeg2_cv.py.
trap 'kill $(jobs -p) 2>/dev/null; exit 1' SIGINT SIGTERM EXIT

TAG="${TAG:-hyp}"                       # use a different tag per condition (hyp, euclid, ...)
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"   # all 10 subjects
N_GPUS="${N_GPUS:-6}"
EPOCHS="${EPOCHS:-100}"
DTYPE="${DTYPE:-float64}"
SEED="${SEED:-42}"

mkdir -p logs
read -r -a SUBJ_ARR <<< "$SUBJECTS"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-$DIR/.venv/bin/python}"
if [ ! -f "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

for ((g = 0; g < N_GPUS; g++)); do
  (
    for ((i = g; i < ${#SUBJ_ARR[@]}; i += N_GPUS)); do
      s="${SUBJ_ARR[$i]}"
      echo "[GPU $g] starting subject $s (tag=$TAG, seed=$SEED)"
      CUDA_VISIBLE_DEVICES=$g \
      TEST_SUBJECTS=$s \
      RUN_TAG="${TAG}_sub${s}_seed${SEED}" \
      EPOCHS=$EPOCHS DTYPE=$DTYPE SEED=$SEED \
      "$PYTHON_BIN" run_things_eeg2_cv.py > "logs/${TAG}_sub${s}_seed${SEED}.log" 2>&1
      echo "[GPU $g] finished subject $s"
    done
  ) &
done

wait
echo "All jobs finished. Results: cv_results_things_eeg2_${TAG}_sub*_seed${SEED}.json"