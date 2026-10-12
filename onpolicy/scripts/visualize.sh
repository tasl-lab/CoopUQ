#!/bin/bash
# Run the released policy and draw it from above (onpolicy/scripts/render_2d.py).
#   bash onpolicy/scripts/visualize.sh iid [iid.mp4]                     the three layouts of a setting side by side
#   bash onpolicy/scripts/visualize.sh simulator/Builds/id_1/id_1.x86_64 [id_1.mp4]    a build made with simulator/build.sh
# Settings: iid, ood, ood_faster, ood_denser. Variables: DURATION (seconds, default 60), SPEED (playback
# speed, default 2), NUM_ROBOTS (default 3), SEED, BASE_PORT. The output may be .mp4 or .gif.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT:$PYTHONPATH"
PYTHON=${PYTHON:-python}
TARGET=${1:?usage: visualize.sh <setting or build.x86_64> [out.mp4]}
NUM_ROBOTS=${NUM_ROBOTS:-3}
SEED=${SEED:-1}
DURATION=${DURATION:-60}
SPEED=${SPEED:-2}
BASE_PORT=${BASE_PORT:-19806}

if [ -f "$TARGET" ]; then
    NAME=$(basename "$TARGET" .x86_64)
    ENV_ARGS="--eval_env_family custom --unity_env_path $(realpath "$TARGET") --n_eval_rollout_threads 1"
else
    NAME=$TARGET
    ENV_ARGS="--eval_env_family $TARGET --eval_scene 0 --unity_env_dir unity_envs --n_eval_rollout_threads 3"
fi
OUT=$(realpath -m "${2:-$NAME.mp4}")

mkdir -p "$ROOT/results"
cd "$ROOT/results"
$PYTHON "$ROOT/onpolicy/scripts/eval/eval_unity.py" \
    --env_name unity --scenario_name 3_scenes --algorithm_name rmappo \
    --experiment_name visualize_${NAME} --seed ${SEED} --use_eval \
    --num_agents ${NUM_ROBOTS} --n_training_threads 1 --n_rollout_threads 1 \
    --use_centralized_V --data_chunk_length 20 \
    ${ENV_ARGS} --base_port ${BASE_PORT} \
    --eval_episodes 1 --episode_length $(( DURATION * 5 + 3 )) --eval_hold_steps 2 \
    --ego_vec_size 6 --other_vec_size $(( (NUM_ROBOTS - 1) * 6 )) --human_num 7 \
    --resolution 0.1 --lidar_map_size 64 \
    --use_prediction --uncertainty_type entropy \
    --num_pred_samples 8 \
    --prediction_checkpoint_path "$ROOT/predictor/checkpoints/ogm_forecaster.pth" \
    --constrained --model_dir "$ROOT/onpolicy/checkpoints/policy" \
    --center_radius 5 --occupied_probability_threshold 0.5 --uq_threshold 0.3 \
    --discomfort_factor 1 --occ_cost_weight 0.5 --entropy_cost_weight 0.5 \
    --cost_horizon 3 --eval_traj_path "traj_${NAME}.npz" \
    > "log_visualize_${NAME}.log" 2>&1 \
    || { echo "evaluation failed, see results/log_visualize_${NAME}.log"; exit 1; }

$PYTHON "$ROOT/onpolicy/scripts/render_2d.py" "traj_${NAME}.npz" --out "$OUT" --speed ${SPEED}
