#!/bin/bash
# Evaluate the released policy.
#   bash onpolicy/scripts/eval.sh [iid ood ood_faster ood_denser]     the four evaluation settings
#   ENV=simulator/Builds/id_1/id_1.x86_64 NUM_ROBOTS=5 bash onpolicy/scripts/eval.sh
#                                                                    a build made with simulator/build.sh
# Other variables: SEED, EPISODES, BASE_PORT, THREADS, EXTRA (further arguments for eval_unity.py).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT:$PYTHONPATH"
PYTHON=${PYTHON:-python}
SEED=${SEED:-1}
EPISODES=${EPISODES:-24}
BASE_PORT=${BASE_PORT:-19006}
THREADS=${THREADS:-3}
NUM_ROBOTS=${NUM_ROBOTS:-3}
if [ -n "$ENV" ]; then
    ENV=$(realpath "$ENV")
    FAMILIES=$(basename "$ENV" .x86_64)
else
    FAMILIES=${@:-"iid ood ood_faster ood_denser"}
fi

mkdir -p "$ROOT/results"
cd "$ROOT/results"

for fam in $FAMILIES; do
    exp="${fam}"; [ -z "$ENV" ] && [ "$NUM_ROBOTS" != 3 ] && exp="${fam}_n${NUM_ROBOTS}"
    if [ -n "$ENV" ]; then ENV_ARGS="--eval_env_family custom --unity_env_path $ENV"
    else ENV_ARGS="--eval_env_family ${fam} --eval_scene 0 --unity_env_dir unity_envs"; fi
    rm -f "evaluating_${exp}_seed_${SEED}.txt"
    echo "=== evaluating ${fam} (seed ${SEED}) ==="
    $PYTHON "$ROOT/onpolicy/scripts/eval/eval_unity.py" \
        --env_name unity --scenario_name 3_scenes --algorithm_name rmappo \
        --experiment_name ${exp} --seed ${SEED} --use_eval \
        --num_agents ${NUM_ROBOTS} --n_training_threads 1 --n_rollout_threads 1 \
        --use_centralized_V --data_chunk_length 20 \
        ${ENV_ARGS} \
        --base_port ${BASE_PORT} --n_eval_rollout_threads ${THREADS} \
        --eval_episodes ${EPISODES} --episode_length 300 --eval_hold_steps 2 \
        --ego_vec_size 6 --other_vec_size $(( (NUM_ROBOTS - 1) * 6 )) --human_num 7 \
        --resolution 0.1 --lidar_map_size 64 \
        --use_prediction --uncertainty_type entropy \
        --num_pred_samples 8 \
        --prediction_checkpoint_path "$ROOT/predictor/checkpoints/ogm_forecaster.pth" \
        --constrained --model_dir "$ROOT/onpolicy/checkpoints/policy" \
        --center_radius 5 --occupied_probability_threshold 0.5 --uq_threshold 0.3 \
        --discomfort_factor 1 --occ_cost_weight 0.5 --entropy_cost_weight 0.5 \
        --cost_horizon 3 ${EXTRA} \
        > "log_${exp}_seed_${SEED}.log" 2>&1 \
        || { echo "evaluation failed, see results/log_${exp}_seed_${SEED}.log"; exit 1; }
    BASE_PORT=$((BASE_PORT + 100))
done

EXPS=$FAMILIES; [ -z "$ENV" ] && [ "$NUM_ROBOTS" != 3 ] && EXPS=$(for f in $FAMILIES; do echo "${f}_n${NUM_ROBOTS}"; done)
$PYTHON "$ROOT/onpolicy/scripts/eval/summarize_eval.py" --seed ${SEED} $EXPS
