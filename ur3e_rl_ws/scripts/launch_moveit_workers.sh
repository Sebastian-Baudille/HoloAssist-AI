#!/usr/bin/env bash
# Launch one MoveIt instance per parallel training worker.
# Usage:  ./scripts/launch_moveit_workers.sh [NUM_ENVS]
# Default NUM_ENVS=4  (matches UR3E_RL_NUM_ENVS default)
#
# Each worker needs its own MoveIt on a matching ROS_DOMAIN_ID:
#   worker 0 → ROS_DOMAIN_ID=30, worker 1 → 31, etc.
#
# Requires: ros2_ws sourced, gnome-terminal available.
# If gnome-terminal is missing the script prints the commands instead.

set -euo pipefail

NUM_ENVS="${1:-4}"
BASE_DOMAIN="${UR3E_RL_BASE_ROS_DOMAIN_ID:-30}"
ROS2_WS="$(cd "$(dirname "$0")/../../../ros2_ws" && pwd)"

MOVEIT_CMD_TEMPLATE="
source /opt/ros/humble/setup.bash && \
source ${ROS2_WS}/install/setup.bash && \
export ROS_DOMAIN_ID=__DOMAIN__ && \
ros2 launch ur_onrobot_moveit_config ur_onrobot_moveit.launch.py \
  ur_type:=ur3e onrobot_type:=rg2 \
  use_sim_time:=false launch_rviz:=false launch_servo:=false
"

echo "Launching ${NUM_ENVS} MoveIt instance(s), domains ${BASE_DOMAIN}...$((BASE_DOMAIN + NUM_ENVS - 1))"

if command -v gnome-terminal &>/dev/null; then
    ARGS=()
    for i in $(seq 0 $((NUM_ENVS - 1))); do
        DOMAIN=$((BASE_DOMAIN + i))
        CMD="${MOVEIT_CMD_TEMPLATE/__DOMAIN__/${DOMAIN}}"
        ARGS+=(--tab --title="MoveIt domain ${DOMAIN}" -- bash -c "${CMD}; exec bash")
    done
    gnome-terminal "${ARGS[@]}" &
    echo "Opened ${NUM_ENVS} terminal tab(s). Wait for all move_group nodes to print"
    echo "'All is well! Everyone is happy!' before starting training."
else
    echo ""
    echo "gnome-terminal not found. Run each of these in a separate terminal:"
    echo ""
    for i in $(seq 0 $((NUM_ENVS - 1))); do
        DOMAIN=$((BASE_DOMAIN + i))
        echo "--- Terminal $((i + 1)) (domain ${DOMAIN}) ---"
        echo "cd ${ROS2_WS}"
        echo "source /opt/ros/humble/setup.bash && source install/setup.bash"
        echo "export ROS_DOMAIN_ID=${DOMAIN}"
        echo "ros2 launch ur_onrobot_moveit_config ur_onrobot_moveit.launch.py \\"
        echo "  ur_type:=ur3e onrobot_type:=rg2 \\"
        echo "  use_sim_time:=false launch_rviz:=false launch_servo:=false"
        echo ""
    done
fi
