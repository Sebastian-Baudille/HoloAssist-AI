#!/bin/bash
# Open RViz connected to training worker 0 (ROS_DOMAIN_ID=30).
# Run this while parallel training is active.

WORKER_ID=${1:-0}
DOMAIN_ID=$((30 + WORKER_ID))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVIZ_CONFIG="$SCRIPT_DIR/training_monitor.rviz"

source /opt/ros/humble/setup.bash
source "$(dirname "$SCRIPT_DIR")/install/setup.bash" 2>/dev/null || true

echo "Connecting to training worker $WORKER_ID (ROS_DOMAIN_ID=$DOMAIN_ID)"
echo "Topics available:"
ROS_DOMAIN_ID=$DOMAIN_ID IGN_PARTITION=$DOMAIN_ID ros2 topic list 2>/dev/null | \
  grep -E "joint_states|tcp_pose|cube_|goal" | sed 's/^/  /'

echo ""
echo "Opening RViz... (mesh errors logged to /tmp/rviz_monitor.log)"
# LIBGL_ALWAYS_SOFTWARE=1 uses Mesa CPU renderer — avoids GPU/OpenGL crashes
# on laptops with hybrid NVIDIA+Intel graphics when interacting with the 3D view.
ROS_DOMAIN_ID=$DOMAIN_ID IGN_PARTITION=$DOMAIN_ID \
  LIBGL_ALWAYS_SOFTWARE=1 \
  rviz2 -d "$RVIZ_CONFIG" 2>/tmp/rviz_monitor.log
