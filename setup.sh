#!/usr/bin/env bash
# HoloAssist-AI — one-shot dependency installer and workspace builder.
#
# Prerequisites (must be installed before running this script):
#   • Ubuntu 22.04 (Jammy)
#   • ROS 2 Humble  →  https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
#   • Python 3.12   →  sudo apt install python3.12 python3.12-venv
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh            # full setup (ROS sim + Python clustering venv)
#   ./setup.sh --ros-only # only build the ROS workspace
#   ./setup.sh --py-only  # only create the Python 3.12 clustering venv

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_WS="$REPO_ROOT/ros2_ws"
VENV_DIR="$REPO_ROOT/clustering/.venv"
ROS_DISTRO="${ROS_DISTRO:-humble}"

# ── helpers ──────────────────────────────────────────────────────────────────
info()  { echo -e "\033[1;34m[setup]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[setup]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[setup]\033[0m $*"; }
die()   { echo -e "\033[1;31m[setup]\033[0m $*" >&2; exit 1; }

ROS_ONLY=false
PY_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --ros-only) ROS_ONLY=true ;;
    --py-only)  PY_ONLY=true  ;;
    *) die "Unknown argument: $arg" ;;
  esac
done

# ── check ROS 2 ───────────────────────────────────────────────────────────────
if ! $PY_ONLY; then
  ROS_SETUP="/opt/ros/$ROS_DISTRO/setup.bash"
  if [[ ! -f "$ROS_SETUP" ]]; then
    die "ROS 2 $ROS_DISTRO not found at $ROS_SETUP.\n" \
        "Install it first: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html"
  fi
  # shellcheck source=/dev/null
  source "$ROS_SETUP"
  info "ROS 2 $ROS_DISTRO sourced."

  # ── rosdep ─────────────────────────────────────────────────────────────────
  if ! command -v rosdep &>/dev/null; then
    info "Installing rosdep..."
    sudo apt-get install -y python3-rosdep
  fi
  if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    sudo rosdep init
  fi
  rosdep update --include-eol-distros -q

  info "Installing ROS package dependencies via rosdep..."
  rosdep install \
    --from-paths "$ROS_WS/src" \
    --ignore-src \
    --rosdistro "$ROS_DISTRO" \
    -r -y

  # ── colcon build ───────────────────────────────────────────────────────────
  info "Building ROS 2 workspace (symlink-install)..."
  cd "$ROS_WS"
  colcon build \
    --packages-select holoassist_sim \
    --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release

  ok "ROS workspace built. Source it with:"
  ok "  source $ROS_WS/install/setup.bash"
fi

# ── Python 3.12 clustering venv ───────────────────────────────────────────────
if ! $ROS_ONLY; then
  PY312=""
  for candidate in python3.12 python3; do
    if command -v "$candidate" &>/dev/null; then
      ver="$("$candidate" -c 'import sys; print(sys.version_info[:2])')"
      if [[ "$ver" == "(3, 12)" ]]; then
        PY312="$candidate"
        break
      fi
    fi
  done

  if [[ -z "$PY312" ]]; then
    warn "Python 3.12 not found — skipping clustering venv."
    warn "Install it with:  sudo apt install python3.12 python3.12-venv"
  else
    info "Creating clustering venv at $VENV_DIR using $PY312..."
    "$PY312" -m venv "$VENV_DIR"
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip -q
    pip install -r "$REPO_ROOT/clustering/requirements.txt"
    deactivate
    ok "Python 3.12 venv ready. Activate with:"
    ok "  source $VENV_DIR/bin/activate"
  fi
fi

ok "Setup complete."
