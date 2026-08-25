#!/bin/bash
# Select the 30-point, sub-5e-4, three-hour sphere 1->5 campaign.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCAL_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
export SPHERE6_CAMPAIGN_CONFIG="${LOCAL_ROOT}/plumbing/config/sphere_six_point_1to5_cannon_blind30_3h_v1.json"
export SPHERE6_CAMPAIGN_CHECK=sphere_six_point_cannon_blind30_3h_checks.py
export SPHERE6_LOCAL_DESIGN="${LOCAL_ROOT}/plumbing/results/sphere_six_point_1to5/cannon_blind30_3h_v1"
export SPHERE6_REMOTE_RUN_ROOT=/n/holylabs/yin_lab/Everyone/yutaizhang/StringMC/sphere_six_point_1to5_blind30_3h_20260824_v1

exec "${SCRIPT_DIR}/stage_submit_sphere_six_point_1to5_blind50.sh" "$@"
