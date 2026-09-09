#!/bin/bash
# Same frozen-code staging as the earlier run, with a separate scan root.
set -euo pipefail
SCAN_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCAN_LOCAL_ROOT=$(cd "${SCAN_SCRIPT_DIR}/../.." && pwd)
if [[ $# != 3 ]]; then
  echo "usage: $0 SSH_HOST REMOTE_RUN_ROOT REMOTE_PYTHON" >&2
  exit 2
fi
export NSRR_G2_DRIVER=Code/genus_2/nsrr_nsnsns_theta_omega_scan.py
export NSRR_G2_DRIVER_TEST=Code/genus_2/test_nsrr_nsnsns_theta_omega_scan.py
export NSRR_G2_JOB_TAG=nsrr-omega-l6
export NSRR_G2_SOURCE_ARRAY_CAP=4
export NSRR_G2_TARGET_ARRAY_CAP=4
export NSRR_G2_SOURCE_TASKS_PER_ARRAY=70
export NSRR_G2_TARGET_TASKS_PER_ARRAY=70
export NSRR_G2_SOURCE_TIME=05:00:00
export NSRR_G2_TARGET_TIME=05:00:00
export NSRR_G2_SOURCE_MEMORY=8G
export NSRR_G2_TARGET_MEMORY=6G
exec bash "${SCAN_SCRIPT_DIR}/stage_submit_nsrr_nsnsns_theta_order8.sh" "$@" \
  "${SCAN_LOCAL_ROOT}/Data Set/nsrr_nsnsns_theta_omega_scan_submission.json" \
  "${SCAN_LOCAL_ROOT}/Code/config/nsrr_nsnsns_theta_omega_scan_20260830.json"
