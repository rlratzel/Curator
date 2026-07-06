#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Assume this script is in the <repo_root>/benchmarking/tools directory
THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-""}
SLACK_BOT_TOKEN=${SLACK_BOT_TOKEN:-""}
SLACK_CHANNEL_ID=${SLACK_CHANNEL_ID:-""}
GDRIVE_FOLDER_ID=${GDRIVE_FOLDER_ID:-""}
GDRIVE_SERVICE_ACCOUNT_FILE=${GDRIVE_SERVICE_ACCOUNT_FILE:-""}
NVIDIA_API_KEY=${NVIDIA_API_KEY:-""}

# get the following vars from the command line, config file(s), etc. and
# set them in this environment:
#   RUN_TARGET
#   CURATOR_IMAGE
#   CONTAINER_NAME
#   SETUP_MODE
#   SETUP_SCRIPT_HOST_PATH
#   SETUP_SCRIPT_CONTAINER_PATH
#   GPUS
#   CONTAINER_MEMORY_BYTES
#   SHM_SIZE_BYTES
#   HOST_CURATOR_DIR
#   CURATOR_BENCHMARKING_DEBUG
#   VOLUME_MOUNTS
#   CONFIG_FILE_COPY_HOSTS
#   CONFIG_FILE_COPY_DESTS
#   CONTAINER_COMMAND
eval_str=$(python "${THIS_SCRIPT_DIR}/gen_runscript_vars.py" "${BASH_SOURCE[0]}" "$@")
eval "$eval_str"

if [ "${RUN_TARGET}" = "container" ] && [ -z "${CONTAINER_NAME}" ]; then
  echo "Internal error: RUN_TARGET is container but CONTAINER_NAME is empty" >&2
  exit 1
fi

if [ "${RUN_TARGET}" = "container" ] && [ ${#CONFIG_FILE_COPY_HOSTS[@]} -ne ${#CONFIG_FILE_COPY_DESTS[@]} ]; then
  echo "Internal error: config copy source/destination arrays differ in length" >&2
  exit 1
fi

# Get the image digest/ID for benchmark reports. This is not known at image build time.
if [ "${RUN_TARGET}" = "container" ]; then
  IMAGE_DIGEST=$(docker inspect "${CONTAINER_NAME}" --format '{{.Image}}' 2>/dev/null) || true
else
  IMAGE_DIGEST=$(docker image inspect "${CURATOR_IMAGE}" --format '{{.Digest}}' 2>/dev/null) || true
  if [ -z "${IMAGE_DIGEST}" ] || [ "${IMAGE_DIGEST}" = "<none>" ]; then
      # Use the image ID as a fallback
      IMAGE_DIGEST=$(docker image inspect "${CURATOR_IMAGE}" --format '{{.ID}}' 2>/dev/null) || true
  fi
fi
if [ -z "${IMAGE_DIGEST}" ] || [ "${IMAGE_DIGEST}" = "<none>" ]; then
    IMAGE_DIGEST="<unknown>"
fi

TARGET_WRAPPER='
set -euo pipefail
if [ "${CURATOR_BENCHMARK_SETUP_MODE}" != "skip" ]; then
  bash "${CURATOR_BENCHMARK_SETUP_SCRIPT}" --mode "${CURATOR_BENCHMARK_SETUP_MODE}" -- "$@"
fi
exec "$@"
'

DOCKER_ENV_ARGS=(
  --env=NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
  --env=IMAGE_DIGEST="${IMAGE_DIGEST}"
  --env=MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}"
  --env=SLACK_BOT_TOKEN="${SLACK_BOT_TOKEN}"
  --env=SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID}"
  --env=GDRIVE_FOLDER_ID="${GDRIVE_FOLDER_ID}"
  --env=GDRIVE_SERVICE_ACCOUNT_FILE="${GDRIVE_SERVICE_ACCOUNT_FILE}"
  --env=CURATOR_BENCHMARKING_DEBUG="${CURATOR_BENCHMARKING_DEBUG}"
  --env=HOST_HOSTNAME="$(hostname)"
  --env=NVIDIA_API_KEY="${NVIDIA_API_KEY}"
  --env=CURATOR_REPO_DIR=/opt/Curator
  --env=CURATOR_BENCHMARK_SETUP_MODE="${SETUP_MODE}"
  --env=CURATOR_BENCHMARK_SETUP_SCRIPT="${SETUP_SCRIPT_CONTAINER_PATH}"
)

copy_file_to_container() {
  local host_path="$1"
  local container_path="$2"
  docker exec "${CONTAINER_NAME}" mkdir -p "$(dirname "${container_path}")"
  docker cp "${host_path}" "${CONTAINER_NAME}:${container_path}"
}

if [ "${RUN_TARGET}" = "container" ]; then
  for idx in "${!CONFIG_FILE_COPY_HOSTS[@]}"; do
    copy_file_to_container "${CONFIG_FILE_COPY_HOSTS[$idx]}" "${CONFIG_FILE_COPY_DESTS[$idx]}"
  done

  if [ "${SETUP_MODE}" != "skip" ]; then
    copy_file_to_container "${SETUP_SCRIPT_HOST_PATH}" "${SETUP_SCRIPT_CONTAINER_PATH}"
  fi

  docker exec \
    --interactive \
    --tty \
    "${DOCKER_ENV_ARGS[@]}" \
    "${CONTAINER_NAME}" \
    bash -lc "${TARGET_WRAPPER}" \
    -- "${CONTAINER_COMMAND[@]}"
else
  GPUS_FLAGS=()
  if [ "${GPUS}" != "none" ]; then
    GPUS_FLAGS=(--gpus "${GPUS}")
  fi

  # --net=host allows the container to use the host's network stack, which Ray requires to
  # communicate between the container and the host. When running multiple benchmarks in parallel,
  # remove this flag so each container uses its own network namespace - this ensures each Ray
  # cluster is confined to its own container and can use the same default ports without
  # conflicting with other containers.
  docker run \
    --rm \
    --net=host \
    --interactive \
    --tty \
    "${GPUS_FLAGS[@]}" \
    --memory="${CONTAINER_MEMORY_BYTES}" \
    --shm-size="${SHM_SIZE_BYTES}" \
    "${VOLUME_MOUNTS[@]}" \
    "${DOCKER_ENV_ARGS[@]}" \
    --entrypoint=bash \
    "${CURATOR_IMAGE}" \
    -lc "${TARGET_WRAPPER}" \
    -- "${CONTAINER_COMMAND[@]}"
fi

exit $?
