#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

MODE=check
RUNNER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#--mode=}"
      shift
      ;;
    --)
      shift
      RUNNER_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown setup option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${MODE}" != "check" && "${MODE}" != "install" ]]; then
  echo "Invalid setup mode: ${MODE}; expected check or install" >&2
  exit 1
fi

find_curator_repo_dir() {
  if [[ -n "${CURATOR_REPO_DIR:-}" && -f "${CURATOR_REPO_DIR}/benchmarking/run.py" ]]; then
    echo "${CURATOR_REPO_DIR}"
    return
  fi

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local candidate
  candidate="$(cd "${script_dir}/../.." 2>/dev/null && pwd)" || true
  if [[ -n "${candidate}" && -f "${candidate}/benchmarking/run.py" ]]; then
    echo "${candidate}"
    return
  fi

  if [[ -f "/opt/Curator/benchmarking/run.py" ]]; then
    echo "/opt/Curator"
    return
  fi

  if [[ -f "$(pwd)/benchmarking/run.py" ]]; then
    pwd
    return
  fi

  echo "Unable to locate Curator repo; set CURATOR_REPO_DIR to the repo root" >&2
  exit 1
}

CURATOR_REPO_DIR="$(find_curator_repo_dir)"
export CURATOR_REPO_DIR
export PYTHONPATH="${CURATOR_REPO_DIR}/benchmarking:${PYTHONPATH:-}"

if [[ "${MODE}" = "install" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "setup install requires uv, but uv is not on PATH" >&2
    exit 1
  fi

  (
    cd "${CURATOR_REPO_DIR}"
    uv sync --extra all --all-groups
  )

  if command -v git >/dev/null 2>&1; then
    git config --global --add safe.directory "${CURATOR_REPO_DIR}" || true
  fi
fi

python - "${CURATOR_REPO_DIR}" "${RUNNER_ARGS[@]}" <<'PY'
from __future__ import annotations

import argparse
import importlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

repo_dir = Path(sys.argv[1])
runner_args = sys.argv[2:]
errors: list[str] = []
warnings: list[str] = []


def add_error(message: str) -> None:
    errors.append(message)


def check_repo() -> None:
    if not (repo_dir / "benchmarking/run.py").is_file():
        add_error(f"benchmarking/run.py was not found under {repo_dir}")
    if not (repo_dir / "benchmarking/scripts").is_dir():
        add_error(f"benchmarking/scripts was not found under {repo_dir}")


def check_imports() -> None:
    for module_name in [
        "nemo_curator",
        "runner.entry",
        "runner.path_resolver",
        "runner.session",
        "runner.utils",
    ]:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            add_error(f"failed to import {module_name}: {exc}")


def parse_runner_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", action="append", type=Path, default=[])
    parser.add_argument("--strict-config-check", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--entries", default=None)
    parser.add_argument("--entries-exact", default=None)
    args, _ = parser.parse_known_args(runner_args)
    return args


def validate_config(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912
    if not args.config:
        return

    from runner.entry import Entry
    from runner.session import Session
    from runner.utils import assert_valid_config_dict, merge_config_files, remove_disabled_blocks, resolve_env_vars

    for config_file in args.config:
        if not config_file.is_file():
            add_error(f"config file is not readable: {config_file}")

    if errors:
        return

    try:
        config_dict = merge_config_files(args.config)
        assert_valid_config_dict(config_dict)
        config_dict = remove_disabled_blocks(config_dict)
        config_dict = resolve_env_vars(config_dict, strict=args.strict_config_check)
    except Exception as exc:  # noqa: BLE001
        add_error(f"failed to load benchmark config: {exc}")
        return

    entries_exact = None
    if args.entries_exact is not None:
        entries_exact = [name.strip() for name in args.entries_exact.split(",") if name.strip()]

    try:
        session = Session.from_dict(config_dict, entry_filter_expr=args.entries, entries_exact=entries_exact)
    except Exception as exc:  # noqa: BLE001
        add_error(f"failed to create benchmark session from config: {exc}")
        return

    if args.list:
        return

    for name, resolved_path in session.path_resolver.path_map.items():
        path = Path(resolved_path)
        if name == "results_path":
            try:
                path.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(prefix=".curator-benchmark-check-", dir=path):
                    pass
            except Exception as exc:  # noqa: BLE001
                add_error(f"results path is not writable from this environment: {path} ({exc})")
        elif not path.exists():
            add_error(f"configured path '{name}' is not visible from this environment: {path}")

    dataset_pattern = re.compile(r"\{dataset:([^,}]+),([^}]+)\}")
    for entry in session.entries:
        if entry.script is None:
            add_error(f"entry '{entry.name}' does not define a script")
            continue

        script_path = entry.script_base_path / entry.script
        if not script_path.exists():
            add_error(f"entry '{entry.name}' script is not visible from this environment: {script_path}")

        for text in [entry.script or "", entry.args or ""]:
            for dataset_name, dataset_format in dataset_pattern.findall(text):
                try:
                    raw_path = session.dataset_resolver.resolve(dataset_name.strip(), dataset_format.strip())
                    resolved_path = Entry.substitute_container_or_host_paths(raw_path, session.path_resolver)
                except Exception as exc:  # noqa: BLE001
                    add_error(f"entry '{entry.name}' has unresolved dataset reference: {exc}")
                    continue

                if not Path(resolved_path).exists():
                    add_error(
                        f"entry '{entry.name}' dataset '{dataset_name},{dataset_format}' "
                        f"is not visible from this environment: {resolved_path}"
                    )

    required_gpus = 0
    for entry in session.entries:
        try:
            required_gpus = max(required_gpus, int(entry.ray.get("num_gpus") or 0))
        except (TypeError, ValueError):
            add_error(f"entry '{entry.name}' has invalid ray.num_gpus: {entry.ray.get('num_gpus')}")

    if required_gpus > 0:
        try:
            result = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True)
        except FileNotFoundError:
            add_error("selected entries require GPUs, but nvidia-smi is not on PATH")
        else:
            if result.returncode != 0:
                add_error(f"selected entries require GPUs, but nvidia-smi failed: {result.stderr.strip()}")
            else:
                visible_gpus = [line for line in result.stdout.splitlines() if line.startswith("GPU ")]
                if len(visible_gpus) < required_gpus:
                    add_error(
                        f"selected entries require {required_gpus} GPU(s), but only "
                        f"{len(visible_gpus)} are visible to this environment"
                    )

    try:
        shm_total = shutil.disk_usage("/dev/shm").total
    except FileNotFoundError:
        warnings.append("/dev/shm is not present; Ray may fail for some benchmarks")
    else:
        if shm_total < 1024**3:
            warnings.append(f"/dev/shm is small ({shm_total} bytes); Ray object store performance may suffer")


check_repo()
check_imports()
if not errors:
    validate_config(parse_runner_args())

for warning in warnings:
    print(f"setup check warning: {warning}", file=sys.stderr)

if errors:
    print("Benchmark environment check failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    sys.exit(1)

print("Benchmark environment check passed.")
PY
