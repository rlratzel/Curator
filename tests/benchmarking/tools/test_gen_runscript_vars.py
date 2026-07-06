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

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))


def _load_module(monkeypatch: pytest.MonkeyPatch, host_curator_dir: Path | None = None) -> types.ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.delenv("CURATOR_IMAGE", raising=False)
    monkeypatch.delenv("CURATOR_BENCHMARKING_IMAGE", raising=False)
    monkeypatch.delenv("CURATOR_BENCHMARK_SETUP", raising=False)
    monkeypatch.setenv("HOST_CURATOR_DIR", str(host_curator_dir or repo_root))
    sys.modules.pop("tools.gen_runscript_vars", None)
    return importlib.import_module("tools.gen_runscript_vars")


def _write_config(tmp_path: Path) -> Path:
    results_path = tmp_path / "results"
    datasets_path = tmp_path / "datasets"
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        f"""
paths:
  - name: results_path
    host_path: {results_path}
  - name: datasets_path
    host_path: {datasets_path}
datasets:
  - name: sample
    formats:
      - type: jsonl
        path: "{{datasets_path}}/sample.jsonl"
entries:
  - name: sample_entry
    script: sample_benchmark.py
    args: "--input-path={{dataset:sample,jsonl}}"
"""
    )
    return config_path


def test_no_config_defaults_to_host_nightly_config(monkeypatch: pytest.MonkeyPatch) -> None:
    gen_runscript_vars = _load_module(monkeypatch)
    default_config = (Path(__file__).resolve().parents[3] / "benchmarking/nightly-benchmark.yaml").resolve()

    eval_str = gen_runscript_vars.get_runscript_eval_str(["gen", "run.sh"])

    assert "CONTAINER_COMMAND=(python /opt/Curator/benchmarking/run.py" in eval_str
    assert f"--config=/MOUNT{default_config}" in eval_str


def test_list_uses_default_config_without_running_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    gen_runscript_vars = _load_module(monkeypatch)
    default_config = (Path(__file__).resolve().parents[3] / "benchmarking/nightly-benchmark.yaml").resolve()

    eval_str = gen_runscript_vars.get_runscript_eval_str(["gen", "run.sh", "--list"])

    assert "CONTAINER_COMMAND=(python /opt/Curator/benchmarking/run.py --list" in eval_str
    assert f"--config=/MOUNT{default_config}" in eval_str
    assert "/path/to/datasets" not in eval_str
    assert "/path/to/model_weights" not in eval_str
    assert "/path/where/results/are/stored" not in eval_str


def test_missing_default_config_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gen_runscript_vars = _load_module(monkeypatch, host_curator_dir=tmp_path / "missing")

    with pytest.raises(FileNotFoundError, match="Default benchmark config not found"):
        gen_runscript_vars.get_runscript_eval_str(["gen", "run.sh"])


def test_default_run_uses_standard_curator_image_and_setup_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gen_runscript_vars = _load_module(monkeypatch)
    config_path = _write_config(tmp_path)

    eval_str = gen_runscript_vars.get_runscript_eval_str(["gen", "run.sh", "--config", str(config_path)])

    assert "RUN_TARGET=image" in eval_str
    assert "CURATOR_IMAGE=nemo_curator:latest" in eval_str
    assert "SETUP_MODE=check" in eval_str
    assert "setup_benchmark_env.sh:/tmp/curator-benchmarking/setup_benchmark_env.sh:ro" in eval_str
    assert "CONTAINER_COMMAND=(python /opt/Curator/benchmarking/run.py" in eval_str
    assert f"--config=/MOUNT{config_path.resolve()}" in eval_str


def test_image_override_and_shell_command_skip_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    gen_runscript_vars = _load_module(monkeypatch)

    eval_str = gen_runscript_vars.get_runscript_eval_str(
        ["gen", "run.sh", "--image", "registry.example/curator:test", "--setup=skip", "--shell", "uv pip list"]
    )

    assert "RUN_TARGET=image" in eval_str
    assert "CURATOR_IMAGE=registry.example/curator:test" in eval_str
    assert "SETUP_MODE=skip" in eval_str
    assert "VOLUME_MOUNTS=()" in eval_str
    assert "CONTAINER_COMMAND=(bash -lc 'uv pip list')" in eval_str


def test_running_container_copies_configs_instead_of_mounting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gen_runscript_vars = _load_module(monkeypatch)
    config_path = _write_config(tmp_path)

    eval_str = gen_runscript_vars.get_runscript_eval_str(
        [
            "gen",
            "run.sh",
            "--container",
            "curator-dev",
            "--entries-exact",
            "sample_entry",
            "--config",
            str(config_path),
        ]
    )

    copied_config_path = f"{gen_runscript_vars._container_config_copy_prefix}{config_path.resolve()}"
    assert "RUN_TARGET=container" in eval_str
    assert "CONTAINER_NAME=curator-dev" in eval_str
    assert "VOLUME_MOUNTS=()" in eval_str
    assert f"CONFIG_FILE_COPY_HOSTS=({config_path.resolve()})" in eval_str
    assert f"CONFIG_FILE_COPY_DESTS=({copied_config_path})" in eval_str
    assert f"--config={copied_config_path}" in eval_str
    assert "--entries-exact sample_entry" in eval_str


def test_container_rejects_image_override(monkeypatch: pytest.MonkeyPatch) -> None:
    gen_runscript_vars = _load_module(monkeypatch)

    with pytest.raises(RuntimeError, match="Cannot use --image with --container"):
        gen_runscript_vars.get_runscript_eval_str(
            ["gen", "run.sh", "--container", "curator-dev", "--image", "curator:test"]
        )


def test_container_rejects_host_source_mount_options(monkeypatch: pytest.MonkeyPatch) -> None:
    gen_runscript_vars = _load_module(monkeypatch)

    with pytest.raises(RuntimeError, match="Cannot use host source mount options with --container"):
        gen_runscript_vars.get_runscript_eval_str(
            ["gen", "run.sh", "--container", "curator-dev", "--use-host-curator"]
        )
