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

import argparse
import os
import shlex
import sys
from pathlib import Path

this_script_path = Path(__file__).parent.absolute()
# Add the parent directory to PYTHONPATH to import the runner modules
sys.path.insert(0, str(this_script_path.parent))

# ruff: noqa: E402
from runner.path_resolver import (
    CONTAINER_CURATOR_DIR,
    DEFAULT_CONTAINER_PATH_PREFIX,
    PathResolver,
)
from runner.utils import (
    assert_valid_config_dict,
    get_total_memory_bytes,
    merge_config_files,
)

_KB = 1024
_MB = 1024 * _KB
_GB = 1024 * _MB
_TB = 1024 * _GB
_max_container_memory_bytes = 2 * _TB
_shm_size_container_memory_percentage = 0.5  # 50%
_container_config_copy_prefix = Path("/tmp/curator-benchmarking/configs")  # noqa: S108
_setup_script_container_path = Path("/tmp/curator-benchmarking/setup_benchmark_env.sh")  # noqa: S108
_default_config_relative_path = Path("benchmarking/nightly-benchmark.yaml")

CURATOR_IMAGE = os.environ.get(
    "CURATOR_IMAGE",
    os.environ.get("CURATOR_BENCHMARKING_IMAGE", "nemo_curator:latest"),
)
GPUS = os.environ.get("GPUS", "all")
HOST_CURATOR_DIR = os.environ.get("HOST_CURATOR_DIR", str(this_script_path.parent.parent.absolute()))
CURATOR_BENCHMARKING_DEBUG = os.environ.get("CURATOR_BENCHMARKING_DEBUG", "0")
CURATOR_BENCHMARK_SETUP = os.environ.get("CURATOR_BENCHMARK_SETUP", "check")

CONTAINER_COMMAND = []
VOLUME_MOUNTS = []
CONFIG_FILE_COPY_HOSTS = []
CONFIG_FILE_COPY_DESTS = []


def print_help(script_name: str) -> None:
    """Print usage and help message for the run script (not this script) to stderr."""
    sys.stderr.write(f"""
  Usage: {script_name} [OPTIONS] [ARGS ...]

  Options:
      --use-host-curator       Mount $HOST_CURATOR_DIR into the container for benchmarking/debugging curator sources without rebuilding the image.
      --use-host-curator-benchmarking
                               Like --use-host-curator, but only for the benchmarking directory. This is useful for using the host benchmarking tools to benchmark Curator installed in the container.
      --image <image>          Docker image to use when starting a new benchmark container.
      --container <name-or-id> Run benchmarks or a shell in an already-running container. The container must already have required mounts, GPUs, shm, and env in place.
      --setup <mode>           Environment setup mode before running benchmarks: check, install, or skip (default: {CURATOR_BENCHMARK_SETUP}).
                               check validates the environment without installing packages. install runs benchmark environment setup first, then check. skip trusts the environment.
      --shell                  Start an interactive bash shell instead of running benchmarks. ARGS, if specified, will be passed to 'bash -c'.
                               For example: '--shell uv pip list | grep cugraph' will run 'uv pip list | grep cugraph' to display the version of cugraph installed in the container.
      --config <path>          Path to a YAML config file. Can be specified multiple times to merge configs.
                               Defaults to $HOST_CURATOR_DIR/benchmarking/nightly-benchmark.yaml when omitted and not using --shell.
      -h, --help               Show this help message and exit.

      ARGS, if specified, are passed to the command run inside the target environment, either the default benchmarking runner or the --shell bash command.

  Optional environment variables to override config and defaults:
      GPUS                          Value for --gpus option to docker run (using: {GPUS}).
      CURATOR_IMAGE                 Docker image to use when --image is not specified (using: {CURATOR_IMAGE}).
                                    CURATOR_BENCHMARKING_IMAGE is still accepted as a backward-compatible fallback.
      HOST_CURATOR_DIR              Curator repo path used with --use-host-curator (see above) (using: {HOST_CURATOR_DIR}).
      CURATOR_BENCHMARKING_DEBUG    Set to 1 for debug mode (regular output, possibly more in the future) (using: {CURATOR_BENCHMARKING_DEBUG}).
      CURATOR_BENCHMARK_SETUP       Default --setup value when omitted (using: {CURATOR_BENCHMARK_SETUP}).
    """)


def combine_dir_paths(patha: str | Path, pathb: str | Path) -> Path:
    """Combine two paths, ensuring the result is an absolute path."""
    return Path(f"{patha}/{pathb}").absolute().expanduser().resolve()


def default_config_file() -> Path:
    """Return the host-side default benchmark config path."""
    return Path(HOST_CURATOR_DIR).expanduser().absolute() / _default_config_relative_path


def container_copy_path(host_path: Path) -> Path:
    """Return the deterministic container path used when copying host files into a running container."""
    if not host_path.is_absolute():
        msg = f"Path '{host_path}' must be an absolute path."
        raise ValueError(msg)
    return _container_config_copy_prefix / host_path.relative_to("/")


def bash_array(values: list[str | Path]) -> str:
    """Return a bash array literal with shell-quoted values."""
    return "(" + " ".join(shlex.quote(str(value)) for value in values) + ")"


def bash_assign(name: str, value: str | Path | int) -> str:
    """Return a shell-safe scalar assignment."""
    return f"{name}={shlex.quote(str(value))}\n"


def get_runscript_eval_str(argv: list[str]) -> str:  # noqa: C901, PLR0912, PLR0915
    """
    Parse CLI args and output env variables for bash integration.
    argv is a list of strings to be treated like sys.argv, where argv[0] is the name of this script.
    argv[1] must be the name of the run.sh bash script, then all other args follow.
    Example: ["this_script.py", "run.sh", "--use-host-curator", "--shell", "--config", "config.yaml"]
    Returns a string of env variables to eval in bash.
    """
    # Make a copy of argv to avoid modifying the caller's list.
    argv = argv.copy()
    # Initialize with defaults, these will be modified in this function.
    container_command = CONTAINER_COMMAND.copy()
    volume_mounts = VOLUME_MOUNTS.copy()
    config_file_copy_hosts = CONFIG_FILE_COPY_HOSTS.copy()
    config_file_copy_dests = CONFIG_FILE_COPY_DESTS.copy()

    # This script must be called with the run.sh bash script as the first arg.
    # It will be removed before parsing the rest of the args.
    if len(argv) > 1:
        script_name = Path(argv[1]).name
        # Show help and exit if -h|--help is passed as first arg. All other options are passed to the
        # container command including -h|--help if other args are present. This provides a way for
        # -h|--help to be passed to the container command while still allowing for -h|--help output
        # for the run.sh script.
        if len(argv) > 2 and argv[2] in ("-h", "--help"):  # noqa: PLR2004
            print_help(script_name)
            sys.exit(1)
        argv.pop(1)
    else:
        msg = "Internal error: script name not provided"
        raise ValueError(msg)

    parser = argparse.ArgumentParser(
        description="Parse benchmarking tool options and output env variables for bash integration.",
        add_help=False,
    )
    parser.add_argument("--use-host-curator", action="store_true")
    parser.add_argument("--use-host-curator-benchmarking", action="store_true")
    parser.add_argument("--image", default=None)
    parser.add_argument("--container", default=None)
    parser.add_argument("--setup", choices=("check", "install", "skip"), default=CURATOR_BENCHMARK_SETUP)
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("--config", action="append", type=Path, default=[])

    args, unknown_args = parser.parse_known_args(argv[1:])
    image = args.image or CURATOR_IMAGE
    run_target = "container" if args.container else "image"
    is_list = "--list" in unknown_args

    if not args.config and not args.shell:
        config_file = default_config_file()
        if not config_file.is_file():
            msg = (
                f"Default benchmark config not found: {config_file}. "
                "Pass --config explicitly or set HOST_CURATOR_DIR to a Curator repo containing "
                f"{_default_config_relative_path}."
            )
            raise FileNotFoundError(msg)
        args.config = [config_file]

    # Set volume mount for host curator or benchmarking directories.
    if args.use_host_curator and args.use_host_curator_benchmarking:
        msg = "Cannot use --use-host-curator and --use-host-curator-benchmarking together."
        raise RuntimeError(msg)

    if args.container and args.image:
        msg = "Cannot use --image with --container."
        raise RuntimeError(msg)

    if args.container and (args.use_host_curator or args.use_host_curator_benchmarking):
        msg = "Cannot use host source mount options with --container; start the container with those mounts instead."
        raise RuntimeError(msg)

    if args.use_host_curator:
        # Do not use combine_dir_paths here since CONTAINER_CURATOR_DIR is assumed to be a unique absolute
        # path (e.g., /opt/Curator from Dockerfile).
        volume_mounts.extend(["--volume", f"{Path(HOST_CURATOR_DIR).absolute()}:{CONTAINER_CURATOR_DIR}"])

    if args.use_host_curator_benchmarking:
        volume_mounts.extend(
            [
                "--volume",
                f"{(Path(HOST_CURATOR_DIR) / 'benchmarking').absolute()}:{Path(CONTAINER_CURATOR_DIR) / 'benchmarking'}",
            ]
        )

    setup_script_host_path = this_script_path / "setup_benchmark_env.sh"
    if run_target == "image" and args.setup != "skip":
        volume_mounts.extend(["--volume", f"{setup_script_host_path}:{_setup_script_container_path}:ro"])

    # Parse config files and set volume mounts for all mapped directories.
    if args.config:
        config_dict = merge_config_files(args.config)
        assert_valid_config_dict(config_dict)

        if run_target == "image" and not is_list:
            path_resolver = PathResolver(config_dict)

            # Add a volume mount for each configured path.
            for host_path, container_path in path_resolver.volume_mount_pairs():
                if not host_path.is_absolute():
                    msg = f"Path '{host_path}' must be an absolute path."
                    raise ValueError(msg)
                volume_mounts.extend(["--volume", f"{host_path}:{container_path}"])

    # Add volume mounts or copy plans for config files so the script in the container can read each one
    # and add each to the benchmark runner args.
    runner_args = unknown_args.copy()
    for config_file in args.config:
        config_file_host = config_file.absolute().expanduser().resolve()
        if run_target == "image":
            container_dir_path = combine_dir_paths(DEFAULT_CONTAINER_PATH_PREFIX, config_file_host)
            volume_mounts.extend(["--volume", f"{config_file_host}:{container_dir_path}"])
        else:
            container_dir_path = container_copy_path(config_file_host)
            config_file_copy_hosts.append(config_file_host)
            config_file_copy_dests.append(container_dir_path)
        # Only add modified --config args if running the benchmark tool entrypoint, not the
        # bash shell entrypoint.
        if not args.shell:
            runner_args.append(f"--config={container_dir_path}")

    if args.shell:
        container_command = ["bash"]
        if len(unknown_args) > 0:
            container_command.extend(["-lc", " ".join(unknown_args)])
    else:
        container_command = ["python", f"{CONTAINER_CURATOR_DIR}/benchmarking/run.py", *runner_args]

    # The total available container memory will be the total host memory, but no more
    # than _max_container_memory_bytes.
    # The container shared memory size is a percentage of the container memory.
    container_memory_bytes = min(get_total_memory_bytes(), _max_container_memory_bytes)
    shm_size_bytes = int(container_memory_bytes * _shm_size_container_memory_percentage)

    # Build and return the string to eval in bash.
    eval_str = ""
    eval_str += bash_assign("RUN_TARGET", run_target)
    eval_str += bash_assign("CURATOR_IMAGE", image)
    eval_str += bash_assign("CONTAINER_NAME", args.container or "")
    eval_str += bash_assign("SETUP_MODE", args.setup)
    eval_str += bash_assign("SETUP_SCRIPT_HOST_PATH", setup_script_host_path)
    eval_str += bash_assign("SETUP_SCRIPT_CONTAINER_PATH", _setup_script_container_path)
    eval_str += bash_assign("GPUS", GPUS)
    eval_str += bash_assign("CONTAINER_MEMORY_BYTES", container_memory_bytes)
    eval_str += bash_assign("SHM_SIZE_BYTES", shm_size_bytes)
    eval_str += bash_assign("HOST_CURATOR_DIR", HOST_CURATOR_DIR)
    eval_str += bash_assign("CURATOR_BENCHMARKING_DEBUG", CURATOR_BENCHMARKING_DEBUG)
    eval_str += f"VOLUME_MOUNTS={bash_array(volume_mounts)}\n"
    eval_str += f"CONFIG_FILE_COPY_HOSTS={bash_array(config_file_copy_hosts)}\n"
    eval_str += f"CONFIG_FILE_COPY_DESTS={bash_array(config_file_copy_dests)}\n"
    eval_str += f"CONTAINER_COMMAND={bash_array(container_command)}\n"
    return eval_str


if __name__ == "__main__":
    print(get_runscript_eval_str(sys.argv))
