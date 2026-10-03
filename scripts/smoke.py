"""Exercise the installed CLI without leaving generated data in the repository."""

import json
import shutil
import subprocess
import tempfile


def main():
    executable = shutil.which("blackbox")
    if executable is None:
        raise SystemExit("Install the project first: uv sync --locked --extra dev")
    with tempfile.TemporaryDirectory(prefix="blackbox-smoke-") as directory:
        command = [executable, "--data-dir", directory]

        def invoke(*args):
            result = subprocess.run([*command, *args], check=True, capture_output=True, text=True)
            return json.loads(result.stdout)

        sample = invoke("sample")
        trace = invoke("show", sample["run_id"])
        state = invoke("checkpoint", sample["run_id"], "4")
        if trace["outcome"] != "success" or len(trace["steps"]) != 8:
            raise SystemExit("Sample trace did not match the expected successful eight-step run")
        if len(state["messages"]) != 4:
            raise SystemExit("Checkpoint did not restore the state before step 4")
        print("CLI smoke check passed: record → inspect → restore")


if __name__ == "__main__":
    main()
