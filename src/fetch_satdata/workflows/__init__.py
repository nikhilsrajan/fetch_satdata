import subprocess
import os
import signal


def run_snakemake(cmd):
    process = subprocess.Popen(
        cmd,
        start_new_session=True   # ← isolates the child into its own process group
    )
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nInterrupt received, stopping Snakemake...")
        os.killpg(process.pid, signal.SIGINT)   # send SIGINT to the process GROUP
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    return process.returncode

