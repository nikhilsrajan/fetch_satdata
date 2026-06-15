from importlib.resources import files

from fetch_satdata.workflows import run_snakemake


def demo_model_deploy(
    input_csv_filepath:str,
    cores:int,
    dry_run: bool = False,
    unlock: bool = False,
):
    snakefile_path = files("fetch_satdata").joinpath("snakefiles/demo_model_deploy/Snakefile")

    config = {
        "input_csv": input_csv_filepath,
    }

    cmd = [
        "snakemake",
        "--snakefile", str(snakefile_path),
        "--cores", str(cores),
        "--config", *[f"{k}={v}" for k, v in config.items()],
    ]

    if dry_run:
        cmd.append("--dry-run")

    if unlock:
        cmd.append("--unlock")

    result = run_snakemake(cmd=cmd)
    return result