import os
from importlib.resources import files

import fetch_satdata.datacube.setup_datacube_run as setup_datacube_run
from fetch_satdata.workflows import run_snakemake


def run_create_datacube(
    catalog_filepath,
    timestamp_col,
    shapefilepath,
    id_col,
    run_folderpath,
    startdate,
    enddate,
    bands,
    scl_mask_classes,
    mosaic_days,
    csv_filepath,
    label_col,
    cores,
    dry_run: bool = False,
    overwrite_setup_csv: bool = True,
    unlock: bool = False,
):
    if overwrite_setup_csv:
        os.remove(csv_filepath)

    if not os.path.exists(csv_filepath):
        setup_datacube_run.main(
            catalog_filepath = catalog_filepath,
            timestamp_col = timestamp_col,
            shapefilepath = shapefilepath,
            id_col = id_col,
            run_folderpath = run_folderpath,
            startdate = startdate,
            enddate = enddate,
            bands = bands,
            scl_mask_classes = scl_mask_classes,
            mosaic_days = mosaic_days,
            csv_filepath = csv_filepath,
            label_col = label_col,
        )

    snakefile_path = files("fetch_satdata").joinpath("snakefiles/create_datacube_inmemory/Snakefile")

    config = {
        "input_csv": csv_filepath,
        "njobs": 1,
        "njobs_load_images": 1,
        "jitter_span": 1,
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

