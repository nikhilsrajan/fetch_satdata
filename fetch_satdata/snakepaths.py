"""
Paths to snakemake files
"""

import os

cur_dir = os.path.dirname(os.path.realpath(__file__))


CREATE_DATACUBE = os.path.abspath(os.path.join(cur_dir, '../snakemake/create_datacube_inmemory/Snakefile'))
DEMO_MODEL_DEPLOY = os.path.abspath(os.path.join(cur_dir, '../snakemake/demo_model_deploy/Snakefile'))
