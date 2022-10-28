# model_evaluator.py
#
# This script represents a workload that evaluates all models which
# have not yet been evaluated on the latest evaluation set. Moreover,
# it will also tag the best model with "production" so that other systems
# can use it for inference.

import wandb
import util
import argparse

entity = "timssweeney"
project = "hackweek_2022_post_kyle"
model_use_case_id = "mnist"
job_type = "evaluator"

settings = wandb.Settings(enable_job_creation=True)

# First, we launch a run which registers this workload with WandB.
run = wandb.init(
    entity=entity,
    project=project,
    job_type=job_type,
    settings=settings,
    config={
        "model_artifact": f"wandb-artifact://{entity}/{project}/{model_use_case_id}_candidates:production"
    },
)

current_production = run.use_artifact(run.config["model_artifact"])
run.link_artifact(
    current_production, f"{entity}/{project}/Production-{model_use_case_id}"
)

run.finish()
