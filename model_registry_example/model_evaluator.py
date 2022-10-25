# model_evaluator.py
#
# This script represents a workload that evaluates all models which
# have not yet been evaluated on the latest evaluation set. Moreover,
# it will also tag the best model with "production" so that other systems
# can use it for inference.

import wandb
import util
import argparse

project             = "hackweek_2022"
model_use_case_id   = "mnist"
job_type            = "evaluator"

settings = wandb.Settings(enable_job_creation=True)

# First, we launch a run which registers this workload with WandB.
run = wandb.init(project=project, job_type=job_type, settings=settings, config={
    "model_artifact": "wandb-artifact://auto-driver/model_registry_example/mnist_model_candidates:latest"
})

# Then we fetch the latest evaluation set.
x_eval, y_eval, dataset = util.download_eval_dataset_from_wb(model_use_case_id)

# Next we fetch the new candidate models for this use case
metric=f"{dataset.name}-ce_loss"
# candidates = util.get_new_model_candidates_from_wb(project, model_use_case_id, metric)

# Evaluate the models and save their metrics to wb.
# for model in candidates:
model = run.config["model_artifact"]
score = util.evaluate_model(model, x_eval, y_eval)
util.save_metric_to_model_in_wb(model, metric, score)

# Finally, promote the best model to production.
util.promote_best_model_in_wb(project, model_use_case_id, metric)

run.finish()
