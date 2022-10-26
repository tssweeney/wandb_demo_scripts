# model_trainer.py
#
# This script represents a workload that trains a model based on the
# latest training data set for a given modelling use case.

import wandb
import util
import argparse

entity              = "timssweeney"
project             = "hackweek_2022_post_kyle"
model_use_case_id   = "mnist"
job_type            = "model_trainer"

# First, we launch a run which registers this workload with WandB.
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',         type=int,   default=128)
parser.add_argument('--epochs',             type=int,   default=2)
parser.add_argument('--optimizer',          type=str,   default="adam")
parser.add_argument('--validation_split',   type=float, default=0.1)
parser.add_argument('--training_data',      type=str,   default=f"wandb-artifact://{entity}/{project}/{model_use_case_id}_ds:latest")

run = wandb.init(project=project, job_type=job_type, config=parser.parse_args(), settings=wandb.Settings(enable_job_creation=True))

# Next we download the latest training data available for this use case from WandB. 
# Again, the domain specific logic is abstracted away in a helper function.
x_train, y_train = util.download_training_dataset_from_wb(run.config['training_data'])

# Then we train a model using this data. For simplicity, we use a sequential model.
model = util.build_and_train_model(x_train, y_train, config=run.config)

# Finally, we publish the model to WandB. This will create a new artifact version
# that serves as a "candidate" model for this use case.
art = util.publish_model_candidate_to_wb(model, model_use_case_id)

# TODO: link model to candidates
run.link_artifact(art, f"{entity}/{project}/{model_use_case_id}_candidates")

run.finish()