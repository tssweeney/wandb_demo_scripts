# dataset_generator.py
#
# This script represents a workload that generates a dataset for 
# a particular modelling use case.

import wandb
import util
import argparse

project             = "model_registry_example"
model_use_case_id   = "mnist"
job_type            = "dataset_builder"

# First, we launch a run which registers this workload with WandB.
parser = argparse.ArgumentParser()
parser.add_argument('--train_size', type=int, default=100, help='number of training examples')
run = wandb.init(project=project, job_type=job_type, config=parser.parse_args())

# Next, we generate the raw data. For simplicity, we will simply download the MNIST
# dataset and truncate it to our desired size. In a real world application, this 
# step may be more complicated and is domain specific.
(x_train, y_train), (x_eval, y_eval) = util.generate_raw_data(run.config.train_size)

# Finally, we publish this dataset to WandB. The utility method generates a WandB 
# Artifact that contains both training and evaluation data which can be visualized
# in the WandB UI.
util.publish_dataset_to_wb(x_train, y_train, x_eval, y_eval, model_use_case_id)