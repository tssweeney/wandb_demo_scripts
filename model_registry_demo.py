import wandb
import util

project = "mr_testing_3"
model_use_case_id = "mnist"

with wandb.init(project=project, job_type="dataset_builder", config={
    "train_size": 100,
    "eval_size": 30,
}) as run:
    (x_train, y_train), (x_eval, y_eval) = util.generate_raw_data(
        train_size=run.config.train_size,
        eval_size=run.config.eval_size,
    )
    util.publish_dataset_to_wb(x_train, y_train, x_eval, y_eval, model_use_case_id)

for _ in range(3):
    with wandb.init(project=project, job_type="model_trainer", config={
        "batch_size": 128,
        "epochs": 5,
        "loss": "categorical_crossentropy",
        "optimizer": "adam",
        "metrics": ["accuracy"],
        "validation_split": 0.1,
    }) as run:
        x_train, y_train = util.download_training_dataset_from_wb(
            model_use_case_id)
        model = util.build_and_train_model(
            x_train,
            y_train,
            loss=run.config.loss,
            optimizer=run.config.optimizer,
            metrics=run.config.metrics,
            batch_size=run.config.batch_size,
            epochs=run.config.epochs,
            validation_split=run.config.validation_split)
        util.publish_model_candidate_to_wb(model, model_use_case_id)

with wandb.init(project=project, job_type="evaluator") as run:
    util.evaluate_and_promot_best_model_to_wb(project, model_use_case_id)
