import wandb

settings = wandb.Settings(enable_job_creation=True)

run = wandb.init(project="model_registry_example", job_type="alerter", settings=settings, config={
    "projectName": "model_registry_example",
    "entityname": "auto-driver",
    "model": "wandb-artifact://auto-driver/model_registry_example/model-neat-brook-9:latest",
})

name = run.config.model.name

wandb.alert(title="New production model ready!", text=f"New model ready in project {run.config.projectName} for entity {run.config.entityname} with name {name}")

run.finish()
