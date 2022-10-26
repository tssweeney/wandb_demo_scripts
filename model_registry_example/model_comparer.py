import wandb


config = {
    "production": "wandb-artifact://auto-driver/model-registry/Production:latest",
    "challenger": "wandb-artifact://auto-driver/model-registry/Challengers:latest",
}

settings = wandb.Settings(enable_job_creation=True)
run = wandb.init(project="model_registry_example", job_type="model-comparer", config=config, settings=settings)
run.link_artifact(run.config["challenger"], "auto-driver/model-registry/Staging", aliases=["shadow"])
run.finish()