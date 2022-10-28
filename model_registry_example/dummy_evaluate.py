import wandb
import random
import string

def func():
    settings = wandb.Settings(enable_job_creation=True)
    run = wandb.init(project="model-registry", entity="vrajiv", job_type="dummy_evaluator", settings=settings)

    # generate random metric name
    letters = string.ascii_lowercase
    random_string = ''.join(random.choice(letters) for i in range(5)) 

    for _ in range(5):
        run.log({random_string: random.random()})

    # run.log_code()
    run.finish()


if __name__ == '__main__':
    func()
