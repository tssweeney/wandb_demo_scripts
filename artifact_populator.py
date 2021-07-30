from numpy.random.mtrand import randint
import wandb
import numpy as np

run = wandb.init(project="demo_project_2")

for teammate in range(3):
    team_id = str(np.random.randint(1000))
    for run_num in range(3):
        run_id = str(np.random.randint(1000))
        last_best = None
        for epoch in range(5):
            aliases = ["latest"]
            epoch_id = str(np.random.randint(100000)) + "" + str(epoch)
            art = wandb.Artifact("run-{}-model".format(team_id + run_id), "run_model")
            with open("model.h5", "w") as f:
                f.write(epoch_id)
            art.add_file("model.h5")
            if (np.random.randint(2) == 1) or epoch == 0:
                aliases.append("best")
                last_best = art
            run.log_artifact(art, aliases=aliases)
        last_best.wait()
        art2 = wandb.Artifact("candidate_model", "model")
        art2.add_reference(last_best.get_path("model.h5"))
        aliases = ["latest"]
        if (np.random.randint(2) == 1) or (teammate == 0 and run == 0):
            aliases.append("production")
        run.log_artifact(art2, aliases=aliases)

