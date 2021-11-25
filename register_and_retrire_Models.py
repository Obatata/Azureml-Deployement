from azureml.core import Workspace, Model, Experiment
import json

list_test = [1, 2, 44]
s = json.dumps(list_test)
print(s)


""""# acces the workspace
ws = Workspace.from_config("./config/config.json")

# Acess the run using run_id
run_by_id = ws.get_run("a3ee8448-fd80-4d97-ba78-20596b698772")

# register model from local to the run object
run_by_id.register_model(
                         model_path="/outputs/models.pkl",
                         model_name="AdultIncome_models",
                         tags={"source":"SDK RUN", "algorithm":"RandomForest"},
                         properties={"Accuracy":"0.79"},
                         description="Combined Models from Run"
                        )

"""

