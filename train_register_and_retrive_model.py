from azureml.core import Workspace, Dataset, Experiment
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
from azureml.core import Model

# Access to Workspace
ws = Workspace.from_config("./config")

# Acess to Dataset
az_dataset = Dataset.get_by_name(ws, "adultincome_trunc")

# Create/Acess an Experiment
experiment = Experiment(workspace=ws, name="Webservice-exp-01")
print("the runs are as follow : ")
print(list(experiment.get_runs()))
print("-"*100)
# Start run experiment using start logging method
new_run = experiment.start_logging()

###############################################################################
# Data manipulation
###############################################################################

# convert Dataset azure to pandas dataframe
df = az_dataset.to_pandas_dataframe()

# Create X and Y variables
X = df.iloc[:, :-1]
Y = df.iloc[:, -1:]

# Create dummy variables
X = pd.get_dummies(X)

# Extract columns names including dummy variables
train_enc_cols = X.columns

# Transform Catgorical columns in Y dataset to dummy
Y = pd.get_dummies(Y)
Y = Y.iloc[:, -1]

# Split Data into X and Y test/train dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                        random_state=1234, stratify=Y)

# Build RandomForest classifier
rf = RandomForestClassifier(random_state=1234)

# Fit the classifier with the data
trained_model = rf.fit(X_train, Y_train)

# Predict the outcome using Test data - Score Model
Y_predict = rf.predict(X_test)

# Get the probability score
Y_probas = rf.predict_proba(X_test)[:, 1]

# Get confusion matrix and accuracy score
cm = confusion_matrix(Y_test, Y_predict)
score = rf.score(X_test, Y_test)
print("score : ", score)
new_run.log("accuracy", score)

###############################################################################
# Transformations and model serialization
###############################################################################

# Save model and transformations as pickle file
model_file = "./outputs/models.pkl"
joblib.dump(value=[train_enc_cols, trained_model], filename=model_file)

# Complete run
new_run.complete()

# Register model

names_models = [model.name for model in Model.list(ws)]

if "AdultIncome_model_local" not in names_models:
    Model.register(workspace=ws,
                   model_path='./outputs/models.pkl', # local path
                   model_name='AdultIncome_model_local',
                   tags={'source':'SDK-Local', 'algorithm':'RandomForest'},
                   properties={'Accuracy': 0.7866},
                   description='AdultIncome model from Local'
                   )

# Retrive model

for model in Model.list(ws):
    print("name {} and version {}".format(model.name, model.version))
    for tag in model.tags:
        value = model.tags[tag]
        print("value of tag is : ", value)

model_path = Model.get_model_path( model_name='AdultIncome_model_local', _workspace=ws)
ref_cols, predictor = joblib.load(model_path)

print("ref_cols : ", ref_cols)
print("predictor : ", predictor)