import mlflow
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME").lower()

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment:
    print(f"Experiment '{EXPERIMENT_NAME}' exists.")
    print(f"Experiment ID: {experiment.experiment_id}")
else:
    print(f"Experiment '{EXPERIMENT_NAME}' does not exist.")
    print(f"Creating experiment '{EXPERIMENT_NAME}'.")
    mlflow.create_experiment(
        EXPERIMENT_NAME, artifact_location=f"s3://{EXPERIMENT_NAME}/")

mlflow.set_experiment(EXPERIMENT_NAME)

mlflow.autolog()

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)
