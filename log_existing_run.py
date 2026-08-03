import mlflow
import yaml
import json

print("Tracking URI:", mlflow.get_tracking_uri())

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

with open("scores.json", "r") as f:
    scores = json.load(f)

with mlflow.start_run() as run:

    mlflow.log_param("BATCH_SIZE", params["BATCH_SIZE"])
    mlflow.log_param("CLASSES", params["CLASSES"])
    mlflow.log_param("EPOCHS", params["EPOCHS"])
    mlflow.log_param("IMAGE_SIZE", params["IMAGE_SIZE"])
    mlflow.log_param("LEARNING_RATE", params["LEARNING_RATE"])

    mlflow.log_metric("accuracy", scores["accuracy"])
    mlflow.log_metric("loss", scores["loss"])
    

    # Upload the already-trained model file as an artifact
    mlflow.log_artifact(
        "artifacts/training/model.h5",
        artifact_path="model"
    )

    print("Run ID:", run.info.run_id)
    print("Logged successfully!")