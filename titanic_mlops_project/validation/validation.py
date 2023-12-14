import numpy as np
from mlflow.models import make_metric, MetricThreshold

# Custom metrics to be included. Return empty list if custom metrics are not needed.
# Please refer to custom_metrics parameter in mlflow.evaluate documentation https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
# TODO(optional) : custom_metrics
def custom_metrics():

    # TODO(optional) : define custom metric function to be included in custom_metrics.
    return []


# Define model validation rules. Return empty dict if validation rules are not needed.
# Please refer to validation_thresholds parameter in mlflow.evaluate documentation https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
# TODO(optional) : validation_thresholds
def validation_thresholds():
    return {
        "accuracy_score": MetricThreshold(
            threshold=0.6,  #
            higher_is_better=True,
        ),
    }


# Define evaluator config. Return empty dict if validation rules are not needed.
# Please refer to evaluator_config parameter in mlflow.evaluate documentation https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
# TODO(optional) : evaluator_config
def evaluator_config():
    return {}
