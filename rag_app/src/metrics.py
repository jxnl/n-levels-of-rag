from sklearn.metrics import ndcg_score
import numpy as np


def calculate_mrr(chunk_id, predictions):
    print(len(predictions))
    return 0 if chunk_id not in predictions else 1 / (predictions.index(chunk_id) + 1)


def calculate_ndcg(chunk_id, predictions):
    print(len(predictions))
    y_pred = np.linspace(1, 0, len(predictions)).tolist()
    y_true = [0 if item != chunk_id else 1 for item in predictions]

    return ndcg_score([y_true], [y_pred])


def slice_predictions_decorator(num_elements: int):
    def decorator(func):
        def wrapper(chunk_id, predictions):
            sliced_predictions = predictions[:num_elements]
            return func(chunk_id, sliced_predictions)

        return wrapper

    return decorator
