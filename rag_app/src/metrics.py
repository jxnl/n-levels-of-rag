from sklearn.metrics import ndcg_score
import numpy as np
from typing import Callable, List


def calculate_recall(chunk_id, predictions):
    return 0 if chunk_id not in predictions else 1


def calculate_mrr(chunk_id, predictions):
    return 0 if chunk_id not in predictions else 1 / (predictions.index(chunk_id) + 1)


def calculate_ndcg(chunk_id, predictions):
    if len(predictions) == 0:
        return "N/A"
    if len(predictions) == 1:
        return 1 if chunk_id in predictions else 0

    y_pred = np.linspace(1, 0, len(predictions)).tolist()
    y_true = [0 if item != chunk_id else 1 for item in predictions]

    return ndcg_score([y_true], [y_pred])


def slice_predictions_at_k(k: int, score: Callable[[str, List[str]], float]):
    def wrapper(chunk_id: str, predictions: List[str]):
        return score(chunk_id, predictions[:k])

    return wrapper
