import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Accuracy is the ratio of correctly predicted observations to the total observations.
    It is useful when the dataset is well-balanced (i.e., the number of positive and 
    negative samples is similar).

    Formula:
    Accuracy = (True Positives + True Negatives) / Total Observations
    
    Args:
    - y_true (np.ndarray): True labels (1D array).
    - y_pred (np.ndarray): Predicted labels (1D array).
    
    Returns:
    - accuracy (float): Accuracy score.
    """
    # Ensure y_true and y_pred have the same shape
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    
    # Calculate the number of correct predictions
    correct_predictions = np.sum(y_true == y_pred)
    
    # Accuracy is the number of correct predictions divided by total predictions
    accuracy = correct_predictions / len(y_true)
    return accuracy


def precision_score(y_true, y_pred):
    """
    Precision is the ratio of true positive observations to the total predicted positive observations.
    It answers the question: "Of all the instances the model predicted as positive, how many were actually positive?"

    Precision is especially useful when the cost of false positives is high (e.g., in spam detection, 
    where labeling a legitimate email as spam is costly).

    Formula:
    Precision = True Positives / (True Positives + False Positives)
    
    Args:
    - y_true (np.ndarray): True labels (1D array).
    - y_pred (np.ndarray): Predicted labels (1D array).
    
    Returns:
    - precision (float): Precision score.
    """
    # Ensure y_true and y_pred have the same shape
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    
    # True positives: y_true is 1 and y_pred is 1
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    
    # Predicted positives: y_pred is 1
    predicted_positives = np.sum(y_pred == 1)
    
    # Avoid division by zero
    if predicted_positives == 0:
        return 0.0
    
    # Precision = True positives / Predicted positives
    precision = true_positives / predicted_positives
    return precision


def f1_score(y_true, y_pred):
    """
    Calculate the F1 score.

    The F1 score is the harmonic mean of precision and recall. It provides a balance between precision 
    and recall. The F1 score is particularly useful in situations where you want to balance both false positives 
    and false negatives.

    Formula:
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

    If either precision or recall is 0, the F1 score will be 0.
    
    Args:
    - y_true (np.ndarray): True labels (1D array).
    - y_pred (np.ndarray): Predicted labels (1D array).
    
    Returns:
    - f1 (float): F1 score.
    """
    # Calculate precision and recall
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # Avoid division by zero
    if precision + recall == 0:
        return 0.0
    
    # F1 Score = 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def recall_score(y_true, y_pred):
    """
    Recall (also called sensitivity or true positive rate) is the ratio of true positive observations
    to the total actual positive observations. It answers the question: "Of all the instances that were actually positive, how many did the model correctly identify?"

    Recall is important when the cost of false negatives is high (e.g., in cancer detection, 
    where missing a positive case could be life-threatening).

    Formula:
    Recall = True Positives / (True Positives + False Negatives)
    
    Args:
    - y_true (np.ndarray): True labels (1D array).
    - y_pred (np.ndarray): Predicted labels (1D array).
    
    Returns:
    - recall (float): Recall score.
    """
    # True positives: y_true is 1 and y_pred is 1
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    
    # Actual positives: y_true is 1
    actual_positives = np.sum(y_true == 1)
    
    # Avoid division by zero
    if actual_positives == 0:
        return 0.0
    
    # Recall = True positives / Actual positives
    recall = true_positives / actual_positives
    return recall
