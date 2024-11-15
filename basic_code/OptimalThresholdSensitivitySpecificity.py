import numpy as np
from sklearn.metrics import recall_score

def optimal_threshold_sensitivity_specificity(thresholds,
                                              true_pos_rates,
                                              false_pos_rates,
                                              y_true,
                                              y_0_scores,
                                              return_thresh=False):
    """ Finds the optimal threshold then calculates sensitivity and specificity.

    Args:
        thresholds (list): the list of thresholds used in computing the ROC score.
        true_pos_rates (list): the TP rate corresponding to each thresholds.
        false_pos_rates (list): the FP rate corresponding to each threshold.
        y_true (list): the ground truth labels of the dataset over which
            sensitivity and specificity will be calculated.
        y_0_scores (list): the model's probability that each item in the dataset is
            class 0, (i.e. the positive class).
        return_thresh (boolean): if True, the calculated optimal threshold is returned

    Returns:
        sensitivity (float): True positive rate when optimal threshold is used
        specificity (float): True negative rate when optimal threshold is used
        accuracy (float): the percentage of lables that were correctly predicted, in [0,1]
        best_threshold (float): if return_thresh is true, this value is the
            decition threshold that maximized combined sensitivity and specificity
    """

    best_threshold = 0.5
    dist = -1
    for i, thresh in enumerate(thresholds):
        current_dist = np.sqrt((np.power(1-true_pos_rates[i], 2)) +
                               (np.power(false_pos_rates[i], 2)))
        if dist == -1 or current_dist <= dist:
            dist = current_dist
            best_threshold = thresh

    y_pred = (y_0_scores >= best_threshold) == False
    print(best_threshold)
    y_pred = np.array(y_pred, dtype=int)
    #accuracy = sum(y_pred == y_true)/len(y_true)
    sensitivity = recall_score(y_true, y_pred, pos_label=0)
    specificity = recall_score(y_true, y_pred, pos_label=1)
    accuracy = (sensitivity + specificity)/2

    if return_thresh:
        metrics = (sensitivity, specificity, accuracy, best_threshold)
    else:
        metrics = (sensitivity, specificity, accuracy)

    return metrics
