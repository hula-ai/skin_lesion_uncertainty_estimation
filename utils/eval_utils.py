import os

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils import compute_class_weight as sk_compute_class_weight
from utils.print_utils import print_confusion_matrix, print_precision_recall
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

MEL = 0  # Melanoma
NV = 1  # Melanocytic nevus
BCC = 2  # Basal cell carcinoma
AKIEC = 3  # Actinic keratosis / Bowen's disease (intraepithelial carcinoma)
BKL = 4  # Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
DF = 5  # Dermatofibroma
VASC = 6  # Vascular lesion

classes = [MEL, NV, BCC, AKIEC, BKL, DF, VASC]
class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']


def save_confusion_matrix(y_true, y_pred, classes, dest_path,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function plots and saves the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(dest_path)


def predictive_entropy(prob):
    """
    Entropy of the probabilities (to measure the epistemic uncertainty)
    :param prob: probabilities of shape [batch_size, C]
    :return: Entropy of shape [batch_size]
    """
    eps = 1e-5
    return -1 * np.sum(np.log(prob+eps) * prob, axis=1)


def mutual_info(mc_prob):
    """
    computes the mutual information
    :param mc_prob: List MC probabilities of length mc_simulations;
                    each of shape  of shape [batch_size, num_cls]
    :return: mutual information of shape [batch_size, num_cls]
    """
    eps = 1e-5
    mean_prob = mc_prob.mean(axis=0)
    first_term = -1 * np.sum(mean_prob * np.log(mean_prob + eps), axis=1)
    second_term = np.sum(np.mean([prob * np.log(prob + eps) for prob in mc_prob], axis=0), axis=1)
    return first_term + second_term


def get_confusion_matrix(y_true, y_pred, norm_cm=True, print_cm=True):
    true_class = np.argmax(y_true, axis=1)
    pred_class = np.argmax(y_pred, axis=1)

    cnf_mat = confusion_matrix(true_class, pred_class, labels=classes)

    total_cnf_mat = np.zeros(shape=(cnf_mat.shape[0] + 1, cnf_mat.shape[1] + 1), dtype=np.float)
    total_cnf_mat[0:cnf_mat.shape[0], 0:cnf_mat.shape[1]] = cnf_mat

    for i_row in range(cnf_mat.shape[0]):
        total_cnf_mat[i_row, -1] = np.sum(total_cnf_mat[i_row, 0:-1])

    for i_col in range(cnf_mat.shape[1]):
        total_cnf_mat[-1, i_col] = np.sum(total_cnf_mat[0:-1, i_col])

    if norm_cm:
        cnf_mat = cnf_mat/(cnf_mat.astype(np.float).sum(axis=1)[:, np.newaxis] + 0.001)

    total_cnf_mat[0:cnf_mat.shape[0], 0:cnf_mat.shape[1]] = cnf_mat

    if print_cm:
        print_confusion_matrix(cm=total_cnf_mat, labels=class_names + ['TOTAL', ])

    return cnf_mat


def get_precision_recall(y_true, y_pred, print_pr=True):

    true_class = np.argmax(y_true, axis=1)
    pred_class = np.argmax(y_pred, axis=1)
    precision, recall, _, _ = precision_recall_fscore_support(y_true=true_class,
                                                              y_pred=pred_class,
                                                              labels=classes,
                                                              warn_for=())
    if print_pr:
        print_precision_recall(precision=precision, recall=recall, labels=class_names)

    return precision, recall


def compute_class_weights(y, wt_type='balanced', return_dict=True):
    # need to check if y is one hot
    if len(y.shape) > 1:
        y = y.argmax(axis=-1)

    assert wt_type in ['ones', 'balanced', 'balanced-sqrt'], 'Weight type not supported'

    classes = np.unique(y)
    class_weights = np.ones(shape=classes.shape[0])

    if wt_type == 'balanced' or wt_type == 'balanced-sqrt':

        class_weights = sk_compute_class_weight(class_weight='balanced',
                                                classes=classes,
                                                y=y)
        if wt_type == 'balanced-sqrt':
            class_weights = np.sqrt(class_weights)

    if return_dict:
        class_weights = dict([(i, w) for i, w in enumerate(class_weights)])

    return class_weights


def jaccard(y_true, y_pred):
    intersect = np.sum(y_true * y_pred) # Intersection points
    union = np.sum(y_true) + np.sum(y_pred)  # Union points
    return (float(intersect))/(union - intersect +  1e-7)


def compute_jaccard(y_true, y_pred):

    mean_jaccard = 0.
    thresholded_jaccard = 0.

    for im_index in range(y_pred.shape[0]):

        current_jaccard = jaccard(y_true=y_true[im_index], y_pred=y_pred[im_index])

        mean_jaccard += current_jaccard
        thresholded_jaccard += 0 if current_jaccard < 0.65 else current_jaccard

    mean_jaccard = mean_jaccard/y_pred.shape[0]
    thresholded_jaccard = thresholded_jaccard/y_pred.shape[0]

    return mean_jaccard, thresholded_jaccard


def uncertainty_fraction_removal(y, y_pred, y_var, num_fracs, num_random_reps, save=False, save_dir=''):
    fractions = np.linspace(1 / num_fracs, 1, num_fracs)
    num_samples = y.shape[0]
    acc_unc_sort = np.array([])
    acc_pred_sort = np.array([])
    acc_random_frac = np.zeros((0, num_fracs))

    remain_samples = []
    # uncertainty-based removal
    inds = y_var.argsort()
    y_sorted = y[inds]
    y_pred_sorted = y_pred[inds]
    for frac in fractions:
        y_temp = y_sorted[:int(num_samples * frac)]
        remain_samples.append(len(y_temp))
        y_pred_temp = y_pred_sorted[:int(num_samples * frac)]
        acc_unc_sort = np.append(acc_unc_sort, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])

    # random removal
    for rep in range(num_random_reps):
        acc_random_sort = np.array([])
        perm = np.random.permutation(y_var.shape[0])
        y_sorted = y[perm]
        y_pred_sorted = y_pred[perm]
        for frac in fractions:
            y_temp = y_sorted[:int(num_samples * frac)]
            y_pred_temp = y_pred_sorted[:int(num_samples * frac)]
            acc_random_sort = np.append(acc_random_sort, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])
        acc_random_frac = np.concatenate((acc_random_frac, np.reshape(acc_random_sort, [1, -1])), axis=0)
    acc_random_m = np.mean(acc_random_frac, axis=0)
    acc_random_s = np.std(acc_random_frac, axis=0)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(fractions, acc_unc_sort, 'o-', lw=1.5, label='uncertainty-based', markersize=3, color='royalblue')

    line1, = ax.plot(fractions, acc_random_m, 'o', lw=1, label='Random', markersize=3, color='black')
    ax.fill_between(fractions,
                    acc_random_m - acc_random_s,
                    acc_random_m + acc_random_s,
                    color='black', alpha=0.3)
    line1.set_dashes([1, 1, 1, 1])  # 2pt line, 2pt break, 10pt line, 2pt break

    ax.set_xlabel('Fraction of Retained Data')
    ax.set_ylabel('Prediction Accuracy')
    if save:
        plt.savefig(os.path.join(save_dir, 'uncertainty_fraction_removal.png'))


def normalized_uncertainty_toleration_removal(y, y_pred, y_var, num_points, save=False, save_dir=''):
    acc_uncertainty, acc_overall = np.array([]), np.array([])
    num_cls = len(np.unique(y))
    y_var = (y_var - y_var.min()) / (y_var.max() - y_var.min())
    per_class_remain_count = np.zeros((num_points, num_cls))
    per_class_acc = np.zeros((num_points, num_cls))
    thresholds = np.linspace(0, 1, num_points)
    remain_samples = []
    for i, t in enumerate(thresholds):
        idx = np.argwhere(y_var >= t)
        y_temp = np.delete(y, idx)
        remain_samples.append(len(y_temp))
        y_pred_temp = np.delete(y_pred, idx)
        acc_uncertainty = np.append(acc_uncertainty, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])
        if len(y_temp):
            per_class_remain_count[i, :] = np.array([len(y_temp[y_temp == c]) for c in range(num_cls)])
            per_class_acc[i, :] = np.array(
                [np.sum(y_temp[y_temp == c] == y_pred_temp[y_temp == c]) / y_temp[y_temp == c].shape[0] for c in
                 range(num_cls)])

    plt.figure()
    plt.plot(thresholds, acc_uncertainty, lw=1.5, color='royalblue', marker='o', markersize=4)
    plt.xlabel('Normalized Tolerated Model Uncertainty')
    plt.ylabel('Prediction Accuracy')
    if save:
        plt.savefig(os.path.join(save_dir, 'uncertainty_toleration_removal.png'))


