import scipy
import json
import numpy as np
import pandas as pd
from typing import Dict, Union
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import ConfusionMatrixDisplay
#from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import f1_score, precision_score, recall_score, \
    accuracy_score

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None


def read_json(filename: str) -> dict:
    """
    This function provides a convenience function for reading in a file
    that is in the json standard.
    :param filename: string specifying the path to the json file
    :return: JSON-style dict with the contents from the file
    """
    with open(filename, 'r') as json_file:
        contents = json.load(json_file)
    return contents


def get_ground_truth_from_dataframe(dataframe: pd.DataFrame, col: str) -> Dict[str, list]:
    """
    This function takes as input the test dataframe, and return a dictionary
    with stream names as keys and the gold standard streams in
    binary vector format.

    """
    out = {}
    for doc_id, content in dataframe.groupby('name'):
        out[doc_id] = content[col].tolist()
    return out


def length_list_to_bin(list_of_lengths: Union[list, np.array]) -> Union[list, np.array]:
    """
    :param list_of_lengths:  containing the lengths of the individual documents
    in a stream as integers.
    :return: list representing the stream in binary format.
    """

    if not all([item > 0 for item in list_of_lengths]):
        raise ValueError

    # Set up the output array
    out = np.zeros(shape=(sum(list_of_lengths)))

    # First document is always a boundary
    out[0] = 1

    # if only one document return the current representation
    if len(list_of_lengths) == 1:
        if type(list_of_lengths) == list:
            return out.tolist()
        else:
            return out

    # Boundaries are at the cumulative sums of the number of pages
    # >>> doc_list = [2, 4, 3, 1]
    # >>> np.cumsum(doc_list) -> [2 6 9]

    # [:-1] because last document has boundary at end of array
    out[np.cumsum(list_of_lengths[:-1])] = 1
    if type(list_of_lengths) == list:
        return out.tolist()
    else:
        return out


def bin_to_length_list(binary_vector: Union[list, np.array]) -> Union[list, np.array]:
    """
    :param binary_vector: np array containing the stream of pages
    in the binary format.
    :return: A numpy array representing the stream as a list of
    document lengths.
    """

    # make sure the vector only contains 1s and zeros
    if not all([item in [0, 1] for item in binary_vector]):
        raise ValueError

    return_type = type(binary_vector)

    if type(binary_vector) == list:
        binary_vector = np.array(binary_vector)

    # We retrieve the indices of the ones with np.nonzero
    bounds = binary_vector.nonzero()[0]

    # We add the length of the array so that it works
    # with ediff1d, as this get the differences between
    # consecutive elements, and otherwise we would miss
    # the list document.
    bounds = np.append(bounds, len(binary_vector))

    # get consecutive indices
    out = np.ediff1d(bounds)

    if return_type == list:
        return out.tolist()
    else:
        return out


def make_index(binary_vec: Union[list, np.array]) -> Dict[int, set]:
    """

    :param binary_vec: Input vector containing a stream in the binary
    representation format.
    :return: A dictionary with the number of the page as key and the pages
    in the same document as the page (page number starts at 0).
    As an example, the stream [1, 0, 0, 1] is represented by the following
    dictionary: {0: {0, 1, 2}, 1: {0, 1, 2}, 2: {0, 1, 2}, 3: {3}}
    """

    # Make sure we accept both the list and numpy arrays types into the function
    if type(binary_vec) == list:
        binary_vec = np.array(binary_vec)

    splits = np.split(np.arange(len(binary_vec)), binary_vec.nonzero()[0][1:])
    repeated_splits = np.repeat(np.array(splits, dtype=object),
                                [len(split) for split in splits], axis=0)
    # Now we have a list of splits, we repeat this split n times, where n
    # is the length of the split
    out = {i: set(item) for i, item in enumerate(repeated_splits)}

    return out


def bcubed(truth, pred):
    assert len(truth) == len(pred)  # same amount of pages
    truth, pred = make_index(truth), make_index(pred)

    df = {i: {'size': len(truth[i]), 'P': 0, 'R': 0, 'F1': 0} for i in truth}
    for i in truth:
        df[i]['P'] = len(truth[i] & pred[i]) / len(pred[i])
        df[i]['R'] = len(truth[i] & pred[i]) / len(truth[i])
        df[i]['F1'] = (2 * df[i]['P'] * df[i]['R']) / (df[i]['P'] + df[i]['R'])

    return pd.DataFrame(df).T


def elm(truth, pred):
    assert len(truth) == len(pred)  # same amount of pages
    truth, pred = make_index(truth), make_index(pred)
    df = {i: {'size': len(truth[i]), 'P': 0, 'R': 0, 'F1': 0} for i in truth}
    for i in truth:
        TP = len((truth[i] & pred[i]) - {i})
        FP = len(pred[i] - truth[i])
        FN = len(truth[i] - pred[i])
        if pred[i] == {i}:
            df[i]['P'] = 1
        else:
            df[i]['P'] = TP / len(pred[i] - {i})
        if truth[i] == {i}:
            df[i]['R'] = 1
        else:
            df[i]['R'] = TP / len(truth[i] - {i})

        df[i]['F1'] = TP / (TP + .5 * (FP + FN)) if not pred[i] == {i} == truth[
            i] else 1
    return pd.DataFrame.from_dict(df, orient='index')


def hammingdamerau(gold: np.array, prediction: np.array) -> float:
    assert len(gold) == len(prediction)
    return damerauLevenshtein("".join([str(item) for item in gold]),
                              "".join([str(item) for item in prediction]),
                              similarity=False, insertWeight=10 ** 5,
                              deleteWeight=10 ** 5) / len(gold)


def window_diff(gold: np.array, prediction: np.array) -> float:
    assert len(gold) == len(prediction)
    # laten we in dit geval ervanuit gaan dat we k berekenen per document
    # En niet over het hele corpus.

    k = int(bin_to_length_list(gold).mean() * 1.5)
    # small check, in case of a singleton cluster, k will be too large
    # (mean == doc_length, k = 1.5*doclength)
    if k > len(gold):
        k = len(gold)

    # met de numpy functie kunnen we sliding windows pakken
    # dit doen we voor allebei de arrays en die vergelijken we dan.

    gold_windows = sliding_window_view(gold, window_shape=k)
    pred_windows = sliding_window_view(prediction, window_shape=k)

    # nu moeten we dus per window kijken of voor beiden de som gelijk is.
    gold_sum = gold_windows.sum(axis=1)
    pred_sum = pred_windows.sum(axis=1)

    # nu hebben we de som voor elke window in allebei de arrays
    # de score is nu gelijk aan de mean van de bool array

    return (gold_sum != pred_sum).mean()


def block_precision(gold: np.array, prediction: np.array) -> float:
    # hier gebruiken we np split. we splitten een stream van 1,2, 3, .., n
    # op aan de hand van de indices  nonzeros in de binaire vectors.
    # vervolgens maken we van deze partities 2 sets met subsets en
    # berekenen de grootte van de intersectie
    gold_splits = np.split(np.arange(len(gold)), gold.nonzero()[0][1:])
    pred_splits = np.split(np.arange(len(prediction)),
                           prediction.nonzero()[0][1:])

    gold_set = set([frozenset(item) for item in gold_splits])
    pred_set = set([frozenset(item) for item in pred_splits])

    return len(gold_set & pred_set) / len(pred_set)


def block_recall(gold: np.array, prediction: np.array) -> float:
    gold_splits = np.split(np.arange(len(gold)), gold.nonzero()[0][1:])
    pred_splits = np.split(np.arange(len(prediction)),
                           prediction.nonzero()[0][1:])

    gold_set = set([frozenset(item) for item in gold_splits])
    pred_set = set([frozenset(item) for item in pred_splits])

    return len(gold_set & pred_set) / len(gold_set)


def block_F1(gold: np.array, prediction: np.array) -> float:
    gold_splits = np.split(np.arange(len(gold)), gold.nonzero()[0][1:])
    pred_splits = np.split(np.arange(len(prediction)),
                           prediction.nonzero()[0][1:])

    # set of sets dan kunnen we makkelijk kijken welke subsets precies overeen komen
    gold_set = set([frozenset(item) for item in gold_splits])
    pred_set = set([frozenset(item) for item in pred_splits])

    P = len(gold_set & pred_set) / len(pred_set)
    R = len(gold_set & pred_set) / len(gold_set)

    # We have to be careful here, if both Precision and recall are zero
    # we just return 0
    if P == 0 and R == 0:
        return 0

    return (2 * P * R) / (P + R)


def f1(gold: np.array, prediction: np.array) -> float:
    return f1_score(gold, prediction)


def precision(gold: np.array, prediction: np.array) -> float:
    return precision_score(gold, prediction)


def recall(gold: np.array, prediction: np.array) -> float:
    return recall_score(gold, prediction)


def make_index_doc_lengths(split):
    l = sum(split)
    pages = list(np.arange(l))
    out = defaultdict(set)
    for block_length in split:
        block = pages[:block_length]
        pages = pages[block_length:]
        for page in block:
            out[page] = set(block)
    return out


def IoU_TruePositives(t, h):
    '''A True Positive is a pair h_block, t_block with an IoU>.5.
    This function returns the sum of all IoUs(h_block,t_block) for these bvlocks in t and h.'''

    def IoU(S, T):
        '''Jaccard similarity between sets S and T'''
        return len(S & T) / len(S | T)

    def get_docs(t):
        '''Get the set of documents (where each document is a set of pagenumbers)'''
        return {frozenset(S) for S in make_index_doc_lengths(t).values()}

    def find_match(S, Candidates):
        '''Finds, if it exists,  the unique T in Candidates such that IoU(S,T) >.5'''
        return [T for T in Candidates if IoU(S, T) > .5]

    t, h = get_docs(t), get_docs(h)  # switch to set of docs representation
    return sum(IoU(S, find_match(S, t)[0]) for S in h if find_match(S, t))


def IoU_P(t, h):
    return IoU_TruePositives(t, h) / len(h)


def IoU_R(t, h):
    return IoU_TruePositives(t, h) / len(t)


def IoU_F1(t, h):
    P, R = IoU_P(t, h), IoU_R(t, h)
    # todo, add the direct definition using FPs and FNs as well.
    # and test they are indeed equal
    return 0 if (P + R) == 0 else 2 * P * R / (P + R)


def calculate_metrics_one_stream(gold_vec, prediction_vec):
    out = {}

    gold_vec = np.array(gold_vec)
    prediction_vec = np.array(prediction_vec)

    prediction_vec[0] = 1
    scores = {'Accuracy': accuracy_score(gold_vec, prediction_vec),
              'Boundary': f1(gold_vec, prediction_vec),
              'Bcubed': bcubed(gold_vec, prediction_vec)['F1'].mean(),
              'ELM': elm(gold_vec, prediction_vec)['F1'].mean(),
              'Block': block_F1(gold_vec, prediction_vec),
              'Weighted Block': IoU_F1(bin_to_length_list(gold_vec),
                                       bin_to_length_list(prediction_vec))}

    scores_precision = {'Accuracy': accuracy_score(gold_vec, prediction_vec),
                        'Boundary': precision(gold_vec, prediction_vec),
                        'Bcubed': bcubed(gold_vec, prediction_vec)['P'].mean(),
                        'ELM': elm(gold_vec, prediction_vec)['P'].mean(),
                        'Block': block_precision(gold_vec, prediction_vec),
                        'Weighted Block': IoU_P(bin_to_length_list(gold_vec),
                                                bin_to_length_list(
                                                    prediction_vec))}

    scores_recall = {'Accuracy': accuracy_score(gold_vec, prediction_vec),
                     'Boundary': recall(gold_vec, prediction_vec),
                     'Bcubed': bcubed(gold_vec, prediction_vec)['R'].mean(),
                     'ELM': elm(gold_vec, prediction_vec)['R'].mean(),
                     'Block': block_recall(gold_vec, prediction_vec),
                     'Weighted Block': IoU_R(bin_to_length_list(gold_vec),
                                             bin_to_length_list(
                                                 prediction_vec))}

    out['precision'] = scores_precision
    out['recall'] = scores_recall
    out['F1'] = scores

    return out


def calculate_scores_df(gold_standard_dict, prediction_dict):
    all_scores = defaultdict(dict)
    for key in gold_standard_dict.keys():
        metric_scores = calculate_metrics_one_stream(gold_standard_dict[key],
                                                     prediction_dict[key])
        for key_m in metric_scores.keys():
            all_scores[key_m][key] = metric_scores[key_m]
    return {key: pd.DataFrame(val) for key, val in all_scores.items()}


def calculate_mean_scores(gold_standard_dict, prediction_dict,
                          show_confidence_bounds=True):
    scores_df = {key: val.T.mean().round(2) for key, val in
                 calculate_scores_df(gold_standard_dict,
                                     prediction_dict).items()}
    scores_combined = pd.DataFrame(scores_df)
    test_scores = scores_combined

    confidence = 0.95

    # total number of documents is the number of ones in the binary array
    n = sum([np.sum(item) for item in prediction_dict.values()])

    z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
    ci_length = z_value * np.sqrt((test_scores * (1 - test_scores)) / n)

    ci_lower = (test_scores - ci_length).round(2)
    ci_upper = (test_scores + ci_length).round(2)

    precision_ci = ci_lower['precision'].astype(str) + '-' + ci_upper[
        'precision'].astype(str)
    recall_ci = ci_lower['recall'].astype(str) + '-' + ci_upper[
        'recall'].astype(str)
    f1_ci = ci_lower['F1'].astype(str) + '-' + ci_upper['F1'].astype(str)

    out = pd.DataFrame(scores_df)
    out = out.rename({0: 'value'}, axis=1)
    out['support'] = sum(
        [np.sum(item).astype(int) for item in gold_standard_dict.values()])
    if show_confidence_bounds:
        out['CI Precision'] = precision_ci
        out['CI Recall'] = recall_ci
        out['CI F1'] = f1_ci

    return out


def show_kde_plots(gold_standard_dict, prediction_dict, save_name=""):
    # Make figures and axes here and plot

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    scores = calculate_scores_df(gold_standard_dict, prediction_dict)

    for i, (key, val) in enumerate(scores.items()):
        table = val.T
        # don't plot metrics without variance, as this throws and error
        # and it is also not very informative.

        table = table.loc[:, (table != table.iloc[0]).any()]

        table.plot(kind='kde', title="%s KDE for metrics" % key, ax=axes[i],
                   xlim=[0.0, 1.0])

    if save_name:
        plt.savefig(save_name)
    plt.show()


def evaluation_report(gold_standard_json, prediction_json, round_num=2,
                      title="", show_confidence_bounds=True):

    print(calculate_mean_scores(gold_standard_json, prediction_json,
                                  show_confidence_bounds=show_confidence_bounds).round(round_num))

def _convert_to_start_middle_end(binary_stream):
    out = []
    length_list = bin_to_length_list(np.array(binary_stream))
    for doc in length_list:
        if doc > 1:
            out.extend([1] + [0] * (doc - 2) + [2])
        else:
            out.extend([1])
    assert len(binary_stream) == len(out)
    return np.array(out)

