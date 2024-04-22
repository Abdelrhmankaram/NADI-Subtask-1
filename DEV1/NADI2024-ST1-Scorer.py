import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def print_usage():
    print(
        """Usage:
    python3 NADI2024-ST1-Scorer.py NADI2024_subtask1_dev1_gold.txt UBC_subtask1_dev_1.txt
        """
    )


COUNTRIES = [
    "Algeria",
    "Bahrain",
    "Egypt",
    "Iraq",
    "Jordan",
    "Kuwait",
    "Lebanon",
    "Libya",
    "Morocco",
    "Oman",
    "Palestine",
    "Qatar",
    "Saudi_Arabia",
    "Sudan",
    "Syria",
    "Tunisia",
    "UAE",
    "Yemen",
]


# Load the file with predicitons
def load_predictions(filename, is_gold=False):
    with open(filename) as f:
        predictions_lists = [l.strip().split(",") for l in f]

    # Check the format of the file
    for i, p_l in enumerate(predictions_lists):
        assert len(p_l) == len(
            COUNTRIES
        ), f"The number of predictions in line {i+1} is not equal to the number of countries ({len(COUNTRIES)})"
        assert all(
            [p in ["0", "1"] if not is_gold else ["0", "1", ""] for p in p_l]
        ), f"Invalid prediction value in line {i+1}"

    return np.array(predictions_lists)


if __name__ == "__main__":
    verbose = 0
    if len(sys.argv) > 4 or len(sys.argv) < 3:
        print_usage()
        exit()

    if len(sys.argv) == 4 and sys.argv[3] != "-verbose":
        print_usage()
        exit()

    if len(sys.argv) == 4:
        verbose = 1

    gold_file = sys.argv[1]
    pred_file = sys.argv[2]

    gold_labels = load_predictions(gold_file, is_gold=True)
    predicted_labels = load_predictions(pred_file, is_gold=False)

    if len(gold_labels) != len(predicted_labels):
        print("both files must have same number of instances")
        exit()

    # Determine the columns for the countries in the gold file
    column_indecies_with_labels_in_gold = [
        i for i, l in enumerate(gold_labels[0]) if l in ["0", "1"]
    ]

    # Only consider the columns with labels in the gold file
    prediction_matrix = predicted_labels[:, column_indecies_with_labels_in_gold]
    gold_matrix = gold_labels[:, column_indecies_with_labels_in_gold]

    assert (
        gold_matrix.shape == prediction_matrix.shape
    ), "The number of instances and predictions in the gold and prediction files must be the same"

   # Compute the scores for each label (country) on its own
    accuracy_scores = [
        accuracy_score(y_true=gold_matrix[:, i], y_pred=prediction_matrix[:, i]) * 100
        for i in range(gold_matrix.shape[1])
    ]
    precision_scores = [
        precision_score(
            y_true=gold_matrix[:, i], y_pred=prediction_matrix[:, i], average="binary", pos_label="1",
        )
        * 100
        for i in range(gold_matrix.shape[1])
    ]
    recall_scores = [
        recall_score(
            y_true=gold_matrix[:, i], y_pred=prediction_matrix[:, i], average="binary", pos_label="1",
        )
        * 100
        for i in range(gold_matrix.shape[1])
    ]
    f1_scores = [
        f1_score(
            y_true=gold_matrix[:, i], y_pred=prediction_matrix[:, i], average="binary", pos_label="1",
        )
        * 100
        for i in range(gold_matrix.shape[1])
    ]

    # Compute the averaged scores
    average_accuracy = np.mean(accuracy_scores)
    average_precision = np.mean(precision_scores)
    average_recall = np.mean(recall_scores)
    average_f1 = np.mean(f1_scores)

    print("\nOVERALL SCORES:")
    ## prints overall scores (accuracy, f1, recall, precision)
    print("MACRO AVERAGE PRECISION SCORE: %.2f" % average_precision, "%")
    print("MACRO AVERAGE RECALL SCORE: %.2f" % average_recall, "%")
    print("MACRO AVERAGE F1 SCORE: %.2f" % average_f1, "%")
    print("MACRO AVERAGE ACCURACY: %.2f" % average_accuracy, "%\n")

    # write to a text file
    with open(pred_file.split("/")[-1].split(".")[0] + "_result.txt", "w") as out_file:
        out_file.write("OVERALL SCORES:")
        ## prints overall scores (accuracy, f1, recall, precision)
        out_file.write("MACRO AVERAGE PRECISION SCORE: %.2f \n" % average_precision)
        out_file.write("MACRO AVERAGE RECALL SCORE: %.2f  \n" % average_recall)
        out_file.write("MACRO AVERAGE F1 SCORE: %.2f  \n" % average_f1)
        out_file.write("MACRO AVERAGE ACCURACY: %.2f  \n" % average_accuracy)
