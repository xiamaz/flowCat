"""
Visualization of misclassifications
---
1. Load pretrained models from keras and list of labels of cases to visualize.
2. Create data sources for the selected labels
3. Modify model for visualization.
4. Generate visualizations for data.
"""
import pandas as pd

from sklearn import preprocessing
import keras

from map_class import inverse_binarize


def split_correctness(prediction):
    values = prediction.drop("correct", axis=1)
    truth = prediction["correct"]
    groups = [c for c in prediction.columns if c != "correct"]

    preds = inverse_binarize(values, groups)

    correct = truth == preds
    incorrect_data = prediction.loc[~correct, :]
    correct_data = prediction.loc[correct, :]
    return correct_data, incorrect_data


def sel_high(data):
    print(data.name)
    sortdata = data.sort_values(data.name, ascending=False)
    print(sortdata)
    seldata = sortdata.iloc[0:5, :]
    return seldata


def get_high_classified(prediction):
    high = prediction.groupby("correct").apply(sel_high)
    return high


def main():
    model = keras.models.load_model("mll-sommaps/models/selected5_toroid_8class_60test_ep100/model_0.h5")

    predictions = pd.read_csv("mll-sommaps/models/selected5_toroid_8class_60test_ep100/predictions_0.csv", index_col=0)

    correct, wrong = split_correctness(predictions)
    for name, group in correct.groupby("correct"):
        print(group)

    high_correct = get_high_classified(correct)


if __name__ == "__main__":
    main()
