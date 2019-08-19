import pandas as pd


def df_get_count(data, tubes):
    """Get count information from the given dataframe with labels as index.
    Args:
        data: dict of dataframes. Index will be used in the returned count dataframe.
        tubes: List of tubes as integers.
    Returns:
        Dataframe with labels as index and ratio of availability in the given tubes as value.
    """
    counts = None
    for tube in tubes:
        count = pd.DataFrame(
            1, index=data[tube].index, columns=["count"])
        count.reset_index(inplace=True)
        count.set_index("label", inplace=True)
        count = count.loc[~count.index.duplicated(keep='first')]
        count.drop("group", axis=1, inplace=True, errors="ignore")
        if counts is None:
            counts = count
        else:
            counts = counts.add(count, fill_value=0)
    counts = counts / len(tubes)
    return counts
