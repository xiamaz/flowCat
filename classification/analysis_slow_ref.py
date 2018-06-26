def top1_uncertainty_slow(data: pd.DataFrame, threshold=0.5) -> pd.DataFrame:
    """Adding a threshold below which cases are sorted to uncertain class."""
    results = {}
    for name, group in data.groupby("group"):
        gpreds = df_prediction_cols(group)

        pred = gpreds.apply(lambda r: predict(r, threshold), axis=1)
        correct = sum(pred == name)
        uncertain = sum(pred == "uncertain")

        results[name] = [
            correct/group.shape[0], uncertain/group.shape[0]
        ]
    return pd.DataFrame.from_dict(
        results, orient="index", columns=["correct", "uncertain"]
    )


def top2_slow(data: pd.DataFrame, normal_t1=True) -> pd.Series:
    """Get the top 2 classifications, while leaving normal as top 1."""
    results = {}

    for name, group in data.groupby("group"):
        gpreds = df_prediction_cols(group)
        if name == "normal" and normal_t1:
            preds = gpreds.apply(
                lambda r: "normal" == prediction_ranks(r)[0], axis=1
            )
        else:
            preds = gpreds.apply(
                lambda r: name in prediction_ranks(r)[:2], axis=1
            )
            n_preds = gpreds.apply(
                lambda r: "normal" == prediction_ranks(r)[0], axis=1
            )
            preds = preds & (~n_preds)

        acc = sum(preds) / group.shape[0]
        results[name] = [acc, 0, 1-acc]
    return pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=["correct", "uncertain", "incorrect"]
    )
