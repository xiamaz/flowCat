import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, roc_curve

from . import base, file_utils
from .stats import auc, accuracy, experiment
from .base import Reporter, plot_figure


def name_diff(first, second):
    """Create shorter name from difference."""
    ftokens = {t for w in first for t in w.split("_")}
    stokens = {t for w in second for t in w.split("_")}

    uniq_tokens = ftokens ^ stokens
    common_tokens = [
        f for f in (ftokens & stokens) if f not in ["random", "dedup"]]

    name = "{}_comp_{}".format(
        "-".join(sorted(common_tokens)), "-".join(sorted(uniq_tokens))
    )
    return name


class Prediction(base.Reporter):
    """Create prediction analysis results from classification and analysis
    using different statistical methods.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._classification = None

    @property
    def classification(self):
        """Get relevant experiments."""
        if self._classification is None:
            filtered = self.classification_files.loc[
                self.classification_files["set"] != "old"
            ]
            pred = file_utils.add_prediction_info(filtered, latest_only=False)
            self._classification = pred.sort_index()
        return self._classification

    def select_sets(self, seta, setb):
        """Select data based on multiindex selection."""
        data_a = self.classification.loc[seta]
        if isinstance(data_a, pd.Series):
            data_a = data_a.to_frame().transpose()

        data_b = self.classification.loc[setb]
        if isinstance(data_b, pd.Series):
            data_b = data_b.to_frame().transpose()
        return data_a, data_b

    def compare(self, seta, setb, path):
        """Create comparisons for the comparison."""
        diff_name = name_diff(seta, setb)
        outdir = path / diff_name
        outdir.mkdir(exist_ok=True, parents=True)

        exp_a, exp_b = self.select_sets(seta, setb)

        def create_stat(exp):
            res = {}
            for name, row in exp.iterrows():
                res[name] = self.experiment_stats(row, outdir)
            stats_df = pd.concat(res.values(), keys=res.keys()).sort_index()
            return stats_df

        stat_a = create_stat(exp_a)
        stat_b = create_stat(exp_b)

        return experiment.ttest_exp_sets(stat_a, stat_b)


    def experiment_stats(self, row, path):
        """Create plots for the given experiment."""
        # load prediction data
        predictions = file_utils.load_predictions(row["predictions"])

        # roc plot path
        pname = "-".join(row.name)
        roc_path = path / "{}_auc.png".format(pname)
        if not roc_path.exists():
            with base.plot_figure(roc_path, figsize=(8, 8), dpi=200) as ax:
                auc.avg_roc_plot(predictions, ax)

        # get list of average binary auc per experiment
        auc_df = auc.auc_dfs(predictions)
        acc_df = accuracy.acc_table_dfs(predictions)
        stats_df = pd.concat([auc_df, acc_df])
        stats_df.index.rename(["subexp", "method"], inplace=True)

        return stats_df

    def write(self, path):
        print(self.classification_files)
        # create plots for each experiment
        # metadata = self.plot_experiments(path)
        # additionally save metadata as latex table
        # tpath = os.path.join(path, "prediction_meta.tex")
        # df_save_latex(metadata, tpath, "llllp{6cm}")
