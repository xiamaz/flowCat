import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
from flowcat import utils, io_functions


input_data = {
    p.name: io_functions.load_json(p / "quantization_error.json")
    for p in
    map(utils.URLPath, [
        "output/4-flowsom-cmp/quantization-error/flowsom-10",
        "output/4-flowsom-cmp/quantization-error/flowcat-refit-s10",
        "output/4-flowsom-cmp/quantization-error/flowsom-32",
        "output/4-flowsom-cmp/quantization-error/flowcat-refit-s32",
    ])
}

input_data = [
    {
        "dataset": k,
        "id": label,
        "tube": tube,
        "qe": value,
        "algo": k.split("-")[0],
        "size": int(k.split("-")[-1].lstrip("s")),
    }
    for k, vv in input_data.items() for label, tube, value in vv
]

data = pd.DataFrame(input_data)

sns.set_style("white")

sns.boxplot(x="size", y="qe", hue="algo", data=data)
plt.ylabel("Mean quantization error")
plt.xlabel("Grid size of SOM")
plt.savefig("output/4-flowsom-cmp/quantization_error_boxplot.png")
plt.close()
