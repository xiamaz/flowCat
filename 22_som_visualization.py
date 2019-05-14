"""
Load SOM csv data and plot them in 2D in some marker channels.

Use that to compare results obtained from flowSOM and own tensorflow SOM implementation.
"""
import flowcat


def plot_grid_tfsom(filepath, output):
    output = flowcat.utils.URLPath(output)
    soms = flowcat.load_som(filepath, subdirectory=False)
    som = soms.get_tube(1)
    output.local.mkdir(parents=True, exist_ok=True)
    flowcat.plots.plot_som_grid(som, output / "plot_test.png")


if __name__ == "__main__":
    plot_grid_tfsom("output/21-tfsom/native-missing", "output/22-plots")
