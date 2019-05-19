"""
Load SOM csv data and plot them in 2D in some marker channels.

Use that to compare results obtained from flowSOM and own tensorflow SOM implementation.
"""
import flowcat


def plot_grid_tfsom(soms, output, tube=1):
    output = flowcat.utils.URLPath(output)
    som = soms.get_tube(tube)
    if som is None:
        print(f"Cannot get tube {tube} for plot in {output}")
        return
    output.local.mkdir(parents=True, exist_ok=True)
    plot_configs = {
        "cd45_ss": ("CD45-KrOr", "SS INT LIN", None),
    }
    for name, channels in plot_configs.items():
        flowcat.plots.plot_som_grid(
            som,
            output / f"plot_t{tube}_{name}.png",
            channels=channels
        )


def main():
    soms = flowcat.load_som("output/21-tfsom/native-missing", subdirectory=False)
    for tube in [1, 2, 3]:
        plot_grid_tfsom(soms, "output/22-plots/native-missing", tube=tube)


if __name__ == "__main__":
    main()
