from flowcat import utils


def remove_stem(marker):
    try:
        color, channel = marker.split("-")
    except ValueError:
        color = marker
    return color


def main():
    bonn_config = utils.load_json("output/00-dataset-test/bonn_config.json")
    munich_config = utils.load_json("output/00-dataset-test/train_config.json")

    selected = {}
    for tube, markers in bonn_config["selected_markers"].items():
        selected[tube] = []
        munich_tube = [remove_stem(m) for m in munich_config["selected_markers"][tube]]
        for marker in markers:
            marker_stem = remove_stem(marker)
            if marker_stem in munich_tube:
                selected[tube].append(marker_stem)
    print(selected)

    utils.save_json(selected, "output/00-dataset-test/munich_bonn_tubes.json")


if __name__ == "__main__":
    main()
