def simple_run(cases):
    """Very simple SOM run for tensorflow testing."""

    num_cases = 5
    gridsize = 32
    max_epochs = 10

    # always load the same normal case for comparability
    with open(f"labels_normal{num_cases}.txt", "r") as f:
        sel_labels = [l.strip() for l in f]

    # data = cases.create_view(num=1, groups=GROUPS) # groups=["normal"], labels=simple_label)
    data = cases.create_view(num=num_cases, groups=["normal"], labels=sel_labels)

    # with open("labels_normal1.txt", "w") as f:
    #    f.writelines("\n".join([d.id for d in data]))

    # only use data from the second tube
    tubedata = data.get_tube(2)

    # load reference
    reference = pd.read_csv("sommaps_new/reference_ep10_s32_planar/t2.csv", index_col=0)
    marker_list = tubedata.data[0].markers if reference is None else reference.columns

    # reference = None
    model = tfsom.TFSom(
        m=gridsize,
        n=gridsize,
        channels=marker_list,
        batch_size=1,
        radius_cooling="exponential",
        learning_cooling="exponential",
        map_type="planar",
        node_distance="euclidean",
        max_epochs=max_epochs,
        initialization_method="random" if reference is None else "reference",
        reference=reference,
        tensorboard=True,
        tensorboard_dir=f'tensorboard_refit_refactor',
        model_name=f"remaptest_remap_n{num_cases}"
    )
    for result in model.fit_map(create_z_score_generator(tubedata.data)()):
        print(result)
    return

    for (lstart, lend) in [(0.4, 0.04), (0.8, 0.08), (0.8, 0.8), (1.0, 0.1)]:

        data_generator = create_z_score_generator(tubedata.data)

        model = tfsom.TFSom(
            m=gridsize,
            n=gridsize,
            channels=marker_list,
            batch_size=1,
            # initial_radius=20,
            # end_radius=1,
            initial_learning_rate=lstart,
            end_learning_rate=lend,
            radius_cooling="exponential",
            learning_cooling="exponential",
            map_type="planar",
            node_distance="euclidean",
            max_epochs=max_epochs,
            initialization_method="random" if reference is None else "reference",
            reference=reference,
            tensorboard=True,
            tensorboard_dir=f'tensorboard_refit_refactor',
            model_name=f"remaptest_l{lstart:3f}-{lend:.3f}_n{num_cases}"
        )
        model.train(
            data_generator(), num_inputs=len(data)  # len(tubedata.data)
        )
    return

    # metainfo
    weights = model.output_weights
    for data in data_generator():
        counts = model.map_to_histogram_distribution(data, relative=False)
        print(counts)
