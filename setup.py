import setuptools

with open("README.md", "r") as readme:
    long_description = readme.read()


setuptools.setup(
    name="flowcat",
    author="Max Zhao",
    author_email="max.zhao@charite.de",
    description="Classifier for flow cytometry data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages("."),
    entry_points={
        'console_scripts': ['flowcat=flowcat.cmdline:main'],
    },
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medicial Science Apps.",
    ],
    install_requires=[
        "keras==2.2.5",
        "argmagic>=0.0.10",
        "tensorflow==1.12",
        "dataslots",
        "scikit-learn",
        "pydot",
        "seaborn",
        "scipy",
        "matplotlib==3.1.0",
        "numpy==1.16.5",
        "pandas",
        "fcsparser @ git+https://github.com/xiamaz/fcsparser.git@bitmask#egg=fcsparser",
        "keras-vis @ git+https://github.com/jakob-he/keras-vis.git@multiinput#egg=keras-vis",
    ],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
)
