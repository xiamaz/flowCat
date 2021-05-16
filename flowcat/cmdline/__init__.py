from argmagic import argmagic_subparsers

from .train import train
from .predict import predict
from .filter import filter
from .reference import reference
from .transform import transform
from .dataset import dataset


def main():
    argmagic_subparsers([
        {"target": predict},
        {"target": train},
        {"target": filter},
        {"target": reference},
        {"target": transform},
        {"target": dataset}
    ])
