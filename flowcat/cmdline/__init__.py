from argmagic import argmagic

from .train import train
from .predict import predict
from .filter import filter
from .reference import reference
from .transform import transform


def main():
    argmagic([predict, train, filter, reference, transform])
