"""This module creates and runs models."""
from digitop.top import run_tensorflow
from digitop.top import build_keras_model
from digitop.classifier import evaluate_classifier
from digitop.classifier import plot_history
from digitop.classifier import iris_classifier
from digitop.classifier import load_iris_model

__version__ = top.version()
