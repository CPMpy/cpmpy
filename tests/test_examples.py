"""Tests all examples in the ../examples folder
with all the solvers available"""
from glob import glob
from os.path import join
import types
import importlib.machinery
import pytest

EXAMPLES = glob(join("..", "examples", "*.py")) + glob(join(".", "examples", "*.py"))

@pytest.mark.parametrize("example", EXAMPLES)
def test_examples(example):
    """Loads example files and executes with default solver

class TestExamples(unittest.TestCase):

    Args:
        example ([string]): Loaded with parametrized example filename
    """
    loader = importlib.machinery.SourceFileLoader("example", example)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
