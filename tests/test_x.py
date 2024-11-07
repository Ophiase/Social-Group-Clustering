import unittest
from .utils import describe_test
import client.engine.clustering;

class TestSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        describe_test("X")

    def run(self, test):
        result = super().run(test)
        print("\n" + "=" * 30 + "\n")
        return result
    
    def test_cli(self):
        print("test")