import unittest
import tempfile
from os.path import join
from os import rmdir

import cpmpy as cp

class TestModel(unittest.TestCase):
    
    def setUp(self) -> None:
        self.tempdir = tempfile.mkdtemp()
        print(self.tempdir)
        return super().setUp()
    
    def tearDown(self) -> None:
        rmdir(self.tempdir)
        return super().tearDown()

    def test_ndarray(self):
        iv = cp.intvar(1,9, shape=3)
        m = cp.Model( iv > 3 )
        m += (iv[0] == 5)
        self.assertTrue(m.solve())

    def test_empty(self):
        m = cp.Model()
        m += [] # should do nothing
        assert(len(m.constraints) == 0)

    def test_io_nempty(self):
        fname = join(self.tempdir, "model")
        iv = cp.intvar(1,9, shape=3)
        m = cp.Model( iv > 3 )
        m += (iv[0] == 5)
        m.to_file(fname)

        loaded = cp.Model.from_file(fname)
        self.assertTrue(loaded.solve())
