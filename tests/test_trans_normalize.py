import unittest

import cpmpy as cp
from cpmpy.transformations.normalize import  normalize_boolexpr


class TransSimplify(unittest.TestCase):
    def test_normalize(self):
        a = cp.boolvar(name='a')
        b = cp.boolvar(name='b')
        c = cp.boolvar(name='c')
        d = cp.boolvar(name='d')
        e = cp.boolvar(name='e')
        f = cp.boolvar(name='f')
        self.assertEqual(str(normalize_boolexpr([a | (b & c) | d])) ,'[or([a, b, d]), or([a, c, d])]')
        self.assertEqual(str(normalize_boolexpr([a | (b & (c & d)) | e]))
                         ,'[or([a, b, e]), or([a, c, e]), or([a, d, e])]')
        self.assertEqual(str(normalize_boolexpr([a | (b.implies(c)) | e])) ,'[or([a, ~b, c, e])]')
        self.assertEqual(str(normalize_boolexpr([a | (b.implies(c.implies(d))) | e])) ,'[or([a, ~b, ~c, d, e])]')
        self.assertEqual(str(normalize_boolexpr([a | ((b.implies(f)).implies(c.implies(d))) | e]))
                         ,'[or([a, b, ~c, d, e]), or([a, ~f, ~c, d, e])]')
        self.assertEqual(str(normalize_boolexpr([a | ((b.implies(f)).implies(c)) | e]))
                         ,'[or([a, b, c, e]), or([a, ~f, c, e])]')
        self.assertEqual(str(normalize_boolexpr([a | ((b.implies(f)).implies(c & d)) | e])),
                         ('[or([a, b, c, e]), or([a, b, d, e]), or([a, ~f, c, '
                          'e]), or([a, ~f, d, e])]'))
        self.assertEqual(str(normalize_boolexpr([b.implies(f & a & c)]))
                         ,'[(b) -> (f), (b) -> (a), (b) -> (c)]')
        self.assertEqual(str(normalize_boolexpr([(b | d).implies(f)])) ,'[(~f) -> (~b), (~f) -> (~d)]')
        self.assertEqual(str(normalize_boolexpr([(b | d).implies(f & a & c)])),
                         ('[(~f) -> (~b), (~f) -> (~d), (~a) -> (~b), (~a) -> (~d), '
                          '(~c) -> (~b), (~c) -> (~d)]'))
        self.assertEqual(str(normalize_boolexpr([(b.implies(c)).implies(f & a)]))
                         ,('[(~f) -> (b), (~f) -> (~c), (~a) -> (b), (~a) -> (~c)]'))
