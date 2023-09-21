import unittest

import cpmpy as cp
from cpmpy.transformations.normalize import  normalize_boolexpr


class TransSimplify(unittest.TestCase):
    def test_normalize(self):
        a ,b ,c ,d ,e ,f = cp.boolvar(shape=6)
        self.assertEqual(str(normalize_boolexpr([a | (b & c) | d])) ,'[or([BV0, BV1, BV3]), or([BV0, BV2, BV3])]')
        self.assertEqual(str(normalize_boolexpr([a | (b & (c & d)) | e]))
                         ,'[or([BV0, BV1, BV4]), or([BV0, BV2, BV4]), or([BV0, BV3, BV4])]')
        self.assertEqual(str(normalize_boolexpr([a | (b.implies(c)) | e])) ,'[or([BV0, ~BV1, BV2, BV4])]')
        self.assertEqual(str(normalize_boolexpr([a | (b.implies(c.implies(d))) | e])) ,'[or([BV0, ~BV1, ~BV2, BV3, BV4])]')
        self.assertEqual(str(normalize_boolexpr([a | ((b.implies(f)).implies(c.implies(d))) | e]))
                         ,'[or([BV0, BV1, ~BV2, BV3, BV4]), or([BV0, ~BV5, ~BV2, BV3, BV4])]')
        self.assertEqual(str(normalize_boolexpr([a | ((b.implies(f)).implies(c)) | e]))
                         ,'[or([BV0, BV1, BV2, BV4]), or([BV0, ~BV5, BV2, BV4])]')
        self.assertEqual(str(normalize_boolexpr([a | ((b.implies(f)).implies(c & d)) | e])),
                         ('[or([BV0, BV1, BV2, BV4]), or([BV0, BV1, BV3, BV4]), or([BV0, ~BV5, BV2, '
                          'BV4]), or([BV0, ~BV5, BV3, BV4])]'))
        self.assertEqual(str(normalize_boolexpr([b.implies(f & a & c)]))
                         ,'[(BV1) -> (BV5), (BV1) -> (BV0), (BV1) -> (BV2)]')
        self.assertEqual(str(normalize_boolexpr([(b | d).implies(f)])) ,'[(~BV5) -> (~BV1), (~BV5) -> (~BV3)]')
        self.assertEqual(str(normalize_boolexpr([(b | d).implies(f & a & c)])),
                         ('[(~BV5) -> (~BV1), (~BV5) -> (~BV3), (~BV0) -> (~BV1), (~BV0) -> (~BV3), '
                          '(~BV2) -> (~BV1), (~BV2) -> (~BV3)]'))
        self.assertEqual(str(normalize_boolexpr([(b.implies(c)).implies(f & a)]))
                         ,('[(~BV5) -> (BV1), (~BV5) -> (~BV2), (~BV0) -> (BV1), (~BV0) -> (~BV2)]'))
