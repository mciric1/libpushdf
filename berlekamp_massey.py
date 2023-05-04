#!/usr/bin/env python3
#
#   Berlekamp-Massey
#

from libpushdf.basic_types import GF, Monomial, Polynomial, PolynomialRing

class BerlekampMassey:
    def __init__(self, ring : PolynomialRing, var_num : int = 0):
        self.ring = ring
        self.var_num = var_num
        self.mon1 = Monomial(self.ring, (1 if i == var_num else 0 for i in range(ring.n_vars)))
        self.reset()

    def reset(self):
        self.C = self.ring(1)
        self.B = self.ring(0)
        self.L = 0
        self.D = self.ring.coeff_ring(1)
        self.M = self.mon1

        self.seq = []

    def add(self, a):
        a = self.ring.coeff_ring(a)
        if a in self.seq: # unlikely
            return False

        i = len(self.seq)
        self.seq.append(a)

        # Compute discrepancy
        d = sum(self.C[tuple(j if k == self.var_num else 0 for k in range(self.ring.n_vars))] * self.seq[i - j]
                for j in range(self.L + 1))

        if d == 0:
            # Discrepancy is 0, done
            return True

        cond = 2 * self.L < i + 1

        if cond:
            # make copy only if needed
            Cprev = self.C

        self.C -= Polynomial(self.ring, [self.M], [d * ~self.D]) * self.B

        if cond:
            # update state
            self.B = Cprev
            self.L = i + 1 - self.L
            self.D = d
            self.M = self.ring.mon0

        self.M *= self.mon1
        return False

    def result(self):
        R = self.C
        # reverse coefficients since this process produces the "reversed" minimal polynomial (c.f. Lee Thesis)
        R.coeffs = list(reversed(R.coeffs))
        return R

    @staticmethod
    def for_sequence(ring, seq):
        assert len(set(seq)) == len(seq)
        bm = BerlekampMassey(ring)
        for a in seq:
            bm.add(a)
        return bm.result()

########################################################################################################################
#   Unit Tests
########################################################################################################################

import unittest

class TestBerlekampMassey(unittest.TestCase):

    def test(self):
        R = PolynomialRing(GF(509), 'z')
        z = R.variables()[0]

        self.assertEqual(BerlekampMassey.for_sequence(R, (5, 17, 65, 257)),
                         (z**2 + 504 * z + 4))
        self.assertEqual(BerlekampMassey.for_sequence(R, (14, 242, 471, 130, 32, 45, 469)),
                         (z**3 + 488 * z**2 + 84 * z + 445))
        self.assertEqual(BerlekampMassey.for_sequence(R, (204, 309, 69, 70, 179, 218, 23)),
                         (z**3 + 360 * z**2 + 148))
