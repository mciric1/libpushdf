#!/usr/bin/env python3

from libpushdf.basic_types import GF, Monomial, PolynomialRing
from libpushdf.linalg import Matrix

def linear_comb(V, vs):
    # obtain the linear combination of vs that produces V
    all_distinct_monoms = set()

    # collect monomials from each v
    for v in vs:
        for mon in v.monomials:
            all_distinct_monoms.add(mon.degrees)

    # check that we have all the monomials in V, otherwise we can't possibly have a linear combination
    if not all_distinct_monoms.issuperset(mon.degrees for mon in V.monomials):
        return False, None

    # alloc Matrix
    n_rows = len(all_distinct_monoms)
    n_cols = len(vs) + 1
    M = Matrix(n_rows, n_cols)

    # populate Matrix
    for r,mon in enumerate(all_distinct_monoms):
        for c,v in enumerate(vs):
            M[r,c] = v[mon]
        M[r,len(vs)] = V[mon]

    # row reduce
    Mr = M.reduced_row_echelon_form()

    # check that the system is not ill-conditioned, row reduced matrices are
    # upper-right-triangular so we need only check the diagonal for 0s
    if any(Mr[i,i] == 0 for i in range(Mr.cols - 1)): # - 1 for len(vs)
        return False, None

    # coefficients of the linear combination are in the last column
    coeffs = Mr.col(Mr.cols - 1)

    # sanity check, we should only ever have as many (potentially) non-zero coefficients as we have inputs (?)
    if not all(c == 0 for c in coeffs[len(vs):]):
        return False, None

    return True, coeffs[:len(vs)]

def FGLM(G, new_order):
    if len(G) == 0:
        return [], []
    ring : PolynomialRing = G[0].ring
    if ring.order == new_order:
        return G
    new_ring = PolynomialRing(ring.coeff_ring, ring.var_names, deriv_var_names=ring.deriv_var_names,
                              deriv_vars_end=ring.deriv_vars_end, order=new_order)
    if len(G) == 1 and G[0] == ring(1):
        return [new_ring(1)], []

    G2 = []
    B2 = [new_ring.mon0]
    B2_NFs = [new_ring(new_ring.mon0)]

    for n in range(ring.n_vars):
        # print("===",n,"===")
        i = 1
        while True:
            # TODO choose based on monomial order, currently only reverse lexicographic
            mon = Monomial(ring, (0 if j != ring.n_vars - 1 - n else i for j in range(ring.n_vars)))
            i += 1

            # compute normal form NF
            _, NF = ring(mon).divide(G)
            NF = new_ring(NF)
            mon.ring = new_ring

            # print(mon, "-G->", NF)

            # now check if NF is a linear combination of elements in B
            is_L, L = linear_comb(NF, B2_NFs)

            # print(is_L, L)

            if is_L:
                # if it is linear combination, we have a new element for G2
                g = mon - sum(new_ring(l) * b for l,b in zip(L,B2))
                # print(g)
                G2.append(g)
                # check termination
                g_lm = g.leading_monomial()
                if g_lm[0] != 0 and all(e == 0 for e in g_lm[1:]):
                    # print("Terminates")
                    return G2, list(reversed(B2))
                # move to next variable if not terminated
                break
            else:
                # if it is not, we have a new element for B2
                B2.append(mon)
                B2_NFs.append(NF)

    return G2, list(reversed(B2))

########################################################################################################################
#   Unit Tests
########################################################################################################################

import unittest

def test_linear_comb(tester : unittest.TestCase, cr, polys, coeffs):
    tester.assertEqual(len(polys), len(coeffs))
    result = sum(coeff * p for coeff,p in zip(coeffs,polys))
    coeffs = [cr(c) for c in coeffs]
    is_linear_comb, coeffs_out = linear_comb(result, polys)
    tester.assertTrue(is_linear_comb)
    tester.assertEqual(len(coeffs_out), len(coeffs))
    tester.assertEqual(set(coeffs), set(coeffs_out))

class TestLinearCombination(unittest.TestCase):

    def test(self):
        R = PolynomialRing(GF(509), "xyz")
        x,y,z = R.variables()

        test_linear_comb(self, R.coeff_ring, (x + 1, y**2 - 1, x*y), (12, 3, -2))
        test_linear_comb(self, R.coeff_ring, (x**2 + y, y - 3, x*y*z), (11, 14, 7))
        test_linear_comb(self, R.coeff_ring, (x*y + x**2 - y**2, y**2 + x**2, z**3), (4, 3, 2))
        test_linear_comb(self, R.coeff_ring, (x*y*z, x**3 - y**2, z**3), (4, 7, 1))
        test_linear_comb(self, R.coeff_ring, (x, y, z), (76, 14, 54))
        test_linear_comb(self, R.coeff_ring, (x**2, y**2, z**2), (11, 17, 23))

class TestFGLM(unittest.TestCase):

    def test(self):
        from libpushdf.basic_types import MonomialOrderLex

        R = PolynomialRing(GF(509), "xyz")
        x,y,z = R.variables()

        G = [
            980*z**2 - 18*y - 201*z + 13,
            35*y*z - 4*y + 2*z - 1,
            10*y**2 - y - 12*z + 1,
            5*x**2 - 4*y + 2*z - 1
        ]

        G2, B2 = FGLM(G, MonomialOrderLex)

        R2 = PolynomialRing(GF(509), "xyz", order=MonomialOrderLex)
        x,y,z = R2.variables()

        self.assertEqual(len(G2), 3)
        self.assertEqual(G2[0], z ** 3 + 102 * z ** 2 + 66 * z + 134)
        self.assertEqual(G2[1], y + 398 * z ** 2 + 96 * z + 480)
        self.assertEqual(G2[2], x ** 2 + 13 * z ** 2 + 179 * z + 282)

        self.assertEqual(len(B2), 4)
        self.assertEqual(B2[0], Monomial(R2, (1,0,0))) # x
        self.assertEqual(B2[1], Monomial(R2, (0,0,2))) # z**2
        self.assertEqual(B2[2], Monomial(R2, (0,0,1))) # z
        self.assertEqual(B2[3], Monomial(R2, (0,0,0))) # 1
