#!/usr/bin/env python3
#
#   libpushdf : LIBrary for computing Rational PushForwards of differential forms
#

from itertools import combinations

import numpy as np

from libpushdf.basic_types import Monomial, MonomialOrderGRevLex, MonomialOrderLex, Polynomial, PolynomialRing
from libpushdf.fgb import groebner_basis
from libpushdf.fglm import FGLM
from libpushdf.linalg import Matrix
from libpushdf.permutation import Sym
from libpushdf.primes_16bit import primes_16bit
from libpushdf.rational_interp import reconstruct_rational_function

def grobner_basis(polys):
    if len(polys) == 0:
        return []
    ring = polys[0].ring
    gb = groebner_basis(polys)
    return [Polynomial(ring, [Monomial(ring, exp) for exp in exps], coeffs) for exps,coeffs in gb]

def companion_matrix(z, B, G):
    b = len(B)
    # reserve
    T = Matrix(b, b)

    for i in range(b):
        # reduce with respect to G
        _, REM = (z * B[i]).divide(G)

        for j in range(b):
            # compare coeffs against standard basis
            T[i,j] = REM[B[j]]

    return T

def derivative_companion_matrix(DR, T, z, a, G, B):
    b = len(B)
    # reserve
    DT = Matrix(b, b)

    for i in range(b):
        # reduce with respect to G
        expr = (z * B[i]).differentiate(DR, a) - sum(DR(T[i,j]) * B[j].differentiate(DR, a) for j in range(b))
        _, REM = expr.divide(G)

        for j in range(b):
            # compare coeffs against standard basis
            DT[i,j] = REM[B[j]]

    return DT

def my_mtx_to_np_mtx(mtx):
    return np.array([a.x for a in mtx.entries], dtype=np.uint64).reshape(mtx.rows,mtx.cols)

def np_mul_mod(mtx1, mtx2, p):
    return np.matmul(mtx1, mtx2) % p

def np_mtx_is_zero(mtx):
    return not mtx.any()

class PushforwardCalculator:
    def __init__(self, z_var_names, a_var_names, form_basis):
        self.z_var_names = z_var_names
        self.a_var_names = a_var_names
        self.FORM_DEG = len(form_basis)
        self.PREFACTORS = tuple((form_basis,p) for p in combinations(range(len(self.a_var_names)), self.FORM_DEG))
        self.N_RETURNS = len(self.PREFACTORS)
        self.SYM = Sym(self.FORM_DEG)

    def ideal(self, Z, A):
        raise NotImplementedError()

    def form(self, Z):
        raise NotImplementedError()

    def ring_changed(self, ring : PolynomialRing):
        self.RA = ring
        self.A = self.RA.variables()
        self.RZ = PolynomialRing(self.RA.coeff_ring, self.z_var_names)
        self.Z = self.RZ.variables()
        self.RZ_A = PolynomialRing(self.RA, self.z_var_names)
        self.ZA = self.RZ_A.variables()

        self.IDEAL = self.ideal(self.ZA, self.A)
        self.FORM = self.form(self.Z)

        assert all(m.ring == self.RZ_A for m in self.IDEAL[0].monomials) , [m.ring for m in self.IDEAL[0].monomials]

        self.IDEAL_A = []
        self.D_RZ = []
        for a_var in self.a_var_names:
            self.D_RZ.append(self.RZ.extend_with_derivatives([a_var]))
            self.IDEAL_A.append(self.IDEAL + [p.differentiate(self.RZ_A.extend_with_derivatives([a_var]), a_var)
                                              for p in self.IDEAL])

    def calc_pushforward_evaluated(self, A_values):
        print("bbox")

        T = None
        T_Derivs = []

        for i,a_var in enumerate(self.a_var_names):
            d_ring = self.D_RZ[i]
            d_ring.order = MonomialOrderGRevLex

            z_vars = d_ring.variables()[d_ring.deriv_vars_end:]

            IDEAL_A_EV = [P.eval_coeffs(A_values, d_ring) for P in self.IDEAL_A[i]]

            G_grevlex = grobner_basis(IDEAL_A_EV)
            G,B = FGLM(G_grevlex, MonomialOrderLex)
            d_ring.order = MonomialOrderLex

            if i == 0:
                T = tuple(companion_matrix(z, B, G) for z in z_vars)

            T_Derivs.append(tuple(my_mtx_to_np_mtx(derivative_companion_matrix(d_ring, t, z, a_var, G, B))
                                  for z,t in zip(z_vars,T)))

        FORM_EV = my_mtx_to_np_mtx(self.FORM(T))

        factors = []

        PRIME = self.RA.coeff_ring.p

        for I,J in self.PREFACTORS:
            factor = np.zeros_like(FORM_EV)

            for perm in self.SYM:
                if any(np_mtx_is_zero(T_Derivs[J[k]][I[perm(k+1)-1]]) for k in range(self.FORM_DEG)):
                    pass
                else:
                    p = np.eye(len(B), dtype=np.uint64)
                    if perm.sign == -1:
                        np.fill_diagonal(p, PRIME - 1)

                    for k in range(self.FORM_DEG):
                        p = np_mul_mod(p, T_Derivs[J[k]][I[perm(k+1)-1]], PRIME)

                    factor += p

            factors.append(self.RA.coeff_ring(int(np.trace(np_mul_mod(FORM_EV, factor, PRIME)))))

        return factors

    def run(self):
        """
        Returns a list of Rational where numerator and denominator are Polynomials with coefficients in the rationals
        """
        results = reconstruct_rational_function(lambda *A_values : self.calc_pushforward_evaluated(A_values),
                                                lambda ring : self.ring_changed(ring), self.N_RETURNS,
                                                list(reversed(primes_16bit)), self.a_var_names)
        return results

    def print_tex(self, results):
        form_result = []
        for (P1,P2),result in zip(self.PREFACTORS,results):
            if result != 0:
                form_result.append(str(result) + " " + " \wedge ".join("d" + self.a_var_names[i] for i in P2))
        print(" + ".join(form_result))
