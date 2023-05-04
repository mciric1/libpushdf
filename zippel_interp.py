#!/usr/bin/env python3
#
#   Zippel Interpolation Algorithm
#

from typing import Callable, List, Optional, Tuple

from libpushdf.basic_types import GF, Mod, Monomial, Polynomial, PolynomialRing, prod
from libpushdf.poly_uni_interp import UnivariateRacer
from libpushdf.vandermonde import solve_shifted_transposed_vandermonde

class MultivariateSparseInterpolator:
    """
    Given prior knowledge on which T monomials have coefficients that do not vanish, we can interpolate a polynomial
    with only T data points using a generalised vandermonde system.
    If some coefficients are known, the number of linear equations needed is reduced by however many are already known.
    """

    def __init__(self, ring : PolynomialRing, blackbox : Callable, powers : List[Tuple[Monomial, Optional[Mod]]],
                 var_nums=None):
        # powers that are non-zero in the target, some may have known coefficients, sorted by monomial ordering

        # powers = list(sorted(powers, key=lambda x: x[0]))

        self.ring = ring
        self.bbox = blackbox
        self.var_nums = var_nums or list(range(self.ring.n_vars))

        self.known_powers = [p.degrees for p,c in powers if c is not None]
        self.known_coeffs = [c for _,c in powers if c is not None]
        self.unknown_powers = [p.degrees for p,c in powers if c is None]

        self.known_part = Polynomial(self.ring, [Monomial(self.ring, p) for p in self.known_powers], self.known_coeffs)
        self.unknown_monoms = [Monomial(self.ring, p) for p in self.unknown_powers]

    def calc_v(self, anchors, degs):
        return prod(a ** p for a,p in zip(anchors,degs))

    def calc_f(self, anchors, i):
        # Only raise the base anchors that we want, keep any others fixed
        anchors_i = [a ** (i + 1) if n in self.var_nums else a for n,a in enumerate(anchors)]
        # For any known coefficients
        sub = sum(c * self.calc_v(anchors_i, p) for p,c in zip(self.known_powers,self.known_coeffs))
        # Calculate
        return self.bbox(*anchors_i) - sub

    def run(self, anchors):
        #anchors = [self.ring.coeff_ring(a) for a in anchors]
        vs = [self.calc_v(anchors, p) for p in self.unknown_powers]
        fs = [self.calc_f(anchors, i) for i,_ in enumerate(self.unknown_powers)]

        coeffs = solve_shifted_transposed_vandermonde(vs, fs)

        # number of required probes is len(fs) == len(coeffs)
        return Polynomial(self.ring, self.unknown_monoms, coeffs) + self.known_part

class ZippelInterpolator:

    def __init__(self, ring : PolynomialRing, blackbox : Callable, degree_bnd : int, anchors = None,
                 homogeneous : bool = False):
        # Supplied
        self.ring = ring
        self.bbox = blackbox
        self.degree_bnd = degree_bnd
        self.homogeneous = homogeneous

        # Derived
        if anchors is None:
            self.anchors_base = self.ring.coeff_ring.rand_elems(self.ring.n_vars, 2, -1)
            if self.homogeneous:
                self.anchors_base[0] = ring.coeff_one
        else:
            self.anchors_base = [self.ring.coeff_ring(a) for a in anchors]

        self.current_total = self.ring(1)
        self.powers = None

        self.start_n = 1 if self.homogeneous else 0
        self.n = None

    def next_var(self):
        # print("NEXT VAR")

        # set or advance n
        if self.n is None:
            self.n = self.start_n
        else:
            self.n += 1

        # complete
        if self.n == self.ring.n_vars:
            # print("done")

            if self.homogeneous:
                # restore the power of the first variable by matching all monomial degrees to the degree bound
                h_mons = [None] * len(self.current_total.monomials)

                for i,mon in enumerate(self.current_total.monomials):
                    degs = list(mon.degrees)
                    degs[0] = self.degree_bnd - mon.degree()
                    h_mons[i] = Monomial(self.ring, degs)

                self.current_total = Polynomial(self.ring, h_mons, self.current_total.coeffs)

            return True

        # print(self.n)

        # set anchors for this run
        self.anchors = self.anchors_base.copy()
        self.anchors[self.n] = self.ring.coeff_one

        if self.degree_bnd == 0:
            self.current_total = self.ring(self.bbox(*self.anchors))
            return True

        # we're going to interpolate each coefficient individually until they are all done, supply degree bound
        self.newtons = {mon : UnivariateRacer(self.ring, None, var_num=self.n, deg_bnd=self.degree_bnd - mon.degree())
                        for mon in self.current_total.monomials}
        self.results = {mon : None for mon in self.current_total.monomials}

        # record known powers for sparse interpolator
        self.powers = [[mon,None] for mon in self.current_total.monomials]

        # add any terms that are already at the total degree bound, as they cannot possibly interpolate to anything
        # other than a constant
        for j,(mon,coeff) in enumerate(zip(self.current_total.monomials, self.current_total.coeffs)):
            if mon.degree() == self.degree_bnd:
                # print(f"{mon} is already at max degree, coeff always evaluates to {coeff} in later interpolations")
                self.results[mon] = self.ring(coeff)
                self.powers[j][1] = coeff

        #if all(result is not None for result in self.results.values()):
        #    # sometimes a stage can finish like this if homogeneous and constant, in which case it's possible that no
        #    # evaluations were made
        #    self.anchors[self.n] *= self.anchors_base[self.n]
        #    self.current_total = self.ring(self.bbox(*self.anchors))
        #    return True

        self.used_prev_result = False
        return False

    def result(self):
        return self.current_total

    def step(self):
        # print("===== step =====")

        # while we still have newtons to complete
        self.anchors[self.n] *= self.anchors_base[self.n]

        # get data
        if self.n == self.start_n:
            # at the lowest stage just take numerical evaluations of the blackbox as-is
            f = self.ring(self.bbox(*self.anchors))
        elif not self.used_prev_result:
            # at higher stages we will have one prior result, the final value from the prior stage, use that if not
            # already used
            f = self.current_total
            self.used_prev_result = True
        else:
            # otherwise use a sparse interpolator now that the non-vanishing terms are known to reduce required
            # blackbox probes

            # if any newtons are done by now, we can evaluate those instead of querying the blackbox
            for j,mon in enumerate(self.current_total.monomials):
                if self.powers[j][1] is None and self.results[mon] is not None:
                    self.powers[j][1] = self.results[mon](self.anchors)

            # interpolate
            f = MultivariateSparseInterpolator(self.ring, self.bbox, self.powers,
                                               var_nums=list(range(self.n))).run(self.anchors)

        # step newton for each coefficient
        for mon in self.current_total.monomials:
            if self.results[mon] is None:
                self.results[mon] = self.newtons[mon].run_one(self.anchors[self.n], f[mon], self.anchors_base[self.n])
                #if self.results[mon] is not None:
                #    print(f"NEWTON DONE {mon} {self.results[mon]} {self.newtons[mon].n_done} {self.newtons[mon].bt_done}")

        # check if this stage is done
        if all(result is not None for result in self.results.values()):
            # assemble final result for this stage
            self.current_total = self.ring(sum(self.results[mon] * mon for mon in self.current_total.monomials))
            # print(f"STAGE {self.n + 1} RESULT = {self.current_total}")
            return True
        # stage is not done, await next step() call to continue interpolation
        return False

########################################################################################################################
#   Unit Tests
########################################################################################################################

import unittest
import time

def run_test(bbox, n_runs, degree_bound, homogeneous=False):
    n_bbox_evals = 0

    def bbox_wrap(*args):
        nonlocal n_bbox_evals
        n_bbox_evals += 1
        return bbox(*args)

    R = PolynomialRing(GF(65521), ''.join(bbox.__code__.co_varnames))

    EXPECTED = bbox_wrap(*R.variables())
    print(EXPECTED)

    t = time.time()
    for _ in range(n_runs):
        n_bbox_evals = 0
        interp = ZippelInterpolator(R, bbox_wrap, degree_bound, homogeneous=homogeneous)
        while not interp.next_var():
            while not interp.step():
                pass
        ev = interp.result()
        #t1 = time.time()
        assert ev == EXPECTED , f"Bad result {ev}"
        #t -= time.time() - t1
    print((time.time() - t) / n_runs)

class ZippelTester(unittest.TestCase):
    def run(self):
        p1 = lambda x,y : (x + 2) * (y + 3) * (x * y + 12)
        run_test(p1, 1000, 10)

        p2 = lambda x,y,z : x * y * z + 2 * x * y
        run_test(p2, 1000, 10)

        p3 = lambda x,y,z : x**3 * y**2 + 12*x*z - 4*z**2 + 17*y
        run_test(p3, 1000, 10)

        p4 = lambda x,y : 3*x**2 * 4*x*y + 2
        run_test(p4, 1000, 10)

        p5 = lambda x,y : 4*x*y - 3*x
        run_test(p5, 1000, 10)

        p6 = lambda x,y : x**6 + x**4 * y**2 + y**6
        run_test(p6, 1000, 10)

        p7 = lambda x,y : x * (x - y) * (y - 1)
        run_test(p7, 1000, 10)

        p8 = lambda x,y,z : y**2
        run_test(p8, 1000, 2, True)

        p9 = lambda x,y : 12
        run_test(p9, 1000, 0)

        p10 = lambda x,y,z : 12
        run_test(p10, 1000, 0, True)
