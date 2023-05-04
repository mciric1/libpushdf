#!/usr/bin/env python3
#
#   Racing Newton against Ben-or/Tiwari
#

from libpushdf.basic_types import Monomial, Polynomial, PolynomialRing
from libpushdf.berlekamp_massey import BerlekampMassey
from libpushdf.vandermonde import solve_shifted_transposed_vandermonde

class UnivariateRacer:
    """
    Interpolates a univariate polynomial by racing the dense Newton interpolation scheme against the
    sparse Ben-Or/Tiwari interpolation scheme.
    """

    def __init__(self, ring : PolynomialRing, bbox, var_num : int = 0, deg_bnd : int = 1000):
        self.ring : PolynomialRing = ring
        self.bbox = bbox
        self.bm = BerlekampMassey(self.ring, var_num=var_num)
        self.n_vars = self.ring.n_vars
        self.var_num = var_num
        self.deg_bnd = deg_bnd
        self.max_i = 2 * self.deg_bnd + 1

        self.mon0 = Monomial(self.ring, (0 for _ in range(self.n_vars)))
        self.mon1 = Monomial(self.ring, (1 if j == self.var_num else 0 for j in range(self.n_vars)))

        #self.eval_template = [self.ring.coeff_zero] * self.n_vars

        self.reset()

    def reset(self):
        self.xs_n = []
        self.fxs_n = []

        self.diffs = []
        self.coeffs = []
        self.poly_base = []

        self.last_insert = 0

        self.n_probes = 0
        self.bad_anchors = [None]
        self.anchor = None
        self.reset_bt()

        self.n_done = False

    def new_anchor(self):
        # Find an anchor that we don't already know is bad
        while self.anchor in self.bad_anchors:
            self.anchor = self.ring.coeff_ring.rand_elem(min=2)
        self.TT = 1

    def reset_bt(self):
        self.xs_bt = []
        self.fxs_bt = []
        self.bt_done = False
        self.bm.reset()
        self.new_anchor()

    def newton_step(self, x, fx):
        if x in self.xs_n:
            return

        self.xs_n.append(x)
        self.fxs_n.append(fx)
        self.diffs.append(fx)
        this_insert = len(self.diffs) - 1

        for i in range(this_insert - self.last_insert):
            self.diffs.append(
                (self.diffs[this_insert + i] - self.diffs[self.last_insert + i]) / (self.xs_n[-1] - self.xs_n[-2 - i]))

        coeff = self.diffs[-1]

        self.coeffs.append(coeff)
        self.last_insert = this_insert

        if len(self.xs_n) > 1:
            poly_base = Polynomial(self.ring, [self.mon1, self.mon0], [1, -self.xs_n[-2]])
            poly_base = poly_base * self.poly_base[-1]
        else:
            poly_base = Polynomial(self.ring, [self.mon0], [1])

        self.poly_base.append(poly_base)

        # either coefficient is 0 or we reach the degree bound
        self.n_done = (len(self.xs_n) > 1 and coeff == 0) or poly_base.degree() == self.deg_bnd

    def bt_step(self, x, f):
        self.xs_bt.append(x)
        self.fxs_bt.append(f)
        self.bt_done = self.bm.add(f)

    def result(self, base=None):
        if self.n_done:
            # print("newton done")
            return sum(p * c for c,p in zip(self.coeffs,self.poly_base))
        elif self.bt_done:
            # print("bt done")
            assert self.anchor is not None
            Lambda = self.bm.result()

            # print(Lambda)

            if Lambda == 1:
                # print("no roots, bt fail")
                self.bad_anchors.append(self.anchor)
                # bad, minimal polynomial has no roots
                return None

            yis = []

            nonzero_degrees = []
            roots_found = 0
            roots_to_find = (self.TT - 1) // 2
            # print(roots_to_find)
            i = 0
            yi = 1
            while roots_found != roots_to_find:
                if i >= self.max_i:
                    # Most likely a wrong early termination
                    return None

                # yi is self.anchor ** i but it's worse to write that

                if yi in yis:
                    # print(f"{self.anchor} {yis} {i} Wrong early termination, restart")
                    self.bad_anchors.append(self.anchor)
                    return None

                if Lambda(tuple(yi if k == self.var_num else 0 for k in range(self.ring.n_vars))) == 0:
                    nonzero_degrees.append(i)
                    yis.append(yi)
                    roots_found += 1

                yi *= base
                i += 1

            coeffs = solve_shifted_transposed_vandermonde(yis, self.fxs_bt[:len(nonzero_degrees)])
            return Polynomial(self.ring,
                              [Monomial(self.ring, tuple(d if k == self.var_num else 0
                                                         for k in range(self.ring.n_vars)))
                               for d in reversed(nonzero_degrees)],
                              list(reversed(coeffs)))
        return None

    def run_one(self, x, f, base=None):
        # step
        # print("NEWTON step")

        if f == 0:
            # Bad anchor, must avoid roots of the function since we won't be able to generate any more points after
            # this one (0*a = 0)
            self.bad_anchors.append(self.anchor)
            self.reset_bt()
        else:
            self.TT += 1
            self.bt_step(x,f)

        self.newton_step(x,f)

        # check if done

        if self.n_done or self.bt_done:
            # if the last run terminated
            res = self.result(base)

            if res is not None:
                # TODO early termination may be wrong, check with extra evaluation

                # self.eval_template[self.var_num] = x
                # if res(self.eval_template) == f:
                return res # done

            # reset bt, but not newton
            self.reset_bt()

        return None

    def next_point(self):
        """
        Generate the next evaluation of the black box function.

        In the racing algorithm, we want evaluations at powers of a predetermined and random anchor point for the sake
        of the Ben-or/Tiwari algorithm. The randomly selected anchor may be a bad choice with some probability, in that
        case we must restart the Ben-or/Tiwari algorithm but need not restart Newton.
        """

        x = self.anchor ** self.TT
        self.TT += 1

        if x in self.xs_bt:
            # anchor forms a small cyclic group in the finite field, reset bt
            self.bad_anchors.append(self.anchor)
            self.reset_bt()
            return self.next_point()

        self.n_probes += 1
        return x,self.bbox(x)

    def run(self):
        res = None
        while res is None:
            res = self.run_one(*self.next_point())
        return res

########################################################################################################################
#   Unit Tests
########################################################################################################################

# TODO migrate to unittest, how to deal with probabilistic failures?
if __name__ == '__main__':
    import time
    from libpushdf.basic_types import GF, LARGEST_u16_PRIME

    def blackbox(x):
        return x**24 + x*x - x

    R = PolynomialRing(GF(509), 'x')
    x = R.variables()[0]
    interp = UnivariateRacer(R, blackbox, deg_bnd=24)
    reference = interp.run()
    assert reference == x**24 + x*x - x
    interp.reset()

    n_probes = []

    t = time.time()
    for i in range(10000):
        print("===== NEW =====")
        r = interp.run()
        if r != reference:
            print("FAIL")
            print(r)
            print("===============")
            assert False
        n_probes.append(interp.n_probes)
        interp.reset()
    print(time.time() - t)

    print(f"Average #(probes) {sum(n_probes) / len(n_probes)}")
