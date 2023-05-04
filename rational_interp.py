#!/usr/bin/env python3
#
#   Multivariate Rational Function Interpolation
#

from typing import Callable

from libpushdf.basic_types import CRT, GF, Polynomial, PolynomialRing, Rational
from libpushdf.rational_sparse_interp import RationalSparseInterpolator
from libpushdf.thiele_interp import ThieleInterpolator
from libpushdf.zippel_interp import ZippelInterpolator

class RationalInterpolator:
    """
    Multivariate rational function black box interpolation
    """

    def __init__(self, ring : PolynomialRing, bbox : Callable, n_returns : int):
        self.ring : PolynomialRing = ring
        self.bbox : Callable = bbox
        self.n_returns : int = n_returns

        self.t_ring = PolynomialRing(self.ring.coeff_ring, 't')

        self.anchors_base = tuple([self.ring.coeff_one] + self.ring.coeff_ring.rand_elems(self.ring.n_vars - 1, 2, -1))
        self.anchors = None
        self.shifts = None

        self.evals = {}

        self.blackbox_t = lambda t : self.bbox(*(yi * t + si for yi,si in zip(self.anchors,self.shifts)))

    def do_one_thiele(self):
        return ThieleInterpolator(self.t_ring, self.blackbox_t, self.n_returns).run()

    def determine_shifts_degs(self):
        # Generate random non-zero anchors and shifts for all variables. Shifts such that f(shifts) is undefined are
        # unusable.
        self.anchors = self.anchors_base
        self.shifts = self.ring.coeff_ring.rand_elems(self.ring.n_vars, 2, -1)

        # Run first interpolation with all variables shifted to ascertain maximum degrees of numerator and denominator
        # with high probability.
        print("  all shifted")
        self.r_orig = self.do_one_thiele()

        # Require normalizing constant from shifts
        # (if this fails we could reroll the shifts and try again? can this even fail?)
        assert all(ri.dnm[0] != 0 or ri.num[0] != 0 for ri in self.r_orig)

        # Obtain degrees, we assume these are the correct degrees (with high probability this is true)
        degree_num = [ri.num.degree() for ri in self.r_orig]
        degree_dnm = [ri.dnm.degree() for ri in self.r_orig]
        # Now check if we need any shifts at all, compare with above which is assumed to be good
        self.shifts = [0] * self.ring.n_vars
        # For each function, introduce one shift at a time until we get the same degrees as before, but with fewer
        # shifts. This is worth spending time on as shifting all variables can massively increase the density of the
        # function.
        for i in range(self.ring.n_vars):
            if i > 0:
                # Add a single random shift per iteration
                self.shifts[i - 1] = self.ring.coeff_ring.rand_elem_distinct(2, self.shifts)

            print(f"  try {i} shifts")
            r = self.do_one_thiele()

            assert not any(ri.dnm[0] == 0 and ri.num[0] == 0 for ri in r) , "No normalization?"

            deg_num = [ri.num.degree() for ri in r]
            deg_dnm = [ri.dnm.degree() for ri in r]

            if all(a1 >= a2 for a1,a2 in zip(deg_dnm, degree_dnm)) and \
               all(a1 >= a2 for a1,a2 in zip(deg_num, degree_num)):
                self.r_orig = r
                break
        else:
            #assert False, f"Failed to find good shifts even with all variables shifted? {self.shifts} {r}"
            pass

        print("  pass")

        self.evals[self.anchors] = self.r_orig

        self.normalize_num = [ri.num[self.ring.mon0] == 1 for ri in self.r_orig]

        # factors in the normalization
        # TODO fill these in whenever an interpolation completes to reduce number of required queries
        self.num_powers = [[(mon.degrees[0],None if mon.degrees[0] != 0 or not ri.num[0] == 1 else 1)
                            for mon in ri.num.monomials] for ri in self.r_orig]
        self.dnm_powers = [[(mon.degrees[0],None if mon.degrees[0] != 0 or not ri.dnm[0] == 1 else 1)
                            for mon in ri.dnm.monomials] for ri in self.r_orig]
        self.extras = RationalSparseInterpolator(self.t_ring, self.blackbox_t, self.n_returns, self.num_powers,
                                                 self.dnm_powers)

    def compute_shifted(self, poly):
        vars = self.ring.variables()
        var_names = self.ring.var_names
        return poly.subst({ vn : v + shift for v,vn,shift in zip(vars,var_names,self.shifts) })

    def blackbox_runner(self, n):
        # print(self.anchors)

        # lookup values or obtain them from univariate interpolation if not done already
        if self.anchors not in self.evals:
            # print("  CACHE MISS")
            ev = self.extras.run()
            self.evals[self.anchors] = ev
            values = ev
        else:
            # print("  CACHE HIT")
            values = self.evals[self.anchors]

        t_poly = self.t_ring(values[n].num if self.numerator else values[n].dnm)

        sft = self.shift_diff.terms_of_degree(self.cur_monom.degree())(self.anchors)
        value = t_poly[self.cur_monom] - sft
        # print(f"{t_poly[cur_monom]} - {sft} = {value}")
        return value

    def run_one(self, monoms, bbox):
        self.shift_diff = self.ring(0)
        result = self.ring(0)

        for cm in monoms:
            self.cur_monom = cm
            # print(f"  CURRENT: {self.cur_monom}")
            # print(f"  DIFF: {self.shift_diff}")
            zintp = ZippelInterpolator(self.ring, bbox, self.cur_monom.degree(), self.anchors_base, True)
            while not zintp.next_var():
                while not zintp.step():
                    pass
            res = zintp.result()
            self.shift_diff += self.compute_shifted(res) - res
            result += res
        return result

    def run(self):
        print("shifts/degs")
        self.determine_shifts_degs()
        print("the rest")

        self.shift_diff = None
        self.cur_monom = None

        results = []

        # TODO also start on lowest degree terms (Hybrid Racer)

        for n in range(self.n_returns):
            # print(n)

            def bbox(*_anchors):
                self.anchors = _anchors
                return self.blackbox_runner(n)

            # print("=== NUM ===")
            self.numerator = True
            result_num = self.run_one(self.r_orig[n].num.monomials, bbox)
            # print(" ", result_num)

            # print("=== DNM ===")
            self.numerator = False
            result_dnm = self.run_one(self.r_orig[n].dnm.monomials, bbox)
            # print(" ", result_dnm)

            # fix result
            result = Rational(result_num, result_dnm)

            if self.normalize_num[n]:
                # Canonicalise with constant 1 in numerator
                num = result.num.coeffs[-1]
                result.num *= ~num
                result.dnm *= ~num
            else:
                # Canonicalise with constant 1 in denominator
                dnm = result.dnm.coeffs[-1]
                result.num *= ~dnm
                result.dnm *= ~dnm

            results.append(result)
        return results

def reconstruct_rational_function(blackbox, ring_changed, n_returns, primes, var_names=None):
    any_failed = True
    prev_polys_nums = None
    prev_polys_dnms = None

    for p in primes:
        print(p)
        CR = GF(p)

        if not any_failed:
            # all done, TODO check agreement over new prime
            break

        R = PolynomialRing(CR, var_names or blackbox.__code__.co_varnames)

        if ring_changed is not None:
            ring_changed(R)

        # TODO allow feeding completed coefficients
        results = RationalInterpolator(R, blackbox, n_returns).run()
        # print(results)

        any_failed = False

        for i,result in enumerate(results):
            # print(f"\nRUN {result}")

            def innerloop(poly, prev):
                nonlocal any_failed

                # if we have a previous run, CRT with it
                if prev is not None:
                    poly = poly.poly_crt(prev[i])

                # print(f"{poly.ring.coeff_ring}")

                # try and promote coefficients of this polynomial to Q
                reconst = poly.poly_rat_rec()

                if not isinstance(reconst, Polynomial):
                    coeffs, monomials = reconst
                    print(f"fail {coeffs}")
                    # failed to promote all coefficients
                    # TODO some may have succeeded, exclude them from future interpolations
                    any_failed = True
                    # return the polynomial from this run
                    return poly
                else:
                    # success
                    # TODO this whole polynomial succeeded, exclude it from future interpolations
                    # print(reconst)
                    return reconst

            if not result.num.ring.coeff_ring.is_rational():
                results[i].num = innerloop(result.num, prev_polys_nums)
            if not result.dnm.ring.coeff_ring.is_rational():
                results[i].dnm = innerloop(result.dnm, prev_polys_dnms)

        prev_polys_nums = [result.num for result in results]
        prev_polys_dnms = [result.dnm for result in results]
    else:
        assert False , "Ran out of primes"

    return results

########################################################################################################################
#   Unit Tests
########################################################################################################################

# TODO migrate to unittest, how to deal with probabilistic failures?
if __name__ == '__main__':
    import time
    from libpushdf.primes_16bit import primes_16bit

    def blackbox(a_1, a_2, a_3, a_4, a_5):
        print(f"    BBOX CALL ({a_1}, {a_2}, {a_3}, {a_4}, {a_5})")
        return [
            #0,
            #1 / (a_1 * a_3),
            #-1 / (a_2 * a_3)
            -1 / (a_1 * a_2),
             1 / (a_1 * a_5),
            -1 / (a_2 * a_3),
            -1 / (a_3 * a_4),
            -1 / (a_4 * a_5),
            #(Rational(3,4) * x + 7 * y) / (4 * x * y + x * Rational(13,2) + y),
            #(x - x * y),
            #(x * y - x * Rational(15,2) - y) / (3 * x + y)
        ]

    results = reconstruct_rational_function(blackbox, None, 5, list(reversed(primes_16bit)))
    for result in results:
        print(result)
