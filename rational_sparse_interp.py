#!/usr/bin/env python3

from typing import Callable

from libpushdf.basic_types import Monomial, Polynomial, PolynomialRing, Rational
from libpushdf.linalg import Matrix

class RationalSparseInterpolator:
    """
    Rational function interpolation via a system of equations solved with Gaussian elimination
    """

    def __init__(self, ring : PolynomialRing, blackbox : Callable, n_returns : int, num_powers : list,
                 dnm_powers : list, var_nums=None, bbox_cache : dict = None):
        self.ring = ring

        assert self.ring.n_vars == 1 , "Only Univariate is supported"

        self.bbox = blackbox
        self.bbox_cache = bbox_cache or {} # needs to be here otherwise all instances share the same dict
        self.n_returns = n_returns
        self.var_nums = var_nums or list(range(self.ring.n_vars))

        self.num_known = [0] * self.n_returns
        self.num_unknown_powers = [0] * self.n_returns
        self.num_known_part = [0] * self.n_returns
        self.num_unknown_monoms = [0] * self.n_returns

        self.dnm_known = [0] * self.n_returns
        self.dnm_unknown_powers = [0] * self.n_returns
        self.dnm_known_part = [0] * self.n_returns
        self.dnm_unknown_monoms = [0] * self.n_returns

        for i in range(self.n_returns):
            self.num_known[i] = [(p,c) for p,c in num_powers[i] if c is not None]
            self.num_unknown_powers[i] = [p for p,c in num_powers[i] if c is None]
            self.num_known_part[i] = Polynomial(self.ring, [Monomial(self.ring, (p,)) for p,_ in self.num_known[i]],
                                                [c for _,c in self.num_known[i]])
            self.num_unknown_monoms[i] = [Monomial(self.ring, (p,)) for p in self.num_unknown_powers[i]]

            self.dnm_known[i] = [(p,c) for p,c in dnm_powers[i] if c is not None]
            self.dnm_unknown_powers[i] = [p for p,c in dnm_powers[i] if c is None]
            self.dnm_known_part[i] = Polynomial(self.ring, [Monomial(self.ring, (p,)) for p,_ in self.dnm_known[i]],
                                                [c for _,c in self.dnm_known[i]])
            self.dnm_unknown_monoms[i] = [Monomial(self.ring, (p,)) for p in self.dnm_unknown_powers[i]]

    def run(self):
        # we need this many distinct evaluations to solve the system
        num_eqns = [len(self.num_unknown_powers[i]) + len(self.dnm_unknown_powers[i]) for i in range(self.n_returns)]

        # Obtain values for the interpolator, as many as we have equations
        # They must be defined for blackbox_t or we'll need to reroll for the ones that are not
        zeta = []
        evals = []

        while len(zeta) != max(num_eqns):
            zi = self.ring.coeff_ring.rand_elem(2)
            if zi in zeta:
                # not distinct
                continue
            if zi in self.bbox_cache:
                # cache hit
                fi = self.bbox_cache[zi]
            else:
                # cache miss, run the function proper
                try:
                    fi = self.bbox(zi)
                except KeyboardInterrupt as e:
                    # user abort
                    raise e
                except:
                    print(f"EVAL FAIL {zi}")
                    continue
                # update cache
                self.bbox_cache[zi] = fi
            zeta.append(zi)
            evals.append(fi)

        rets = [0] * self.n_returns

        for i in range(self.n_returns):
            num_eqn = num_eqns[i]
            num_unknown_powers = self.num_unknown_powers[i]
            dnm_unknown_powers = self.dnm_unknown_powers[i]
            dnm_known = self.dnm_known[i]
            num_known = self.num_known[i]
            num_unknown_monoms = self.num_unknown_monoms[i]
            dnm_unknown_monoms = self.dnm_unknown_monoms[i]
            num_known_part = self.num_known_part[i]
            dnm_known_part = self.dnm_known_part[i]

            # Build the linear system
            dnm_coeffs_start = len(num_unknown_powers)

            def get_mtx_item(k):
                row,col = divmod(k, num_eqn + 1)

                zz = zeta[row]

                if col < dnm_coeffs_start:
                    # Numerator contribution
                    return zz ** num_unknown_powers[col]
                elif col < num_eqn:
                    # Denominator contribution
                    return -evals[row][i] * zz ** dnm_unknown_powers[col - dnm_coeffs_start]
                else:
                    # Solved coefficients contribution
                    return evals[row][i] * sum(dnm_coeff * zz ** dnm_power for dnm_power,dnm_coeff in dnm_known) \
                                         - sum(num_coeff * zz ** num_power for num_power,num_coeff in num_known)

            # Solve the system of equations via Gauss-Jordan
            mtx = Matrix(num_eqn, num_eqn + 1, [get_mtx_item(k) for k in range(num_eqn * (num_eqn + 1))])
            ans = mtx.reduced_row_echelon_form().col(mtx.cols - 1)

            # Construct polynomials
            num_poly = Polynomial(self.ring, num_unknown_monoms, ans[:dnm_coeffs_start])
            dnm_poly = Polynomial(self.ring, dnm_unknown_monoms, ans[dnm_coeffs_start:])

            rets[i] = Rational(num_poly + num_known_part, dnm_poly + dnm_known_part)

        return rets
