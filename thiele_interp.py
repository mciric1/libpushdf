#!/usr/bin/env python3

from libpushdf.basic_types import Monomial, Polynomial, PolynomialRing, Rational

class ThieleInterpolator:
    """
    Univariate rational function interpolation via Thiele's continued fraction
    """

    def __init__(self, ring : PolynomialRing, bbox, n_returns=1, bbox_cache=None):
        self.ring : PolynomialRing = ring
        self.bbox = bbox
        self.mon0 = Monomial(self.ring, (0,))
        self.mon1 = Monomial(self.ring, (1,))
        self.poly1 = self.ring(1)
        self.n_returns = n_returns
        self.bbox_cache = bbox_cache or {} # has to be here or all instances of the class will share the same dict...
        self.reset()

    def reset(self):
        self.t = []
        self.f = []
        self.coeffs = [[] for _ in range(self.n_returns)]
        self.done = [False] * self.n_returns
        self.n_probes = 0

    def compute_coeff(self, i, j, k):
        if i == j and i < len(self.coeffs[k]):
            # already computed, a full cache for i,j is unnecessary
            return self.coeffs[k][i]

        if j == 0:
            coeff = self.f[i][k]
        else:
            c1 = self.compute_coeff(i,     j - 1, k)
            c2 = self.compute_coeff(j - 1, j - 1, k)

            if self.done[k]:
                return None

            dnm = c1 - c2
            if dnm == 0:
                coeff = None
                self.done[k] = True
            else:
                coeff = (self.t[i] - self.t[j - 1]) / dnm

        return coeff

    def next_point(self):
        # Obtain a valid evaluation
        while True:
            t = self.ring.coeff_ring.rand_elem()

            if t in self.bbox_cache:
                f = self.bbox_cache[t]
            else:
                self.n_probes += 1
                try:
                    f = self.bbox(t)
                except KeyboardInterrupt as e:
                    # user abort
                    raise e
                except ZeroDivisionError:
                    print(f"f({t}) EVAL FAIL")
                    continue
                assert len(f) == self.n_returns , "Number of returns different to expected."
                if t not in self.bbox_cache:
                    self.bbox_cache[t] = f
                if not t in self.t:
                    break
        return t,f

    def add_point(self, t, ft):
        self.t.append(t)
        self.f.append(ft)

        i = len(self.t) - 1

        for k in range(self.n_returns):
            if self.done[k]:
                continue

            coeff = self.compute_coeff(i, i, k)

            assert coeff is not None or self.done[k]

            if self.done[k]:
                self.coeffs[k].append(self.ring.coeff_ring(1))
            else: # Continue
                self.coeffs[k].append(coeff)

    def run_one(self, tft=None):
        """
        Runs a single Thiele Interpolation with chance of failure
        """
        if tft is not None:
            self.add_point(*tft)

        while not all(d for d in self.done):
            self.add_point(*self.next_point())

    def result(self):
        rets = []

        for c in self.coeffs:
            t = self.t[:len(c)-1]

            # build the function, automatically gcd-reduced
            assert len(c) >= 2

            if len(c) == 2:
                ret = Rational(self.ring(0), self.poly1)
            else:
                ret = Rational(Polynomial(self.ring, [self.mon1, self.mon0], [1, -t[-2]]), c[-2])

                for x,ci in reversed(list(zip(t[:-2], c[1:-1]))):
                    ret = Rational(self.ring(ci), self.poly1) + ret
                    ret = Rational(Polynomial(self.ring, [self.mon1, self.mon0], [1, -x]), self.poly1) / ret

            ret = Rational(self.ring(c[0]), self.poly1) + ret

            if type(ret.num) != Polynomial:
                ret.num = self.ring(ret.num)
            if type(ret.dnm) != Polynomial:
                ret.dnm = self.ring(ret.dnm)

            if ret.dnm[0] != 0:
                # Canonicalise with constant 1 in denominator
                dnm = ret.dnm[0]
                ret.num *= ~dnm
                ret.dnm *= ~dnm
            elif ret.num[0] != 0:
                # Canonicalise with constant 1 in numerator
                num = ret.num[0]
                ret.num *= ~num
                ret.dnm *= ~num
            else:
                assert False , "Not in canonical form?"

            rets.append(ret)

        return rets

    def run(self):
        res = None

        # Perform initial interpolation
        self.run_one()

        while True:
            # Check with another point incase the interpolation failed by terminating too early.
            x,f = self.next_point()
            res = self.result()
            try:
                if all(resi(x) == fi for resi,fi in zip(res,f)):
                    break
            except:
                pass

            print("Failure, continue")

            n_probes = self.n_probes
            self.reset()
            self.n_probes = n_probes
            # Continue interpolation if terminated too early, reusing previous point
            self.run_one((x,f))

        return res
