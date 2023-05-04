#!/usr/bin/env python3

import operator
import random
from functools import reduce
from time import time
from typing import List, Union

def isiterable(x):
    return isinstance(x, (tuple, list))

def prod(l):
    return reduce(operator.mul, l)

########################################################################################################################
#   Modular Arithmetic
########################################################################################################################

def gcd(a, b):
    while b != 0:
        a %= b
        a,b = b,a
    return abs(a)

def xgcd(a, b):
    prevx, x = 1, 0
    prevy, y = 0, 1
    while b != 0:
        q, r = divmod(a, b)
        x, prevx = prevx - q * x, x
        y, prevy = prevy - q * y, y
        a, b = b, r
    return a, prevx, prevy

def CRT(a, b):
    # NOTE this fails with ZeroDivisionError if there is no solution
    x1, p1 = a.x, a.p
    x2, p2 = b.x, b.p
    return Mod(x1 * (~Mod(p2, p1)).x * p2 + x2 * (~Mod(p1, p2)).x * p1, p1 * p2)

# Various useful primes
LARGEST_u32_PRIME = 4294967295
LARGEST_s32_PRIME = 2147483647
LARGEST_u16_PRIME = 65521
LARGEST_s16_PRIME = 32749

class Mod:
    """
    Arithmetic in GF(p)
    """

    def __init__(self, x : int, p : int):
        self.x = x
        self.p = p

        if self.x not in range(self.p):
            # print(f"Out of range {self.x} {self.p}")
            self.x %= self.p

        # Init for prime if not done
        if self.p not in Mod.__invert__.cache:
            Mod.__invert__.cache[self.p] = {}

    def __str__(self):
        return str(self.x)

    def __repr__(self):
        return f"Mod({self.x}, {self.p})"

    def __hash__(self):
        return hash((self.x, self.p))

    def cvt_other(self, other):
        if isinstance(other, int):
            other = Mod(other, self.p)
        elif isinstance(other, Rational):
            other = other.to_mod(self.p)
        return other

    def __add__(self, other):
        other = self.cvt_other(other)
        assert self.p == other.p
        r = self.x + other.x
        if r >= self.p:
            r -= self.p
        return Mod(r, self.p)

    def __radd__(self, other):
        other = self.cvt_other(other)
        assert self.p == other.p
        r = self.x + other.x
        if r >= self.p:
            r -= self.p
        return Mod(r, self.p)

    def __sub__(self, other):
        other = self.cvt_other(other)
        assert self.p == other.p
        r = self.x - other.x
        if r < 0:
            r += self.p
        return Mod(r, self.p)

    def __rsub__(self, other):
        other = self.cvt_other(other)
        assert self.p == other.p
        r = other.x - self.x
        if r < 0:
            r += other.p
        return Mod(r, other.p)

    def __abs__(self):
        # The only abs() on a finite field is the trivial one
        return self

    def __mul__(self, other):
        other = self.cvt_other(other)
        assert self.p == other.p
        return Mod(self.x * other.x, self.p)

    def __rmul__(self, other):
        other = self.cvt_other(other)
        assert self.p == other.p
        return Mod(self.x * other.x, self.p)

    def __pow__(self, other):
        assert isinstance(other, int)
        r = pow(self.x, other, self.p)
        return Mod(r, self.p)

    def __invert__(self):
        """
        Multiplicative inverse
        """
        if self.x == 0:
            raise ZeroDivisionError

        if self.x in Mod.__invert__.cache[self.p]:
            # Cache hit
            return Mod.__invert__.cache[self.p][self.x]
        else:
            # Cache miss, calculate
            a,x,_ = xgcd(self.x, self.p)
            # assert a == 1, 'x must be coprime to p' # always true for p prime
            if x < 0:
                x += self.p
            result = Mod(x, self.p)
            # Add to cache and return
            Mod.__invert__.cache[self.p][self.x] = result
            return result

    def __neg__(self):
        """
        Additive inverse
        """
        r = self.p - self.x
        return Mod(r, self.p)

    def __truediv__(self, other):
        other = self.cvt_other(other)
        assert self.p == other.p
        return self * ~other

    def __rtruediv__(self, other):
        other = self.cvt_other(other)
        return other / self

    def __eq__(self, other):
        if isinstance(other, Mod):
            # Same field
            assert self.p == other.p
            return self.x == other.x
        elif isinstance(other, int):
            # Test equality mod p
            return self.x == other % self.p
        elif isinstance(other, Rational):
            return self.x == other.to_mod(self.p).x
        elif other == None:
            return False
        else:
            assert False , f"Comparing {type(self)} and {type(other)}"

    def __lt__(self, other):
        return self.x < other.x

    def __gt__(self, other):
        if isinstance(other, int):
            other = Mod(other, self.p)
        return self.x > other.x

    def rational_reconstruction(self):
        """
        Attempts to find an (x = r / t) such that
            n = x mod p

        If such an x is found, it is unique.
        It will succeed if |r| and |t| <= sqrt(p/2)

        Inputs:
        x : Integer 0 <= x < p (member of GF(p))
        p : Modulus
        """
        r, old_r = self.x, self.p
        t, old_t = 1, 0

        while 2 * r**2 >= self.p:
            quot = old_r // r
            old_r, r = r, old_r - quot * r
            old_t, t = t, old_t - quot * t

        if 2 * t**2 >= self.p or gcd(r, t) != 1:
            raise Exception("Rational Reconstruction Failed")

        return Rational(r, t)

Mod.__invert__.cache = {}

########################################################################################################################
#   Rational Numbers
########################################################################################################################

class Rational:
    def __init__(self, num, dnm):
        self.num = num
        self.dnm = dnm
        self.canonicalise()

    def tup(self):
        return self.num, self.dnm

    def __str__(self):
        if self.dnm == 0 or self.dnm == 1:
            return f"{self.num}"
        return f"\\frac{{{self.num}}}{{{self.dnm}}}"

    def __repr__(self):
        return f"Rational({self.num}, {self.dnm})"

    def canonicalise(self):
        # For consistency, require denominator 1 when numerator is 0
        if self.num == 0:
            self.dnm = self.dnm.ring(1) if type(self.dnm) == Polynomial else 1
        # Check div0, denominator can be 0 only when numerator is 0
        if self.num != 0 and self.dnm == 0:
            raise ZeroDivisionError
        # If denominator is 1 we're done
        if self.dnm == 1:
            return
        if type(self.dnm) == int:
            # Move sign out of the denominator
            if self.dnm < 0:
                self.dnm = -self.dnm
                self.num = -self.num
        if self.num != 0:
            # Remove common factors
            if type(self.num) == int and type(self.dnm) == int: # TODO polynomial gcd?
                g = gcd(self.num, self.dnm)
                self.num //= g
                self.dnm //= g

    def cvt_other(self, other):
        if isinstance(other, int):
            other = Rational(other, 1)
        return other

    def __add__(self, other):
        other = self.cvt_other(other)
        return Rational(self.num * other.dnm + self.dnm * other.num, self.dnm * other.dnm)

    def __radd__(self, other):
        other = self.cvt_other(other)
        return Rational(self.num * other.dnm + self.dnm * other.num, self.dnm * other.dnm)

    def __sub__(self, other):
        other = self.cvt_other(other)
        return Rational(self.num * other.dnm - self.dnm * other.num, self.dnm * other.dnm)

    def __rsub__(self, other):
        other = self.cvt_other(other)
        return Rational(self.dnm * other.num - self.num * other.dnm, self.dnm * other.dnm)

    def __mul__(self, other):
        other = self.cvt_other(other)
        if isinstance(other, Mod):
            return other * self
        return Rational(self.num * other.num, self.dnm * other.dnm)

    def __rmul__(self, other):
        other = self.cvt_other(other)
        return Rational(self.num * other.num, self.dnm * other.dnm)

    def __truediv__(self, other):
        other = self.cvt_other(other)
        return Rational(self.num * other.dnm, self.dnm * other.num)

    def __rtruediv__(self, other):
        other = self.cvt_other(other)
        return Rational(other.num * self.dnm, other.dnm * self.num)

    def __pow__(self, other):
        assert isinstance(other, int)
        return Rational(self.num ** other, self.dnm ** other)

    def __invert__(self):
        return Rational(self.dnm, self.num)

    def __neg__(self):
        return Rational(-self.num, self.dnm)

    def __abs__(self):
        return Rational(abs(self.num), self.dnm)

    def cmp(self, other, op):
        other = self.cvt_other(other)
        assert isinstance(other, Rational) , f"Comparing {type(self)} and {type(other)}"

        return op(self.num * other.dnm, other.num * self.dnm)

    def __eq__(self, other):
        if isinstance(other, int) and other == 0 and self.num == 0:
            return True
        other = self.cvt_other(other)
        return self.num == other.num and self.dnm == other.dnm

    def __lt__(self, other):
        return self.cmp(other, lambda x,y : x < y)

    def __gt__(self, other):
        return self.cmp(other, lambda x,y : x > y)

    def __ge__(self, other):
        return self.cmp(other, lambda x,y : x >= y)

    def __call__(self, x):
        num_eval = self.num(x) if callable(self.num) else self.num
        dnm_eval = self.dnm(x) if callable(self.dnm) else self.dnm
        return num_eval / dnm_eval

    def to_mod(self, p):
        return Mod(self.num, p) * ~Mod(self.dnm, p)

########################################################################################################################
#   Coefficient Rings
########################################################################################################################

class CoefficientRing:
    def rand_elem(self, min : int = 0):
        raise NotImplementedError()

    def rand_elems(self, num : int, min : int = 0, max : int = 0):
        raise NotImplementedError()

    def rand_elem_distinct(self, min : int = 0, existing : list =[]):
        raise NotImplementedError()

    def is_rational(self):
        return isinstance(self, RationalField)

class RationalField(CoefficientRing):
    def __call__(self, arg : Union[Rational, int]):
        if isinstance(arg, Rational):
            return arg
        elif isinstance(arg, int):
            return Rational(arg, 1)
        else:
            raise ValueError(f"{arg} cannot be a member of a rational field")

    def __str__(self):
        return "The Rational Numbers"

    def rand_elem(self):
        # Bounds are arbitrary for testing purposes
        return Rational(random.randint(0, 100), random.randint(1, 100))

    def rand_elems(self, num):
        # Bounds are arbitrary for testing purposes
        samp1 = random.sample(range(0, 100), num)
        samp2 = random.sample(range(1, 100), num)
        return [Rational(n, d) for n,d in zip(samp1,samp2)]

QQ = RationalField()

class GF(CoefficientRing):
    def __init__(self, p : int):
        assert p > 0
        self.p = p

    def __repr__(self):
        return f"GF({self.p})"

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        if isinstance(other, GF):
            return self.p == other.p
        return False

    def __call__(self, arg : Union[Mod, int]):
        if isinstance(arg, Mod):
            return arg if arg.p == self.p else Mod(arg.x, min(self.p, arg.p))
        elif isinstance(arg, int):
            return Mod(arg, self.p)
        elif isinstance(arg, Rational):
            return Mod(arg.num, self.p) / Mod(arg.dnm, self.p)
        else:
            raise ValueError(f"{arg} cannot be a member of a prime field")

    def rand_elem(self, min : int = 0):
        return Mod(random.randint(min, self.p - 1), self.p)

    def rand_elems(self, num : int, min : int = 0, max : int = 0):
        samp = random.sample(range(min, self.p + max), num)
        return [Mod(x, self.p) for x in samp]

    def rand_elem_distinct(self, min : int = 0, existing : list = []):
        # Obtains a random element not in `existing`
        r = self.rand_elem(min)
        while r in existing:
            r = self.rand_elem(min)
        return r

########################################################################################################################
#   Polynomial Rings
########################################################################################################################

def monomial_cmp_lex(m1, m2):
    for m1d, m2d in zip(m1, m2):
        if m1d > m2d:
            return True
        if m2d > m1d:
            return False
    return False

MonomialOrderLex = monomial_cmp_lex

def monomial_cmp_grlex(m1, m2):
    # first compare total degree
    m1d = sum(m1)
    m2d = sum(m2)
    if m1d > m2d:
        return True
    if m2d > m1d:
        return False
    # break ties with lex order
    return monomial_cmp_lex(m1, m2)

MonomialOrderGrLex = monomial_cmp_grlex

def monomial_cmp_grevlex(m1, m2):
    # first compare total degree
    m1d = sum(m1)
    m2d = sum(m2)
    if m1d > m2d:
        return True
    if m2d > m1d:
        return False

    # break ties with reverse lex order
    for m1d, m2d in reversed(list(zip(m1, m2))):
        if m1d < m2d:
            return True
        if m2d < m1d:
            return False
    return True

MonomialOrderGRevLex = monomial_cmp_grevlex

class PolynomialRing:
    def __init__(self, coeff_ring : CoefficientRing, var_names : List[str], deriv_var_names : List[str] = [],
                 deriv_vars_end=0, order=MonomialOrderGRevLex):
        self.coeff_ring = coeff_ring
        self.var_names = list(var_names)
        self.n_vars = len(self.var_names)
        self.deriv_var_names = list(deriv_var_names)
        self.n_deriv_vars = len(self.deriv_var_names)
        self.deriv_vars_end = deriv_vars_end
        assert len(set(self.var_names)) == self.n_vars , "Variable names must be distinct"
        assert len(set(self.deriv_var_names)) == self.n_deriv_vars , "Derivative names must be distinct"
        self.coeff_zero = self.coeff_ring(0)
        self.coeff_one = self.coeff_ring(1)
        self.mon0 = Monomial(self, (0,) * self.n_vars)
        self.order = order
        self.variables_cached = None

    def extend_with_derivatives(self, vars):
        # creates from this Polynomial Ring, a Ring containing the original Ring as a subring, plus basis elements
        # corresponding to derivatiives w.r.t. `vars`

        # order derivatives first for lex ordering
        var_names = [f"[d{vn}/d{dv}]" for dv in vars for vn in self.var_names]
        var_names += self.var_names

        return PolynomialRing(self.coeff_ring, var_names, deriv_var_names=vars,
                              deriv_vars_end=len(self.var_names) * len(vars), order=self.order)

    def to_coeff_ring(self, coeff_ring : CoefficientRing):
        return PolynomialRing(coeff_ring, self.var_names, self.deriv_var_names, self.order)

    def __call__(self, element):
        if isinstance(element, Polynomial):
            if element.ring is self:
                return element
            elif element.ring is self.coeff_ring:
                return Polynomial(self, [self.mon0], [self.coeff_ring(element)])
            else:
                return Polynomial(self, [self(mon) for mon in element.monomials],
                                        [self.coeff_ring(coeff) for coeff in element.coeffs])
        elif isinstance(element, Monomial):
            if element.ring is self:
                return Polynomial(self, [element], [self.coeff_ring(1)])
            elif element.ring is self.coeff_ring:
                return Polynomial(self, [self.mon0], [self.coeff_ring(element)])
            else:
                new_degrees = [0] * self.n_vars
                for i,vn1 in enumerate(self.var_names):
                    for j,vn2 in enumerate(element.ring.var_names):
                        if vn1 == vn2:
                            new_degrees[i] = element.degrees[j]
                            break
                return Monomial(self, new_degrees)
        elif isinstance(element, Mod) or isinstance(element, Rational) or isinstance(element, int):
            return Polynomial(self, [self.mon0], [self.coeff_ring(element)])
        else:
            assert False , f"{element} cannot be part of this polynomial ring"

    def __eq__(self, other):
        if type(other) != PolynomialRing:
            return False
        return self.coeff_ring == other.coeff_ring and self.var_names == other.var_names and self.order == other.order

    def variables(self):
        # If already computed
        if self.variables_cached is not None:
            return self.variables_cached

        # Compute them
        variables = []
        for i in range(len(self.var_names)):
            degrees = [0] * self.n_vars
            degrees[i] = 1
            variables.append(Polynomial(self, [Monomial(self, degrees)], [1]))
        self.variables_cached = tuple(variables)
        return self.variables_cached

    def __str__(self):
        return f"Polynomial Ring in {self.n_vars} variable(s) {self.var_names} over {self.coeff_ring}"

    def __repr__(self) -> str:
        return f"PolynomialRing({repr(self.coeff_ring)}, {repr(self.var_names)})"

def poly_can_be_coeff(r1, r2):
    """
    Can elements of r1 source coefficients from r2
    """
    return type(r1.coeff_ring) is PolynomialRing and r1.coeff_ring == r2

########################################################################################################################
#   Monomial
########################################################################################################################

class Monomial:
    def __init__(self, ring : PolynomialRing, degrees : tuple):
        degrees = tuple(degrees)
        assert all(deg >= 0 for deg in degrees) , f"Degrees should be nonnegative, got {degrees}"
        assert ring.n_vars == len(degrees) , "Degrees should match number of variables"
        self.degrees = degrees
        self.ring = ring

    def __hash__(self):
        return hash(self.degrees)

    def __call__(self, x : tuple):
        assert len(x) == len(self.degrees)
        return prod(xi ** d for xi,d in zip(x,self.degrees))

    def eval_some(self, eval_map):
        degrees = [0] * len(self.degrees)
        coeff = 1
        for i,d in enumerate(self.degrees):
            if i in eval_map:
                # evaluate this variable
                coeff *= self.ring.coeff_ring(eval_map[i]) ** d
            else:
                # no evaluation to do
                degrees[i] = d
        return Polynomial(self.ring, [Monomial(self.ring, degrees)], [coeff])

    def subst(self, subst_map):
        """
        Substitute variables for values in subst_map
        """
        poly = self.ring(1)
        vars = self.ring.variables()
        for d,var,name in zip(self.degrees,vars,self.ring.var_names):
            if name in subst_map:
                # substitute this variable
                poly *= subst_map[name] ** d
            elif d != 0:
                # no evaluation to do
                poly *= var ** d

        return poly

    def __mul__(self, other):
        if not isinstance(other, Monomial):
            return Polynomial(self.ring, [self], [other])

        # For polynomials with coefficients in another polynomial ring
        if poly_can_be_coeff(self.ring, other.ring):
            return Polynomial(self.ring, [self], [other])
        if poly_can_be_coeff(other.ring, self.ring):
            return Polynomial(other.ring, [other], [self])

        assert self.ring == other.ring , "Polynomial rings should match"
        degrees = (a + b for a,b in zip(self.degrees, other.degrees))
        return Monomial(self.ring, degrees)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"Monomial({repr(self.ring)}, {repr(self.degrees)})"

    def __str__(self):
        if all(deg == 0 for deg in self.degrees):
            return "1"
        return " ".join([f"{name}^{{{degree}}}" if degree != 1 else f"{name}" \
            for name,degree in zip(self.ring.var_names, self.degrees) if degree != 0])

    def __truediv__(self, other):
        x = [self.degrees[i] - other.degrees[i] for i in range(len(self.degrees))]
        if all(power >= 0 for power in x):
            return Monomial(self.ring, x)
        else:
            raise ArithmeticError("Monomials with negative exponents are not members of the ring")

    def __getitem__(self, k : int):
        return self.degrees[k]

    def degree(self):
        return sum(self.degrees)

    def __eq__(self, other):
        #assert self.ring == other.ring
        return self.degrees == other.degrees

    def __gt__(self, other):
        return self.ring.order(other.degrees, self.degrees)

    def divisible_by(self, other):
        return all((a - b >= 0 for a,b in zip(self.degrees, other.degrees)))

    def gcd(self, other):
        return Monomial(self.ring, (min(i, j) for i,j in zip(self.degrees, other.degrees)))

    def lcm(self, other):
        return (self * other) / self.gcd(other)

    def differentiate(self, deriv_ring : PolynomialRing, a : str):
        # if we are the '1' monomial
        if all(deg == 0 for deg in self.degrees):
            return deriv_ring(0)

        # otherwise compute derivative term, the derivative of a monomial is in general a polynomial

        if a in deriv_ring.deriv_var_names:
            # total derivative

            var_diff = deriv_ring.n_vars - deriv_ring.deriv_vars_end
            da = deriv_ring.deriv_var_names.index(a) * var_diff - deriv_ring.deriv_vars_end

            monomials = []
            coeffs = []
            for i in range(deriv_ring.deriv_vars_end, deriv_ring.n_vars):
                if self.degrees[i] == 0:
                    # This monomial vanishes under differentiation
                    continue

                new_degs = [d for d in self.degrees]
                new_degs[i] -= 1 # Reduce power by 1
                assert new_degs[da + i] == 0 # Only support first derivatives
                new_degs[da + i] = 1 # Add derivative variable

                monomials.append(Monomial(deriv_ring, new_degs))
                coeffs.append(self.degrees[i])

            p = Polynomial(deriv_ring, monomials, coeffs)
        elif a in deriv_ring.var_names:
            # partial derivative with respect to a single variable

            ia = deriv_ring.var_names.index(a)

            prev_deg = self.degrees[ia]
            if prev_deg == 0:
                p = deriv_ring(0)
            else:
                p = Polynomial(deriv_ring,
                               [Monomial(deriv_ring, (d - 1 if i == ia else d for i,d in enumerate(self.degrees)))],
                               [deriv_ring.coeff_ring(prev_deg)])
        else:
            p = deriv_ring(0)
        return p

########################################################################################################################
#   Polynomial
########################################################################################################################

class Polynomial:
    def __init__(self, ring : PolynomialRing, monomials : List[Monomial], coeffs):
        # assert all(mon.ring == ring for mon in monomials)
        assert len(monomials) == len(coeffs), "Number of coefficients and monomials must match"
        assert all(isinstance(monomial, Monomial) for monomial in monomials), "monomials must be of Monomial type"
        self.ring = ring
        self.monomials = monomials
        # Promote to member of coefficient ring
        self.coeffs = [self.ring.coeff_ring(coeff) for coeff in coeffs]
        # Strip terms with coefficient 0
        zero = self.ring.coeff_zero

        self.monomials = [self.monomials[i] for i in range(len(self.monomials)) if self.coeffs[i] != zero]
        self.coeffs = [coeff for coeff in self.coeffs if coeff != zero]

        # Sort according to monomial ordering
        sorted_poly = list(sorted([T for T in zip(self.monomials, self.coeffs)], key=lambda T: T[0]))
        self.monomials = [m for m,c in sorted_poly]
        self.coeffs = [c for m,c in sorted_poly]

        assert all(self.monomials[i] < self.monomials[i+1] for i in range(len(self.monomials)-1)), \
            "Monomials must be distinct and ordered according to the relevant monomial order"

    def __hash__(self):
        return hash(tuple(self.coeffs + self.monomials))

    @staticmethod
    def ZERO(ring):
        return Polynomial(ring, [], [])

    def univariate_in(self, d):
        """
        Turns this polynomial into a univariate polynomial in `d`, the coefficients are promoted to elements of a
        polynomial ring containing the remaining variables.
        """
        coeff_ring = PolynomialRing(self.ring.coeff_ring, [vn for i,vn in enumerate(self.ring.var_names) if i != d])
        NEW_R = PolynomialRing(coeff_ring, self.ring.var_names[d])
        monomials = {}
        coeffs = {}
        for coeff,monomial in zip(self.coeffs,self.monomials):
            deg = monomial.degrees[d]
            monomials[deg] = Monomial(NEW_R, (deg,))
            rem_monomial = Monomial(coeff_ring, [deg for i,deg in enumerate(monomial.degrees) if i != d])

            p = Polynomial(coeff_ring, [rem_monomial], [coeff])
            if coeffs not in coeffs:
                coeffs[coeffs] = p
            else:
                coeffs[coeffs] += p

        monomials = [monomial for deg,monomial in reversed(sorted(monomials.items(), key=lambda i : i[0]))]
        coeffs = [coeff for deg,coeff in reversed(sorted(coeffs.items(), key=lambda i : i[0]))]

        return Polynomial(NEW_R, monomials, coeffs)

    def __str__(self):
        terms = []
        for coeff,monomial in zip(self.coeffs,self.monomials):
            if coeff == self.ring.coeff_zero:
                continue
            if coeff == self.ring.coeff_ring(1) and monomial.degree() != 0:
                terms.append(f"{monomial}")
            elif coeff == self.ring.coeff_ring(-1) and monomial.degree() != 0:
                terms.append(f"-{monomial}")
            elif monomial.degree() == 0:
                terms.append(f"{coeff}")
            else:
                coeff_str = f"{coeff}"
                if " " in coeff_str:
                    coeff_str = f"({coeff_str})"
                terms.append(f"{coeff_str} {monomial}")

        if len(terms) == 0:
            return "0"

        return " + ".join(terms)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = (key,)
        elif not isinstance(key, tuple):
            key = tuple(key)

        # return the coefficient of the monomial, or 0 if not present
        for coeff,monomial in zip(self.coeffs,self.monomials):
            if monomial.degrees == key:
                return coeff
        return self.ring.coeff_ring(0)

    def __repr__(self):
        return f"Polynomial({repr(self.ring)}, {repr(self.monomials)}, {repr(self.coeffs)})"

    def __call__(self, x):
        if not isiterable(x):
            x = (x,)
        return sum(m(x) * c for c,m in zip(self.coeffs,self.monomials))

    def eval_some(self, eval_map):
        return sum(self.ring(c) * m.eval_some(eval_map) for c,m in zip(self.coeffs,self.monomials))

    def subst(self, subst_map):
        """
        Substitute variables for values in subst_map
        """
        return sum(self.ring(c) * m.subst(subst_map) for c,m in zip(self.coeffs,self.monomials))

    def terms_of_degree(self, n):
        coeffs = [c for c,m in zip(self.coeffs,self.monomials) if m.degree() == n]
        monomials = [m for _,m in zip(self.coeffs,self.monomials) if m.degree() == n]
        return Polynomial(self.ring, monomials, coeffs)

    def eval_coeffs(self, x, ring):
        return sum(ring(c(x)) * ring(m) for c,m in zip(self.coeffs,self.monomials))

    def homogenize(self, y : tuple, s : tuple = None):
        # P(x) -> P(t * y + s) for `t` a formal variable
        if s is None:
            s = (0,) * len(y)
        assert len(y) == len(s)
        hom_ring = PolynomialRing(self.ring.coeff_ring, 't')
        t = Polynomial(hom_ring, [Monomial(hom_ring, (1,))], [1])
        x = tuple(hom_ring(yi) * t + hom_ring(si) for yi,si in zip(y,s))
        return sum(hom_ring(c) * prod(xi ** d for xi,d in zip(x, m.degrees)) \
                   for c,m in zip(self.coeffs, self.monomials))

    def poly_crt(self, other):
        if not isinstance(other, Polynomial):
            other = self.ring(other)

        if not isinstance(self.ring.coeff_ring, GF) or not isinstance(other.ring.coeff_ring, GF):
            raise NotImplementedError()

        assert self.monomials == other.monomials , "Monomial contents does not match"

        monomials = self.monomials
        coeffs = [CRT(coeff, other[monomial]) for coeff,monomial in zip(self.coeffs, monomials)]
        new_ring = self.ring.to_coeff_ring(GF(coeffs[0].p))

        return Polynomial(new_ring, monomials, coeffs)

    def poly_rat_rec(self):
        any_failed = False

        def ratrec(c):
            nonlocal any_failed

            try:
                rec = c.rational_reconstruction()
                # success
            except:
                # failed, needs larger prime
                rec = None
                any_failed = True
            return rec

        if not isinstance(self.ring.coeff_ring, GF):
            raise NotImplementedError()

        rec_monomials = self.monomials
        rec_coeffs = [ratrec(coeff) for coeff in self.coeffs]

        if any_failed:
            return rec_coeffs, rec_monomials

        return Polynomial(self.ring.to_coeff_ring(QQ), rec_monomials, rec_coeffs)

    def __abs__(self):
        return self

    def __add__(self, other):
        dst_ring = self.ring

        if not isinstance(other, Polynomial):
            other = self.ring(other)

        if poly_can_be_coeff(other.ring, self.ring):
            self = Polynomial(other.ring, [other.ring.mon0], [self])
            dst_ring = other.ring
        elif poly_can_be_coeff(self.ring, other.ring):
            other = Polynomial(self.ring, [self.ring.mon0], [other])

        i = 0
        j = 0
        monomials = []
        coeffs = []
        L1 = len(self.monomials)
        L2 = len(other.monomials)
        while i < L1 or j < L2:
            if (i < L1 and j < L2) and self.monomials[i] == other.monomials[j]:
                # term exists in both polynomials
                monomials.append(self.monomials[i])
                coeffs.append(self.coeffs[i] + other.coeffs[j])
                i += 1
                j += 1
            elif i == L1 or (j < L2 and self.monomials[i] > other.monomials[j]):
                # term exists in first polynomial
                monomials.append(other.monomials[j])
                coeffs.append(other.coeffs[j])
                j += 1
            else:
                # term exists in second polynomial
                monomials.append(self.monomials[i])
                coeffs.append(self.coeffs[i])
                i += 1
        return Polynomial(dst_ring, monomials, coeffs)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + other.__neg__()

    def __rsub__(self, other):
        return other + self.__neg__()

    def __mul__(self, other):
        if isinstance(other, Monomial):
            other = Polynomial(other.ring, [other], [other.ring.coeff_one])
        if not isinstance(other, Polynomial):
            other = self.ring(other)

        # For polynomials with coefficients in another polynomial ring
        if poly_can_be_coeff(self.ring, other.ring):
            return Polynomial(self.ring, self.monomials, [other * c for c in self.coeffs])
        if poly_can_be_coeff(other.ring, self.ring):
            return Polynomial(other.ring, other.monomials, [self * c for c in other.coeffs])

        L1, L2 = len(self.monomials), len(other.monomials)

        if L1 == 0 or L2 == 0:
            # Multiplication where one is the 0 polynomial
            return self.ring(0)

        f,g = self,other
        if L1 < L2:
            f,g = g,f

        if L2 == 0:
            return Polynomial.ZERO(self.ring)
        elif L2 == 1:
            mons = [m * g.monomials[0] for m in f.monomials]
            coeffs = [c * g.coeffs[0] for c in f.coeffs]
            p = sum(m * c for m,c in zip(mons,coeffs))
            return p
        else:
            # recursive divide-and-conquer
            h1 = g.monomials[:L2//2]
            h2 = g.monomials[L2//2:]
            c1 = g.coeffs[:L2//2]
            c2 = g.coeffs[L2//2:]
            return f * Polynomial(self.ring, h1, c1) + f * Polynomial(self.ring, h2, c2)

    def __rmul__(self, other):
        # Polynomial rings are commutative
        return self * other

    def __neg__(self):
        """
        Returns the additive inverse of this polynomial
        """
        return Polynomial(self.ring, self.monomials, [-coeff for coeff in self.coeffs])

    def divide(self, divisors):
        if any(div.is_zero() for div in divisors):
            raise ZeroDivisionError

        if not all(isinstance(divisor, Polynomial) for divisor in divisors):
            divisors = [self.ring(divisor) for divisor in divisors]

        p = Polynomial(self.ring, self.monomials, self.coeffs)
        quots = [Polynomial.ZERO(self.ring) for _ in divisors]
        r = Polynomial.ZERO(self.ring)

        # print("divide(1)")
        # for div in divisors:
        #     print(div)

        while len(p.monomials) != 0:
            # print("divide(2)")
            # print(f"p={p}")

            for i in range(len(divisors)):
                LM_p = p.monomials[0]
                LM_i = divisors[i].monomials[0]
                LC_p = p.coeffs[0]
                LC_i = divisors[i].coeffs[0]

                # print(f"({LC_p} {LM_p}) / ({LC_i} {LM_i})")

                if LM_p.divisible_by(LM_i):
                    # print("Divisible")
                    quots[i].monomials.append(LM_p / LM_i)
                    quots[i].coeffs.append(LC_p / LC_i)
                    p = p - (Polynomial(self.ring, [LM_p / LM_i], [LC_p / LC_i]) * divisors[i])
                    break
            else:
                r.monomials.append(p.monomials.pop())
                r.coeffs.append(p.coeffs.pop())

        # print("divide(3)")
        # for quot in quots:
        #     print(quot)

        # print("divide(4)")
        rem = Polynomial(self.ring, list(reversed(r.monomials)), list(reversed(r.coeffs)))
        # print(rem)

        return quots, rem

    def divmod(self, other):
        quots, rem = self.divide([other])
        return quots[0], rem

    def __truediv__(self, other):
        return Rational(self, self.ring(other))

    def __rtruediv__(self, other):
        return Rational(self.ring(other), self)

    def __floordiv__(self, other):
        return self.divmod(other)[0]

    def __pow__(self, power):
        p = Polynomial(self.ring, [self.ring.mon0], [1])
        for _ in range(power):
            p *= self
        return p

    def is_zero(self):
        return len(self.monomials) == 0 # and len(self.coeffs) == 0

    def leading_monomial(self):
        if self.is_zero():
            return Monomial(self.ring, [0] * self.ring.n_vars)
        return self.monomials[0]

    def leading_coeff(self):
        if self.is_zero():
            return self.ring.coeff_zero
        return self.coeffs[0]

    def leading_term(self):
        if self.is_zero():
            return Polynomial.ZERO(self.ring)
        return Polynomial(self.ring, [self.leading_monomial()], [self.leading_coeff()])

    def degree(self):
        return self.leading_monomial().degree()

    def S_polynomial(self, other):
        if self.is_zero() or other.is_zero():
            return self.ring(0)
        else:
            LCM = self.leading_monomial().lcm(other.leading_monomial())
            s_f = Polynomial(self.ring, [LCM / self.leading_monomial()],
                                        [self.ring.coeff_ring(1) / self.leading_coeff()])
            s_g = Polynomial(other.ring, [LCM / other.leading_monomial()],
                                         [other.ring.coeff_ring(1) / other.leading_coeff()])
            return s_f * self - s_g * other

    def __eq__(self, other):
        if not isinstance(other, Polynomial):
            other = self.ring(other)
        return self.monomials == other.monomials and self.coeffs == other.coeffs

    def differentiate(self, deriv_ring : PolynomialRing, a : str):
        p = deriv_ring(self)

        # Differentiate monomial, leave coefficient
        res = sum(deriv_ring(coeff) * monomial.differentiate(deriv_ring, a)
                  for coeff,monomial in zip(p.coeffs,p.monomials))

        if isinstance(p.ring.coeff_ring, PolynomialRing):
            # Differentiate coefficient, leave monomial (product rule)
            res += sum(deriv_ring(coeff.differentiate(p.ring.coeff_ring, a)) * deriv_ring(monomial)
                       for coeff,monomial in zip(p.coeffs,p.monomials))

        return res

########################################################################################################################
#   Unit Tests
########################################################################################################################

import unittest

class TestGCD(unittest.TestCase):

    def test_gcd(self):
        self.assertEqual(gcd(4, 3), 1)
        self.assertEqual(gcd(12, 3), 3)
        self.assertEqual(gcd(21, 9), 3)
        self.assertEqual(gcd(12, 4), 4)
        self.assertEqual(gcd(49, 7), 7)
        self.assertEqual(gcd(1, 2), 1)
        self.assertEqual(gcd(1, -2), gcd(1, 2))
        self.assertEqual(gcd(-1, 2), gcd(1, 2))
        self.assertEqual(gcd(-1, -2), gcd(1, 2))

    def test_xgcd(self):
        self.assertEqual(xgcd(30, 18), (6, -1, 2))
        self.assertEqual(xgcd(18, 30), (6, 2, -1))
        self.assertEqual(xgcd(2, -1), (-1, 0, 1))

class TestMod(unittest.TestCase):

    def test_conversion(self):
        for _ in range(100000):
            p = random.randint(2, 65525)
            x = random.randint(2, 65525)
            self.assertEqual(Mod(x, p), x % p)

    def test_addition(self):
        for _ in range(100000):
            p = random.randint(2, 65525)
            x1 = random.randint(2, 65525)
            x2 = random.randint(2, 65525)
            self.assertEqual(Mod(x1, p) + Mod(x2, p), (x1 + x2) % p)

    def test_subtraction(self):
        for _ in range(100000):
            p = random.randint(2, 65525)
            x1 = random.randint(2, 65525)
            x2 = random.randint(2, 65525)
            self.assertEqual(Mod(x1, p) - Mod(x2, p), (x1 - x2) % p)

    def test_multiplication(self):
        for _ in range(100000):
            p = random.randint(2, 65525)
            x1 = random.randint(2, 65525)
            x2 = random.randint(2, 65525)
            self.assertEqual(Mod(x1, p) * Mod(x2, p), (x1 * x2) % p)

    def test_inversion(self):
        ps = [65413, 65419, 65423, 65437, 65447, 65449, 65479, 65497, 65519, 65521]
        for p in ps:
            x = Mod(random.randint(2, p - 1), p)
            ix = ~x
            self.assertEqual(x * ix, 1)
            self.assertEqual(~ix, x)

    def test_division(self):
        ps = [65413, 65419, 65423, 65437, 65447, 65449, 65479, 65497, 65519, 65521]
        for p in ps:
            x1 = Mod(random.randint(2, p - 1), p)
            x2 = Mod(random.randint(2, p - 1), p)
            self.assertEqual(x1 / x2, x1 * ~x2)

    def test_negation(self):
        for _ in range(100000):
            p = random.randint(2, 65525)
            x = random.randint(2, 65525)
            self.assertEqual(-Mod(x, p), (-x) % p)

    def test_rat_rec(self):
        ps = [65413, 65419, 65423, 65437, 65447, 65449, 65479, 65497, 65519, 65521]
        for p in ps:
            num = random.randint(2, 65525)
            dnm = random.randint(2, 65525)

            x = Mod(num, p) * ~Mod(dnm, p)
            try:
                rr = x.rational_reconstruction()
                self.assertEqual(rr.num, num)
                self.assertEqual(rr.dnm, dnm)
            except:
                pass

        def do_test(r, p):
            F = GF(p)
            rp = F(r)
            rec = rp.rational_reconstruction()
            self.assertEqual(F(rec), rp)
            self.assertEqual(rec, r)

        do_test(Rational(12, 1), 65521)
        do_test(Rational(7, 5), 65521)
        do_test(Rational(-1, 1), 65521)
        do_test(Rational(3,4), 65521)

    def test_crt(self):
        def do_test(r, p1, p2):
            self.assertEqual(CRT(GF(p1)(r), GF(p2)(r)), GF(p1 * p2)(r))

        do_test(Rational(-1, 1), 65521, 509)
        do_test(Rational( 7, 3), 65521, 509)

class TestRational(unittest.TestCase):

    def test_conversion(self):
        self.assertEqual(Rational(0, 0).tup(), (0, 1))
        self.assertEqual(Rational(6, 3).tup(), (2, 1))
        self.assertEqual(Rational(7*4, 3*4).tup(), (7, 3))

    def test_addition(self):
        self.assertEqual((Rational(1, 3) + Rational(1, 3)).tup(), (2, 3))
        self.assertEqual((Rational(4, 5) + Rational(6, 7)).tup(), (58, 35))
        self.assertEqual((Rational(3, 6) + Rational(3, 4)).tup(), (5, 4))
        self.assertEqual((Rational(7, 8) + Rational(5, 6)).tup(), (41, 24))
        self.assertEqual((Rational(4, 3) + Rational(6, 3)).tup(), (10, 3))

    def test_multiplication(self):
        self.assertEqual((Rational(1, 3) * Rational(9, 7)).tup(), (3, 7))
        self.assertEqual((Rational(4, 5) * Rational(12, 11)).tup(), (48, 55))
        self.assertEqual((Rational(3, 2) * Rational(-1, 2)).tup(), (-3, 4))

    def test_inversion(self):
        self.assertEqual((~Rational(2, 3)).tup(), (3, 2))
        self.assertEqual((~Rational(2, -3)).tup(), (-3, 2))

        try:
            ~Rational(0, 0)
            self.fail()
        except ZeroDivisionError:
            self.assertTrue(True)

    def test_division(self):
        self.assertEqual((Rational(2, 3) / Rational(3, 4)).tup(), (8, 9))
        self.assertEqual((Rational(3, 5) / Rational(8, 7)).tup(), (21, 40))
        self.assertEqual((Rational(0, 1) / Rational(4, 1)).tup(), (0, 1))

    def test_ordering(self):
        self.assertLess(Rational(5, 3), Rational(7, 4))
