## libpushdf: Python Library for Computing Pushforwards of Rational Differential Forms

This repository contains a python3 library for computing the pushforward of a rational differential form through the common zeros of a polynomial ideal, based on the methods reported in [10.1007/JHEP10(2022)003](https://inspirehep.net/files/dc4e5b5d5fdb28e4c0031b95e12b39d8).

Much of the computations are performed over finite fields, performing rational function reconstruction techniques reported by [FireFly](https://arxiv.org/abs/1904.00009) and [FiniteFlow](https://arxiv.org/abs/1905.08019) to obtain the final analytic result. This allows using the [FGb library](https://www-polsys.lip6.fr/~jcf/FGb/index.html) for fast Groebner Basis computations over finite fields and avoids much computer algebra.

#### Requirements

- Requires python3.6+ running on linux, due to limitations on available FGb binaries. For use on Windows it is recommended to use the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install).
- The `numpy`, `cython` and `cysignals` python3 packages are required.
- Requires `build-essential`, `libgmp-dev` apt packages or equivalent for your distribution in order to build the native code interface to FGb. `cysignals` may require `autoconf` to build.

Run `python3 fgb/setup.py build_libfgb && python3 fgb/setup.py build` in the root of the repository to build the native code interface to FGb before use.

### Example

```py
#!/usr/bin/env python3
#
#   Example usage of libpushdf, based on the running example in 10.1007/JHEP10(2022)003
#

from libpushdf import PushforwardCalculatorMP

class PushforwardExample(PushforwardCalculatorMP):
    """
    Compute the pushforward of the differential form
        \frac{1}{z1 z2} dz1 ^ dz2
    through the polynomials
        < a1 z1 + a2 z2 , a3 z2^2 - 1 >
    """

    def __init__(self):
        # The tuple (0,1) corresponds to dz1 ^ dz2
        super().__init__(['z1', 'z2'], ['a1', 'a2', 'a3'], (0,1))

    def ideal(self, Z, A):
        z1, z2 = Z
        a1, a2, a3 = A

        # The polynomials are written in the form f(Z, A) = 0
        return [
            a1 * z1 + a2 * z2,
            a3 * z2 ** 2 - 1,
        ]

    def form(self, Z):
        z1, z2 = Z
        return 1 / (z1 * z2)

if __name__ == '__main__':
    pf = PushforwardExample()
    results = pf.run()
    pf.print_tex(results)
```
