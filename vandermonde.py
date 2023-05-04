
def solve_shifted_transposed_vandermonde(vs, fs):
    """
    Solves a vandermonde system.

    Adapted from Numerical Recipes in C
    """
    n_eqs = len(vs)

    if n_eqs == 0:
        return [0]

    result = [0] * n_eqs
    cis = [0] * n_eqs

    cis[n_eqs - 1] = -vs[0]

    for i in range(1, n_eqs):
        xx = vs[i]

        for j in range(n_eqs - 1 - i, n_eqs - 1):
            cis[j] -= xx * cis[j + 1]

        cis[n_eqs - 1] -= xx

    for i in range(n_eqs):
        xx = vs[i]
        t = b = 1
        s = fs[n_eqs - 1]

        for j in reversed(range(1, n_eqs)):
            b = cis[j] + xx * b
            s += fs[j - 1] * b
            t = xx * t + b

        result[i] = s / t / xx

    return result
