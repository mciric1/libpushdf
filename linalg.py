#!/usr/bin/env python3

import math

from libpushdf.basic_types import Mod

class Vector:

    def __init__(self, *args):
        # messy checks for allowing various "nice" constructors
        if len(args) == 1:
            if not '__iter__' in type(args[0]).__dict__:
                a = (args[0],)
            else:
                a = args[0]
            self.elems = list(a)
        else:
            self.elems = list(args)

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.elems == other.elems
        else:
            return self.elems == other

    def __len__(self):
        return len(self.elems)

    def __iter__(self):
        for x in self.elems:
            yield x

    def __setitem__(self, i, v):
        self.elems[i] = v

    def __getitem__(self, i):
        return self.elems[i]

    def __str__(self):
        return f"<{str(tuple(self.elems))[1:-1]}>"

    def __repr__(self):
        return f"Vector({', '.join(str(e) for e in self.elems)})"

    def __add__(self, other):
        assert len(self) == len(other)
        return Vector(self.elems[i] + other.elems[i] for i in range(len(self.elems)))

    def __sub__(self, other):
        assert len(self) == len(other)
        return Vector(self.elems[i] - other.elems[i] for i in range(len(self.elems)))

    def __mul__(self, other):
        return Vector(self.elems[i] * other for i in range(len(self.elems)))

    def __rmul__(self, other):
        return Vector(self.elems[i] * other for i in range(len(self.elems)))

    def __truediv__(self, other):
        return Vector(self.elems[i] / other for i in range(len(self.elems)))

    # def __rtruediv__(self, other):
    #     return Vector(self.elems[i] / other for i in range(len(self.elems)))

    @staticmethod
    def inner(a, b):
        return sum(ai * bi for ai,bi in zip(a,b))

    def magnitude(self):
        return math.sqrt(Vector.inner(self, self))



def LUP_Dec(M):
    assert M.rows == M.cols
    A = M.copy()
    n = M.rows

    P = Vector(*[i for i in range(n)], 0)

    for i in range(n):
        max_a = 0
        imax = i

        for k in range(i, n):
            abs_a = abs(A[k,i])
            if abs_a > max_a:
                max_a = abs_a
                imax = k

        if max_a == 0:
            return None, None # Singular Input

        if imax != i:
            tmp = P[i]
            P[i] = P[imax]
            P[imax] = tmp

            # row swap
            A.row_swap(i, imax)

            # flip parity
            P[n] ^= 1

        for j in range(i + 1, n):
            A[j,i] /= A[i,i]

            for k in range(i + 1, n):
                A[j,k] -= A[j,i] * A[i,k]

    return A, P

def LUP_Solve(LU, P, b):
    n = LU.rows

    x = Vector(*[0 for _ in range(n)])

    for i in range(n):
        x[i] = b[P[i]]

        for k in range(i):
            x[i] -= LU[i,k] * x[k]

    for i in range(n - 1, -1, -1):
        for k in range(i + 1, n):
            x[i] -= LU[i,k] * x[k]
        x[i] /= LU[i,i]

    return x

def LUP_Det(LU, P):
    # Returns the determinant of A = L*U

    if LU is None:
        return 0 # Singular input

    n = LU.rows

    det = 1 if P[n] == 0 else -1
    for i in range(n):
        det *= LU[i,i]

    return det

class Matrix:

    def __init__(self, rows, cols, entries=None):
        self.rows = rows
        self.cols = cols
        self.entries = entries or [0] * rows * cols

    @staticmethod
    def zeros(n, m=None):
        """
        n x m matrix of zeros
        """
        m = m or n
        return Matrix(n, m, [0 for _ in range(n * m)])

    @staticmethod
    def ident(n):
        """
        n x n identity matrix
        """
        return Matrix(n, n, [1 if i == j else 0 for i in range(n) for j in range(n)])

    def copy(self):
        return Matrix(self.rows, self.cols, self.entries.copy())

    def is_zero(self):
        return all(e == 0 for e in self.entries)

    def __str__(self):
        return str(self.entries)

    def pretty_print(self):
        print(f"{self.rows}, {self.cols} :: ")
        for i in range(self.rows):
            print(", ".join([str(self[i,j]) for j in range(self.cols)]))

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        if self.rows == other.rows and self.cols == other.cols:
            return self.entries == other.entries
        return False

    def __setitem__(self, i, v):
        r,c = i
        if r >= self.rows: raise IndexError("Row too large")
        if c >= self.cols: raise IndexError("Column too large")
        self.entries[r * self.cols + c] = v

    def __getitem__(self, i):
        r,c = i
        if r >= self.rows: raise IndexError("Row too large")
        if c >= self.cols: raise IndexError("Column too large")
        return self.entries[r * self.cols + c]

    def col(self, i):
        # Get col i as a vector
        return Vector(self[j,i] for j in range(self.rows))

    def row(self, i):
        # Get row i as a vector
        return Vector(self[i,j] for j in range(self.cols))

    def __add__(self, other):
        if type(other) == int:
            assert self.rows == self.cols
            # Equivalent to self + other * Matrix.ident(self.rows)
            return Matrix(self.rows, self.cols, [
                self[i,j] + other if i == j else self[i,j] for i in range(self.rows) for j in range(self.cols)
            ])

        assert self.rows == other.rows and self.cols == other.cols

        return Matrix(self.rows, self.cols, [
            self.entries[i] + other.entries[i] for i in range(len(self.entries))
        ])

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return Matrix(self.rows, self.cols, [-e for e in self.entries])

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if type(other) in (Vector, list, tuple):
            # Multiply vector
            assert self.cols == len(other)
            return Vector(sum(self[i,j] * other[j] for j in range(self.cols)) for i in range(self.rows))
        elif type(other) == Matrix:
            # Multiply matrix
            assert self.cols == other.rows

            entries = [0] * self.rows * other.cols
            for i in range(self.rows):
                for j in range(other.cols):
                    entries[i * self.rows + j] = sum(self[i,k] * other[k,j] for k in range(self.cols))

            return Matrix(self.rows, other.cols, entries)
        else:
            # Assumed scalar multiplier
            return Matrix(self.rows, self.cols, [
                other * e for e in self.entries
            ])

    def __rmul__(self, other):
        if not isinstance(other, (Matrix, Vector, list, tuple)):
            # Commutative
            return self * other
        raise NotImplementedError()

    def __pow__(self, other):
        assert type(other) == int
        assert self.rows == self.cols

        if other == 0:
            return Matrix.ident(self.rows)
        elif other == 1:
            return self.copy()
        elif other > 1:
            exp = self.copy()
            res = Matrix.ident(self.rows)
            # exponentiation by squaring
            while other > 0:
                if other % 2 == 0:
                    exp *= exp
                    other //= 2
                else:
                    res *= exp
                    other -= 1
            return res
        else:
            # negative, power + inversion
            exp = self ** -other
            return ~exp

    def __truediv__(self, other):
        # computes (other)^{-1}(self) without explicitly computing (other)^{-1}
        n = self.rows

        LU, P = LUP_Dec(other)
        if LU is None:
            raise Exception("Singular Inverse Matrix")

        ret = Matrix(self.rows, self.cols, [0 for _ in range(self.rows * self.cols)])

        for j in range(n):
            # calculate the column through forward-backward-substitution
            col = LUP_Solve(LU, P, self.col(j))
            # set the column
            for i in range(n):
                ret[i,j] = col[i]

        return ret

    def __invert__(self):
        assert self.rows == self.cols
        return Matrix.ident(self.rows) / self

    def trace(self):
        assert self.rows == self.cols
        return sum(self[i,i] for i in range(self.rows))

    def det(self):
        assert self.rows == self.cols
        if self.rows == 2:
            return self[0,0] * self[1,1] - self[0,1] * self[1,0]
        return LUP_Det(*LUP_Dec(self))

    def row_div(self, i, v):
        # divides every element in row i by v
        for c in range(self.cols):
            self[i,c] /= v

    def row_sub(self, i, row):
        # subtract row from row i element-wise
        for c in range(self.cols):
            self[i,c] -= row[c]

    def row_swap(self, i, j):
        # swaps rows i and j
        for c in range(self.cols):
            self[i,c], self[j,c] = self[j,c], self[i,c]

    def reduced_row_echelon_form(self):
        M = self.copy()

        lead = 0
        for r in range(self.rows):
            if self.cols <= lead:
                return M

            i = r

            while M[i, lead] == 0:
                i += 1

                if self.rows == i:
                    i = r
                    lead += 1

                    if self.cols == lead:
                        return M

            if i != r:
                M.row_swap(i, r)

            M.row_div(r, M[r, lead])

            for j in range(self.rows):
                if j != r:
                    M.row_sub(j, M.row(r) * M[j, lead])

            lead += 1

        return M

    def submatrix(self, row_indices, col_indices):
        assert len(set(row_indices)) == len(row_indices) , "Rows should be distinct"
        assert len(set(col_indices)) == len(col_indices) , "Cols should be distinct"
        assert len(row_indices) == len(col_indices) , "Submatrix should be square"

        M = Matrix(len(row_indices), len(col_indices))
        for i in range(M.rows):
            for j in range(M.cols):
                M[i,j] = self[row_indices[i],col_indices[j]]
        return M

    def minor(self, row_indices, col_indices):
        return self.submatrix(row_indices, col_indices).det()

########################################################################################################################
#   Unit Tests
########################################################################################################################

import unittest
from libpushdf.basic_types import Rational

class TestMatrix(unittest.TestCase):
    def test_init(self):
        M = Matrix(2, 2, [
            1, 2,
            3, 4
        ])
        self.assertEqual(M[0,0], 1)
        self.assertEqual(M[0,1], 2)
        self.assertEqual(M[1,0], 3)
        self.assertEqual(M[1,1], 4)

    def test_arith(self):
        M = Matrix(2, 2, [
            1, 2,
            3, 4
        ])

        N = Matrix(2, 2, [
            1, 2,
            3, 4
        ])

        A = M**2 * N - M**2
        B = M**2 * (N - 1)
        self.assertEqual(A, B)

        M2 = Matrix(3,3, [
             4, -1,  3,
            -2,  1,  2,
            -1,  3, -3
        ])
        self.assertEqual(4 * M2 + 1, Matrix(3,3, [
            17, -4, 12,
            -8,  5,  8,
            -4, 12, -11
        ]))

    def test_det(self):
        M = Matrix(3, 3, [
            0, 1, 2,
            3, 4, 5,
            6, 7, 8
        ])

        self.assertEqual(M.det(), 0)
        self.assertEqual(M.minor((1,2),(1,2)), -3)
        self.assertEqual(M.minor((0,2),(0,2)), -12)

    def test_mul_vector(self):
        M = Matrix(3, 3, [
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        ])
        v = Vector(1, 2, 3)
        self.assertEqual(M * v, Vector(14, 32, 50))

    def test_invert(self):
        M = Matrix(3,3, [
             2, -1, -2,
            -4,  6,  3,
            -4, -2,  8
        ])
        M.entries = [Rational(e,1) for e in M.entries]
        Minv = ~M
        self.assertEqual(Minv.entries, [
            Rational(9, 4), Rational(1, 2), Rational(3,  8),
            Rational(5, 6), Rational(1, 3), Rational(1, 12),
            Rational(4, 3), Rational(1, 3), Rational(1,  3)
        ])
        self.assertEqual(M * Minv, Matrix.ident(M.rows))

    def test_div(self):
        M1 = Matrix(3,3, [
             2, -1, -2,
            -4,  6,  3,
            -4, -2,  8
        ])
        M1.entries = [Rational(e,1) for e in M1.entries]
        M2 = Matrix(3,3, [
             4, -1,  3,
            -2,  1,  2,
            -1,  3, -3
        ])
        M2.entries = [Rational(e,1) for e in M2.entries]

        M3 = M2 / M1
        self.assertEqual(M3.entries, [
            Rational(61,  8), Rational(-5, 8), Rational(53,  8),
            Rational(31, 12), Rational(-1, 4), Rational(35, 12),
            Rational(13,  3), Rational( 0, 1), Rational(11,  3)
        ])
