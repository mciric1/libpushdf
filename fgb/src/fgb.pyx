# distutils: language = c

import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cysignals.signals cimport sig_on, sig_off, sig_check

cdef extern from "fgb_impl.c":
    # "style.h"
    ctypedef unsigned UI32
    ctypedef int I32

    # "protocol_maple.h"
    ctypedef struct SFGB_Comp_Desc:
        I32 _force_elim
        UI32 _index

    ctypedef struct SFGB_Options:
        SFGB_Comp_Desc _env
        I32 _verb

    ctypedef SFGB_Options* FGB_Options

    # "call_fgb.h"
    ctypedef void* Dpol
    ctypedef Dpol Dpol_INT

    void threads_FGb(int t)
    void init_FGb_Modp(const int p)
    void FGb_set_coeff_I32(Dpol p, UI32 i0, I32 buf)
    I32 FGb_export_poly(I32 n, I32 m, I32* E, I32* P, Dpol p)
    void FGb_set_default_options(FGB_Options options)

    # "fgb_impl.c"
    void init_fgb_gmp()
    void set_elim_order(UI32 bl1, UI32 bl2, char** liste)
    Dpol create_poly(UI32 n)
    void set_expos2(Dpol p, UI32 i0, I32* e, const UI32 nb)
    void full_sort_poly2(Dpol p)
    UI32 fgb(Dpol* p, UI32 np, Dpol* q, UI32 nq, double* t0, FGB_Options options) nogil
    UI32 nb_terms(Dpol p)
    void reset_memory()
    void restoreptr()
    I32 fgb_internal_version()

cdef class FGbRunner:
    cdef:
        Dpol* output_basis
        Dpol* input_basis
        char** variables
        I32* exponents
        UI32 n_variables
        int n_input, max_base
        object pystr_variables # to keep references alive
        double cputime

    def __cinit__(self, list polys, int n_vars, list var_names, int n_elim_variables, int max_base):
        self.n_input = len(polys)
        self.n_variables = n_vars
        self.max_base = max_base
        self.output_basis = <Dpol*> PyMem_Malloc(self.max_base * sizeof(Dpol))
        self.input_basis = <Dpol*> PyMem_Malloc(self.n_input * sizeof(Dpol))
        self.variables = <char**> PyMem_Malloc(self.n_variables * sizeof(char*))
        self.exponents = <I32*> PyMem_Malloc(self.n_variables * sizeof(I32*))
        if not self.input_basis or not self.output_basis or not self.variables or not self.exponents:
            raise MemoryError()

        self.pystr_variables = [x.encode('ASCII') for x in var_names]

        cdef int i
        for (i, x) in enumerate(self.pystr_variables):
            self.variables[i] = x

        set_elim_order(n_elim_variables, self.n_variables - n_elim_variables, self.variables)

        cdef Dpol q
        cdef UI32 j, n_monoms

        for k,p in enumerate(polys):
            n_monoms = len(p.monomials)

            q = create_poly(n_monoms)

            for i in range(n_monoms):
                FGb_set_coeff_I32(q, i, <I32>p.coeffs[i].x)

                for j in range(self.n_variables):
                    self.exponents[j] = p.monomials[i][j]

                set_expos2(q, i, self.exponents, self.n_variables)

            full_sort_poly2(q)
            self.input_basis[k] = q

    def __dealloc__(self):
        PyMem_Free(self.output_basis)
        PyMem_Free(self.input_basis)
        PyMem_Free(self.variables)
        PyMem_Free(self.exponents)

    cdef list run(self, FGB_Options options):
        cdef UI32 n_output

        sig_on()
        n_output = fgb(self.input_basis, self.n_input, self.output_basis, self.max_base, &self.cputime, options)
        sig_off()

        output = [0] * n_output

        cdef int i
        cdef UI32* monoms
        cdef I32* coeffs
        cdef UI32 n_monoms
        cdef Dpol q
        cdef UI32 j
        cdef UI32* e
        for i in range(n_output):

            q = self.output_basis[i]
            n_monoms = nb_terms(q)
            monoms = <UI32*> PyMem_Malloc(n_monoms * self.n_variables * sizeof(UI32))
            coeffs = <I32*> PyMem_Malloc(n_monoms * sizeof(I32))
            if not monoms or not coeffs:
                raise MemoryError()

            sig_check()

            FGb_export_poly(self.n_variables, n_monoms, <I32*>monoms, coeffs, q)

            my_monoms = [0] * n_monoms
            my_coeffs = [0] * n_monoms

            for j in range(n_monoms):
                e = monoms + j * self.n_variables

                my_monoms[j] = tuple(e[k] for k in range(self.n_variables))
                my_coeffs[j] = coeffs[j]

            output[i] = (my_monoms, my_coeffs)

            PyMem_Free(monoms)
            PyMem_Free(coeffs)

        return output

def fgb_eliminate(polys, p, int n_vars, var_names, n_elim_variables, threads, force_elim, matrix_bound, verbosity, max_base):
    cdef SFGB_Options Opt
    cdef FGB_Options options = &Opt

    try:
        init_fgb_gmp()
        init_FGb_Modp(p)
        threads_FGb(threads)
        FGb_set_default_options(options)
        options._env._force_elim = force_elim
        options._env._index = matrix_bound
        options._verb = verbosity

        return FGbRunner(polys, n_vars, var_names, n_elim_variables, max_base).run(options)
    except KeyboardInterrupt as e:
        # user abort
        raise e
    finally:
        reset_memory()
        restoreptr()

MAX_PRIME = 65521

def groebner_basis(polys, **kwds):
    r"""
    Compute a Groebner basis for a polynomial ideal with coefficients in a finite field using FGb.
    Monomial ordering must be grevlex.
    Prime field must be 2 <= p <= 65521 < 2^16

    INPUT:
    - ``polys``          -- a polynomial sequence generating the ideal.

    - ``threads``        -- integer (default: `1`).

    - ``force_elim``     -- integer (default: `0`); if ``force_elim=1``, then the computation will
                            return only the result of the elimination, if an elimination order is used.

    - ``verbosity``      -- integer (default: `0`), display progress info.

    - ``matrix_bound``   -- integer (default: `500000`); this is is the maximal size of the matrices
                            generated by F4. This value can be increased according to available memory.

    - ``max_base``       -- integer (default: `100000`); maximum number of polynomials in output.
    OUTPUT:
        the Groebner basis.
    EXAMPLES:
    """
    kwds.setdefault('threads', 1)
    kwds.setdefault('force_elim', 0)
    kwds.setdefault('verbosity', 0)
    kwds.setdefault('matrix_bound', 500000)
    kwds.setdefault('max_base', 100000)

    ring = polys[0].ring
    if ring.coeff_ring.p > MAX_PRIME:
        raise NotImplementedError("maximum prime field size is %s" % MAX_PRIME)
    return fgb_eliminate(polys, ring.coeff_ring.p, ring.n_vars, ring.var_names, ring.n_vars, **kwds)
