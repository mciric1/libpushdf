#define USE_MY_OWN_IO 1
#define LIBMODE 1
#include "call_fgb.h"

#if LIBMODE == 1
int FGb_verb_info = 0;
#endif

void init_fgb_gmp(void)
{
    FGB(saveptr)();
}

void set_elim_order(UI32 bl1, UI32 bl2, char **list)
{
    FGB(PowerSet)(bl1, bl2, list);
}

Dpol create_poly(UI32 n) {
    return FGB(creat_poly)(n);
}

void set_expos2(Dpol p, UI32 i0, I32 *e, const UI32 nb)
{
    FGB(set_expos2)(p, i0, e, nb);
}

void full_sort_poly2(Dpol p)
{
   FGB(full_sort_poly2)(p);
}

UI32 fgb(Dpol *p, UI32 np, Dpol *q, UI32 nq, double *t0, FGB_Options opts)
{
    return FGB(fgb)(p, np, q, nq, t0, opts);
}

UI32 nb_terms(Dpol p)
{
    return FGB(nb_terms)(p);
}

void reset_memory(void)
{
    FGB(reset_memory)();
}

void restoreptr(void)
{
    FGB(restoreptr)();
}

I32 fgb_internal_version(void)
{
    return FGB(internal_version)();
}
