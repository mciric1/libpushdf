#!/usr/bin/env python3
#
#   Permutation implementation
#

import itertools

class Permutation:
    def __init__(self, p):
        if type(p) == dict: # 1:1 mapping
            self.map = p.copy()
        elif type(p) == tuple: # cyclic
            self.map = {}
            def map_single_cyc(q):
                self.map.update({ e : q[i % len(p)] for i,e in enumerate(q, 1) })
            if len(p) != 0 and type(p[0]) == tuple: # single-cycle
                for t in p:
                    map_single_cyc(t)
            else:
                map_single_cyc(p)
        else:
            raise TypeError()

        # precalc sign
        self.sign = self.sgn()

    def __call__(self, i):
        if not isinstance(i, int):
            raise TypeError()
        return self.map.get(i, i)

    def __eq__(self, p):
        return self.map == p.map

    def __iter__(self):
        for k,v in self.map.items():
            yield k,v

    def cyc_for(self, n):
        """
        Get cycle for which element n is first
        """
        cyc = [n]
        k = self.map[n]
        while k != n:
            cyc.append(k)
            k = self.map[k]
        return cyc

    def cyc(self):
        """
        Get all cycles for this permutation
        """
        cycles = []
        # Generate cycles
        for k,_ in self:
            cycles.append(self.cyc_for(k))
        # Sort cycles for smallest element first while preserving order
        for cyc in cycles:
            while any([cyc[i] < cyc[0] for i in range(len(cyc))]):
                e = cyc[0]
                del cyc[0]
                cyc.append(e)
        # Deduplicate cycles and return
        return list(map(list, set(map(tuple, cycles))))

    def is_even(self):
        """
        Determines if this permutation is even
        """
        acc = sum([len(cycle) - 1 for cycle in self.cyc()])
        return acc % 2 == 0

    def sgn(self):
        return 1 if self.is_even() else -1

    def cyc_str(self):
        """
        Produces cycle notation for this permutation
        """
        nontrivial_cycles = []
        cycles = self.cyc()
        for cyc in cycles:
            if len(cyc) > 1:
                nontrivial_cycles.append(cyc)
        if len(nontrivial_cycles) > 0:
            return "(" + ")(".join([",".join([f"{e}" for e in cyc]) for cyc in nontrivial_cycles]) + ")"
        return "(IDENT)"

    def __str__(self):
        return str(self.map)

def Sym(n):
    """
    Generate the Symmetric Group of order n
    """
    return tuple(Permutation(dict(zip(range(1,n+1),e))) for e in itertools.permutations(range(1,n+1)))
