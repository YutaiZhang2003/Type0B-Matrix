"""Exact finite checks of the graph signs; distinct from numerical CFT tests."""
from itertools import product
from pathlib import Path
import hashlib
import json


def bits(n):
    return list(product((0, 1), repeat=n))


class Graph:
    def __init__(self, name, vertices, pairs, external):
        self.name = name
        self.vertices = vertices
        self.pairs = pairs
        self.external = external
        self.reference = [h for p in pairs for h in p] + external
        self.target = [h for v in vertices for h in v]
        assert set(self.reference) == set(self.target)
        assert len(self.reference) == len(set(self.reference))
        self.edge = {h: i for i, p in enumerate(pairs) for h in p}
        self.edge.update({h: len(pairs)+i for i, h in enumerate(external)})
        self.n = len(pairs)+len(external)
        pos = {h: i for i, h in enumerate(self.target)}
        self.inversions = [(h,k) for i,h in enumerate(self.reference)
                           for k in self.reference[i+1:] if pos[h]>pos[k]]

    def S(self, x):
        return sum(x[self.edge[h]]*x[self.edge[k]] for h,k in self.inversions)%2

    def K(self, x, y):
        xy = tuple(a^b for a,b in zip(x,y))
        return (self.S(xy)+self.S(x)+self.S(y)
                +sum(x[self.edge[v[1]]]*y[self.edge[v[2]]]
                     for v in self.vertices))%2

    def lr_sign(self, x, y):
        xy = tuple(a^b for a,b in zip(x,y))
        actual = self.S(xy)+self.S(x)+self.S(y)
        actual += sum(x[e]*y[e] for e in range(len(self.pairs)))
        actual += sum(y[self.edge[v[i]]]*x[self.edge[v[j]]]
                      for v in self.vertices for i in range(3) for j in range(i+1,3))
        d = [sum(x[self.edge[h]] for h in v)%2 for v in self.vertices]
        dy = [sum(y[self.edge[h]] for h in v)%2 for v in self.vertices]
        expected = sum(dy[i]*d[j] for i in range(len(d)) for j in range(i+1,len(d)))
        expected += sum(y[self.edge[h]]*x[self.edge[k]]
                        for i,h in enumerate(self.external) for k in self.external[i+1:])
        return actual%2, expected%2


graphs = [
    Graph('theta', [('1L','2L','3L'),('1R','2R','3R')],
          [('1L','1R'),('2L','2R'),('3L','3R')], []),
    Graph('glasses', [('1L','2L','2R'),('1R','3L','3R')],
          [('1L','1R'),('2L','2R'),('3L','3R')], []),
    Graph('Ramond triangle with three external NS legs',
          [('a','0L','2R'),('b','1L','0R'),('c','2L','1R')],
          [('0L','0R'),('1L','1R'),('2L','2R')], ['a','b','c']),
]


def main():
    results = []
    for g in graphs:
        values = bits(g.n)
        kernels = {(x,y):g.K(x,y) for x in values for y in values}
        count = 0
        for x,y in product(values, repeat=2):
            assert g.lr_sign(x,y)[0] == g.lr_sign(x,y)[1], (g.name,x,y)
            for z in values:
                xy = tuple(a^b for a,b in zip(x,y))
                yz = tuple(a^b for a,b in zip(y,z))
                assert (kernels[x,y]+kernels[xy,z]-kernels[y,z]-kernels[x,yz])%2 == 0
                count += 1
        results.append(dict(graph=g.name, correlator_sign_cases=len(values)**2,
                            associative_cocycle_cases=count, passed=True))
    for x in bits(3):
        assert graphs[0].S(x) == (x[0]*x[1]+x[0]*x[2]+x[1]*x[2])%2
        assert graphs[1].S(x) == 0
        for y in bits(3):
            assert graphs[0].K(x,y) == sum(x[i]*y[j]+x[j]*y[i]
                                         for i in range(3) for j in range(i+1,3))%2
            assert graphs[1].K(x,y) == (x[1]*y[1]+x[2]*y[2])%2
    loops = [(graphs[0],(0,1,1),2), (graphs[1],(0,1,0),1),
             (graphs[1],(0,0,1),1), (graphs[2],(1,1,1,0,0,0),3)]
    for g,c,m in loops:
        phase = (1j)**(-m)*(-1)**g.S(c)
        assert phase**2*(-1)**g.K(c,c) == 1
        for x in bits(g.n):
            assert g.K(c,x) == g.K(x,c)
    # All parity assignments satisfying the even auxiliary vertex condition.
    local_cases = 0
    for A,B,p,f,d1,d2 in bits(6):
        d3 = d1^d2
        epsilon2 = B  # B here denotes the total second-slot physical parity.
        lhs = (1j)**((A+d1)%2)*(-1j)**A
        lhs *= (-1)**(f*(A+epsilon2)+A*d1+(epsilon2+p)*d3)
        rhs = (1j)**d1*(-1)**(f*(A+epsilon2)+p*d3+epsilon2*d3)
        assert lhs == rhs
        local_cases += 1
    data = dict(passed=True, graph_checks=results, closed_loop_cases=len(loops),
                local_R_vertex_phase_cases=local_cases,
                scope='Exact permutation, graded tensor, and phase identities; not numerical CFT blocks.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out = Path(__file__).with_name('graph_sign_results.json')
    out.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__ == '__main__':
    main()
