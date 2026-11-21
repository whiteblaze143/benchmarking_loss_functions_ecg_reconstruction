import itertools
import math
import unittest
from metrics import charitos_density, discrete_pattern_reference, discrete_pattern_from_runs, charitos_author_binned_port


def intervals_from_binary(x):
    starts = [i for i,v in enumerate(x) if v and (i == 0 or not x[i-1])]
    ends = [i+1 for i,v in enumerate(x) if v and (i == len(x)-1 or not x[i+1])]
    return list(zip(starts,ends))


class MetricsTests(unittest.TestCase):
    def test_edge_cases(self):
        self.assertIsNone(charitos_density([],10))
        self.assertIsNone(charitos_density([(0,10)],10))
        for intervals in [[(-1,2)],[(2,1)],[(0,2),(1,3)],[(0,1),(1,2)]]:
            with self.assertRaises(ValueError):charitos_density(intervals,10)

    def test_single_episode(self):
        for a,b in [(0,1),(4,5),(0,9),(1,10)]:
            self.assertAlmostEqual(charitos_density([(a,b)],10),1)

    def test_exhaustive_shortest_windows(self):
        # Independently enumerate all windows and compare against the block-gap
        # envelope at every integer AF amount for all binary sequences up to 8.
        for n in range(2,9):
            for x in itertools.product((0,1),repeat=n):
                k=sum(x)
                if k==0 or k==n:continue
                episodes=intervals_from_binary(x)
                blocks=[]
                for i,(a,_) in enumerate(episodes):
                    for j in range(i,len(episodes)):
                        total=sum(b-a for a,b in episodes[i:j+1])
                        blocks.append((total,episodes[j][1]-a-total))
                for p in range(1,k+1):
                    brute=min(b-a for a in range(n) for b in range(a+1,n+1) if sum(x[a:b])>=p)
                    envelope=p+min(gap for total,gap in blocks if total>=p)
                    self.assertEqual(brute,envelope)
                d=charitos_density(episodes,n)
                self.assertTrue(math.isfinite(d) and -1e-12<=d<=1+1e-12)
                reverse=[(n-b,n-a) for a,b in reversed(episodes)]
                self.assertAlmostEqual(d,charitos_density(reverse,n))
                scaled=[(a*3.7,b*3.7) for a,b in episodes]
                self.assertAlmostEqual(d,charitos_density(scaled,n*3.7))

    def test_discrete_single_episode(self):
        for x in [[0,1,0,0],[1,1,0,0],[0,1,1,1]]:
            self.assertAlmostEqual(discrete_pattern_reference(x)['density'],1)

    def test_optimized_discrete_matches_literal_equations(self):
        for n in range(2,10):
            for x in itertools.product((0,1),repeat=n):
                actual=discrete_pattern_from_runs(intervals_from_binary(x),n)
                expected=discrete_pattern_reference(x)
                for key in expected:
                    if expected[key] is None:self.assertIsNone(actual[key])
                    else:self.assertAlmostEqual(actual[key],expected[key])

    def test_author_port_edge_conventions(self):
        self.assertEqual(charitos_author_binned_port([0,0]),0)
        self.assertEqual(charitos_author_binned_port([1,1],timeunits=1),1)
        self.assertTrue(math.isinf(charitos_author_binned_port([1,1],timeunits=1,limit=False)))
        for unit in [0,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):charitos_author_binned_port([1],timeunits=unit)

    def test_known_distinction(self):
        # The continuous and finite beat-indexed normalizations differ.
        self.assertAlmostEqual(charitos_density([(0,1),(3,4)],4),0.5)
        self.assertAlmostEqual(discrete_pattern_reference([1,0,0,1])['density'],1/3)

if __name__=='__main__':unittest.main()
