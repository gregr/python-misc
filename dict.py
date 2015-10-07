from __future__ import absolute_import
from collections import defaultdict
from operator import add, sub


countdict = lambda orig={}: defaultdict(int, orig)
setdict = lambda orig={}: defaultdict(set, orig)


class SetOps(object):
    def __init__(self, merge, diff, is_empty):
        self._merge = merge
        self._diff = diff
        self._is_empty = is_empty

    def union(self, d0, d1):
        result = d0.copy()
        for key, agg in d1.iteritems():
            result[key] = self._merge(result[key], agg)
        return result

    def difference(self, d0, d1):
        result = d0.copy()
        for key, agg in d1.iteritems():
            current = result.get(key)
            if current is not None:
                new = self._diff(current, agg)
                if self._is_empty(new):
                    del result[key]
                else:
                    result[key] = new
        return result

    def difference_symmetric(self, d0, d1):
        shadow0 = self.difference(d0, d1)
        shadow1 = self.difference(d1, d0)
        return self.union(shadow0, shadow1)

    def intersection(self, d0, d1):
        shadow = self.difference(d0, d1)
        return self.difference(d0, shadow)


# TODO: should countdict union use max/min instead of +/-?
countdict_setops = SetOps(add, sub, lambda x: x <= 0)
setdict_setops = SetOps(set.union, sub, lambda x: len(x) == 0)


if __name__ == '__main__':
    import unittest

    class TestCountDictSetOps(unittest.TestCase):
        def setUp(self):
            self.d0 = countdict(dict(a=1, b=5, c=3))
            self.d1 = countdict(dict(b=3, c=5, d=7))

        def test_union(self):
            self.assertEqual(countdict_setops.union(self.d0, self.d1),
                             dict(a=1, b=8, c=8, d=7))

        def test_intersection(self):
            self.assertEqual(countdict_setops.intersection(self.d0, self.d0),
                             self.d0)
            self.assertEqual(countdict_setops.intersection(self.d0, self.d1),
                             dict(b=3, c=3))

    class TestSetDictSetOps(unittest.TestCase):
        def setUp(self):
            self.d0 = setdict(
                dict(a=set([1, 3]), b=set([3, 5]), c=set([5, 7])))
            self.d1 = setdict(
                dict(d=set([0, 1]), b=set([2, 5]), c=set([5, 8])))

        def test_union(self):
            self.assertEqual(setdict_setops.union(self.d0, self.d1),
                             dict(a=set([1, 3]), b=set([2, 3, 5]),
                                  c=set([5, 7, 8]), d=set([0, 1])))

        def test_intersection(self):
            self.assertEqual(setdict_setops.intersection(self.d0, self.d0),
                             self.d0)
            self.assertEqual(setdict_setops.intersection(self.d0, self.d1),
                             dict(b=set([5]), c=set([5])))

    unittest.main()
