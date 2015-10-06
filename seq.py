from __future__ import absolute_import
from itertools import izip_longest


def cross(*xss):
    results = [()]
    for xs in xss:
        results = [result + (x,) for result in results for x in xs]
    return results


def subseqs(xs):
    if xs:
        prev = subseqs(xs[1:])
        return [(xs[0],) + y for y in prev] + prev
    else:
        return [()]


def chunk(iterable, n):
    args = [iter(iterable)] * n
    return list(izip_longest(fillvalue=None, *args))
