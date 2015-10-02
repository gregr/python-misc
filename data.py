from __future__ import absolute_import
from bisect import bisect_left
import csv
from itertools import chain, izip_longest
import math
from .dict import countdict, countdict_setops
from .logging import Meter

default_meter_period = 10000


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return list(izip_longest(fillvalue=None, *args))


def validate_sanity(file_name):
    counts = set()
    for row in csv.reader(open(file_name)):
        counts.add(len(row))
    return len(counts) <= 1


class ColumnSummary(object):
    def __init__(self, freqs):
        self.cats = freqs
        self.nums = countdict()

    def enable_interval(self):
        self.nums = countdict()
        for key, count in self.cats.items():
            try:
                self.nums[float(key)] = count
                del self.cats[key]
            except ValueError:
                pass

    def disable_interval(self):
        for key, count in self.nums.iteritems():
            self.cats[str(key)] = count
        self.nums = countdict()

    def refresh(self):
        self.sum = 0
        self.count = 0
        for num, count in self.nums.iteritems():
            self.count += count
            self.sum += num * count
        num_keycount = float(len(self.nums))
        self.ratio = num_keycount / (num_keycount + float(len(self.cats)))
        if num_keycount > 0:
            self.coverage = num_keycount / self.count
            self.min = min(self.nums)
            self.max = max(self.nums)
        else:
            self.coverage = 0.0
            self.min = None
            self.max = None

    def stats(self):
        if self.count > 0:
            self.mean = self.sum / self.count

    def __getitem__(self, key):
        if isinstance(key, float):
            return self.nums[key]
        return self.cats[key]

    def __setitem__(self, key, val):
        if isinstance(key, float):
            self.nums[key] = val
        self.cats[key] = val

    def __delitem__(self, key):
        if isinstance(key, float):
            del self.nums[key]
        del self.cats[key]

    def iterkeys(self):
        return chain(self.cats.iterkeys(), self.nums.iterkeys())

    def union(self, summ):
        cats = countdict_setops.union(self.cats, summ.cats)
        return ColumnSummary(cats)

    def difference(self, summ):
        cats = countdict_setops.difference(self.cats, summ.cats)
        return ColumnSummary(cats)

    def intersection(self, summ):
        shadow = self.difference(summ)
        return self.difference(shadow)

    def difference_symmetric(self, summ):
        shadow = self.difference(summ)
        return summ.difference(self).union(shadow)


class Column(object):
    def __init__(self, summary):
        self.summary = summary
        self.move = {}
        self.moved_from = {}
        for key in self.summary.iterkeys():
            self.move[key] = key
            self.moved_from[key] = key

    def move(self, source, target, force=False):
        keys = set(self.moved_from.iterkeys())
        if not force and target in keys:
            raise RuntimeError("target already exists; src='%s' tgt='%s'",
                               source, target)
        orig = self.moved_from[source]
        del self.moved_from[source]
        self.moved_from[target] = orig
        self.move[orig] = target
        count = self.summary[source]
        del self.summary[source]
        self.summary[target] += count

    def transforms(self):
        return self.move

    def uniques(self):
        return tuple(self.summary.iterkeys())

    def union(self, col):
        return Column(self.summary.union(col.summary))

    def difference(self, col):
        return Column(self.summary.difference(col.summary))

    def intersection(self, col):
        return Column(self.summary.intersection(col.summary))

    def difference_symmetric(self, col):
        return Column(self.summary.difference_symmetric(col.summary))


class Frame(object):
    def __init__(self, names, cols):
        self.names = names
        self.cols = cols
        self.name_to_index = {}
        for idx, name in enumerate(names):
            self.name_to_index[name] = idx
        self.name_to_index_orig = self.name_to_index.copy()
        self.removed = set()

    def get(self, name):
        idx = self.name_to_index.get(name)
        if idx is None:
            return None
        return self.cols[idx]

    def __getitem__(self, name):
        return self.cols[self.name_to_index[name]]

    def __delitem__(self, name):
        idx = self.name_to_index[name]
        del self.name_to_index[name]
        self.removed.add(name)
        self.cols.pop(idx)
        self.names.pop(idx)
        for name in self.names[idx:]:
            self.name_to_index[name] -= 1

    def filter(self, ratio=None, coverage=None):
        misses = {}
        hits = {}
        for idx, cols in enumerate(self.cols):
            name = self.names[idx]
            summ = cols.summary
            summ.refresh()
            if ((coverage is not None and summ.coverage < coverage) or
                    (ratio is not None and summ.ratio < ratio)):
                misses[name] = summ
                continue
            hits[name] = summ
        return hits, misses

    def transforms(self):
        transforms = [col.transforms() for col in self.cols]
        for idx in sorted(self.name_to_index_orig[name]
                          for name in self.removed):
            transforms.insert(idx, None)
        names = [None] * len(self.names)
        for name, idx in self.name_to_index.iteritems():
            names[idx] = name
        return names, transforms

    def zip(self, frame):
        for name, col0 in zip(self.names, self.cols):
            col1 = frame.get(name)
            if col1 is not None:
                yield name, (col0, col1)

    def zipmap(self, op, frame):
        names, colpairs = zip(*self.zip(frame))
        cols = [op(c0, c1) for c0, c1 in colpairs]
        return Frame(names, cols)

    def union(self, frame):
        return self.zipmap(Column.union, frame)

    def difference(self, frame):
        return self.zipmap(Column.difference, frame)

    def intersection(self, frame):
        return self.zipmap(Column.intersection, frame)

    def difference_symmetric(self, frame):
        return self.zipmap(Column.difference_symmetric, frame)


def summarize(file_name, header=None, limit=None,
              meter_period=default_meter_period):
    reader = csv.reader(open(file_name))
    if header is None:
        header = reader.next()
    freqs = [countdict() for _ in header]
    meter = Meter(meter_period, 'total rows processed: %d')
    for ridx, row in enumerate(reader):
        if limit is not None and ridx >= limit:
            break
        for cidx, col in enumerate(row):
            freqs[cidx][col] += 1
        meter.inc(1)
    meter.log()
    frame = Frame(header, [Column(ColumnSummary(freq)) for freq in freqs])
    return frame


def pairwise_freqs(file_name, col0, col1, header=None, limit=None,
                   meter_period=default_meter_period):
    reader = csv.reader(open(file_name))
    if header is None:
        header = reader.next()
    name_to_index = dict((name, idx) for idx, name in enumerate(header))
    idx0 = name_to_index[col0]
    idx1 = name_to_index[col1]
    freqs = countdict()
    meter = Meter(meter_period, 'total rows processed: %d')
    for ridx, row in enumerate(reader):
        if limit is not None and ridx >= limit:
            break
        freqs[(row[idx0], row[idx1])] += 1
        meter.inc(1)
    meter.log()
    return freqs


def chi_squared(pair_freqs):
    keys = sorted(pair_freqs.iterkeys())
    ks0, ks1 = map(sorted, map(set, zip(*keys)))
    gs0 = grouper(sorted(pair_freqs.iteritems()), len(ks1))
    gs1 = zip(*gs0)
    ts0, ts1 = [[sum(zip(*grp)[1]) for grp in gs] for gs in (gs0, gs1)]
    total = float(sum(ts0))
    result = 0
    for k0, t0 in zip(ks0, ts0):
        for k1, t1 in zip(ks1, ts1):
            observed = pair_freqs[(k0, k1)]
            expected = (t0 / total) * t1
            result += ((observed - expected) ** 2) / expected
    deg_freedom = (len(ks0) - 1) * (len(ks1) - 1)
    return result, deg_freedom


def chi_squared_with(*args, **kwargs):
    return chi_squared(pairwise_freqs(*args, **kwargs))

confidences = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001)

chi_squareds = [
    (0.004, 0.02, 0.06, 0.15, 0.46, 1.07, 1.64, 2.71, 3.84, 6.64, 10.83),
    (0.10, 0.21, 0.45, 0.71, 1.39, 2.41, 3.22, 4.60, 5.99, 9.21, 13.82),
    (0.35, 0.58, 1.01, 1.42, 2.37, 3.66, 4.64, 6.25, 7.82, 11.34, 16.27),
    (0.71, 1.06, 1.65, 2.20, 3.36, 4.88, 5.99, 7.78, 9.49, 13.28, 18.47),
    (1.14, 1.61, 2.34, 3.00, 4.35, 6.06, 7.29, 9.24, 11.07, 15.09, 20.52),
    (1.63, 2.20, 3.07, 3.83, 5.35, 7.23, 8.56, 10.64, 12.59, 16.81, 22.46),
    (2.17, 2.83, 3.82, 4.67, 6.35, 8.38, 9.80, 12.02, 14.07, 18.48, 24.32),
    (2.73, 3.49, 4.59, 5.53, 7.34, 9.52, 11.03, 13.36, 15.51, 20.09, 26.12),
    (3.32, 4.17, 5.38, 6.39, 8.34, 10.66, 12.24, 14.68, 16.92, 21.67, 27.88),
    (3.94, 4.87, 6.18, 7.27, 9.34, 11.78, 13.44, 15.99, 18.31, 23.21, 29.59)]


def chi_squared_prob(chi_squared, deg_freedom):
    if deg_freedom < 1:
        return 1.0
    if deg_freedom > 10:
        return None
    scores = chi_squareds[deg_freedom - 1]
    idx = bisect_left(scores, chi_squared, 0, len(scores))
    return confidences[idx]


def chi_squared_correlation(chi_squared, dim0, dim1, population):
    dim_factor = min(dim0, dim1) - 1
    if dim_factor == 0:
        return 0.0
    return math.sqrt(chi_squared / (population * dim_factor))


# TODO: interval correlation


def transformed_row(transforms, row):
    for col, trans in zip(row, transforms):
        if trans is not None:
            yield trans.get(col, col)


def apply_transforms(src_fname, tgt_fname, names, transforms,
                     limit=None, meter_period=default_meter_period):
    reader = csv.reader(open(src_fname))
    writer = csv.writer(open(tgt_fname, 'w'))
    reader.next()
    writer.writerow(names)
    meter = Meter(meter_period, 'total rows written: %d')
    for ridx, row in enumerate(reader):
        if limit is not None and ridx >= limit:
            break
        writer.writerow(tuple(transformed_row(transforms, row)))
        meter.inc(1)
    meter.log()


def mapfilter_cols(frame, op=lambda x: x, pred=lambda _: True):
    return [(frame.names[idx], op(col)) for idx, col in enumerate(frame.cols)
            if pred(col)]


def uniques_lteq(threshold):
    def pred(col):
        return len(col.uniques()) <= threshold
    return pred


def frame_uniques_lteq(frame, threshold=1):
    return mapfilter_cols(frame, pred=uniques_lteq(threshold))


def show_low_uniques(frame, threshold, detailed=False):
    matches = frame_uniques_lteq(frame, threshold)
    if detailed:
        for name, col in matches:
            print name, col.summary.cats
    else:
        names = zip(*matches)[0]
        print names
    print 'low uniques col count:', len(matches)


def show_anomalies(frame, ratio=None, coverage=None):
    hits, misses = frame.filter(ratio=ratio, coverage=coverage)
    print 'hits'
    for idx, summ in sorted(hits.iteritems()):
        print idx, summ.ratio, summ.coverage, len(summ.nums), summ.cats
    print
    print 'misses'
    for idx, summ in sorted(misses.iteritems()):
        print idx, summ.ratio, summ.coverage, summ.nums, len(summ.cats)
