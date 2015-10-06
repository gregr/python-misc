from __future__ import absolute_import
from bisect import bisect_left
import csv
from itertools import chain, izip_longest
import logging
import math
from .dict import countdict, countdict_setops
from .logging import Meter

default_meter_period = 10000


def cross(*xss):
    results = [()]
    for xs in xss:
        results = [result + (x,) for result in results for x in xs]
    return results


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
        self.refresh()

    def enable_interval(self):
        self.nums = countdict()
        for key, count in self.cats.items():
            try:
                self.nums[float(key)] = count
                del self.cats[key]
            except ValueError:
                pass
        self.refresh()

    def disable_interval(self):
        for key, count in self.nums.iteritems():
            self.cats[str(key)] = count
        self.nums = countdict()
        self.refresh()

    def refresh(self):
        self.sum = 0
        self.count = 0
        for num, count in self.nums.iteritems():
            self.count += count
            self.sum += num * count
        num_keycount = float(len(self.nums))
        total_keycount = (num_keycount + len(self.cats))
        if total_keycount > 0:
            self.ratio = num_keycount / float(total_keycount)
        else:
            self.ratio = 0.0
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
        remapping = dict((src, tgt) for src, tgt in self.move.iteritems()
                         if src != tgt)
        return remapping

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
            if ((coverage is not None and summ.coverage < coverage) or
                    (ratio is not None and summ.ratio < ratio)):
                misses[name] = summ
                continue
            hits[name] = summ
        return hits, misses

    def transforms(self):
        name_to_remapping = dict((name, col.transforms())
                                 for name, col in zip(self.names, self.cols))
        return self.removed, name_to_remapping

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


def process_csv(file_name, make_observe, names=None, limit=None,
                meter_period=default_meter_period):
    logging.info('processing: %s; names=%s limit=%s', file_name, names, limit)
    reader = csv.reader(open(file_name))
    if names is None:
        names = reader.next()
    observe = make_observe(names)
    meter = Meter(meter_period, 'total rows processed: %d')
    for ridx, row in enumerate(reader):
        if limit is not None and ridx >= limit:
            break
        if observe(ridx, row):
            break
        meter.inc(1)
    meter.log()
    logging.info('finished processing: %s', file_name)


def summarize(file_name, *args, **kwargs):
    logging.info('summarizing: %s', file_name)
    state = [None, None]

    def make_observe(names):
        freqs = [countdict() for _ in names]
        state[0] = names
        state[1] = freqs

        def observe(ridx, row):
            for cidx, col in enumerate(row):
                freqs[cidx][col] += 1
        return observe
    process_csv(file_name, make_observe, *args, **kwargs)
    names, freqs = state
    frame = Frame(names, [Column(ColumnSummary(freq)) for freq in freqs])
    logging.info('finished summarizing: %s', file_name)
    return frame


def filter_rows(file_name, make_pred, max_results=None, **kwargs):
    logging.info('filtering: %s', file_name)
    found = []

    def make_observe(names):
        pred = make_pred(names)

        def observe(ridx, row):
            if pred(row):
                found.append((ridx, row))
                if max_results is not None and len(found) >= max_results:
                    return True
        return observe
    process_csv(file_name, make_observe, **kwargs)
    logging.info('finished filtering: %s', file_name)
    return found


def col_match_pred(col_targets):
    def make_pred(names):
        matchers = []
        for idx, name in enumerate(names):
            tgt = col_targets.get(name)
            if tgt is not None:
                matchers.append((idx, tgt))

        def pred(row):
            for idx, tgt in matchers:
                if row[idx] != tgt:
                    return False
            return True
        return pred
    return make_pred


def tuplepair_freqs(file_name, col_tuple_pairs, *args, **kwargs):
    logging.info('computing frequencies in %s of pairs: %s',
                 file_name, col_tuple_pairs)
    freqss = [countdict() for _ in col_tuple_pairs]

    def make_observe(names):
        name_to_index = dict((name, idx) for idx, name in enumerate(names))
        idxss = [([name_to_index[col0] for col0 in cols0],
                  [name_to_index[col1] for col1 in cols1])
                 for cols0, cols1 in col_tuple_pairs]

        def observe(ridx, row):
            for (idxs0, idxs1), freqs in zip(idxss, freqss):
                t0 = tuple(row[idx0] for idx0 in idxs0)
                t1 = tuple(row[idx1] for idx1 in idxs1)
                freqs[(t0, t1)] += 1
        return observe
    process_csv(file_name, make_observe, *args, **kwargs)
    for freqs in freqss:
        ks0, ks1 = map(set, zip(*freqs.iterkeys()))
        for pair in cross(ks0, ks1):
            freqs[pair] += 0
    logging.info('finished computing frequencies in %s of pairs: %s',
                 file_name, col_tuple_pairs)
    return freqss


def pearson_correlation_with(file_name, col_pairs, *args, **kwargs):
    logging.info('computing pearson correlation in %s of pairs: %s',
                 file_name, col_pairs)
    state = [[0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in col_pairs]

    def make_observe(names):
        name_to_index = dict((name, idx) for idx, name in enumerate(names))
        idxss = [(name_to_index[col0], name_to_index[col1])
                 for col0, col1 in col_pairs]

        def observe(ridx, row):
            for idx0, idx1 in idxss:
                try:
                    x = float(row[idx0])
                    y = float(row[idx1])
                    state[0] += 1
                    state[1] += x
                    state[2] += y
                    state[3] += x * x
                    state[4] += y * y
                    state[5] += x * y
                except ValueError:
                    pass
        return observe
    process_csv(file_name, make_observe, *args, **kwargs)
    pair_sums = state
    r = [((n * Sxy - Sx * Sy) /
          math.sqrt((n * Sx2 - (Sx ** 2)) * (n * Sy2 - (Sy ** 2))))
         for n, Sx, Sy, Sx2, Sy2, Sxy in pair_sums]
    logging.info('finished computing pearson correlation in %s of pairs: %s',
                 file_name, col_pairs)
    return r


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
    return result, deg_freedom, (len(ks0), len(ks1), total)


def chi_squared_with(*args, **kwargs):
    return map(chi_squared, tuplepair_freqs(*args, **kwargs))

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


def chi_squared_prob_correlation(chi_squared, deg_freedom, dims_pop):
    return (chi_squared_prob(chi_squared, deg_freedom),
            chi_squared_correlation(chi_squared, *dims_pop))


def names_and_remappings(current_names, removed_names, name_to_remapping):
    name_to_remapping = name_to_remapping.copy()
    name_to_idx = dict((name, idx) for idx, name in enumerate(current_names))
    name_to_idx
    remappings = [{}] * len(current_names)
    for name in removed_names:
        name_to_remapping[name] = None
    for name, remapping in name_to_remapping.iteritems():
        idx = name_to_idx.get(name)
        if idx is not None:
            remappings[idx] = remapping
    new_names = [name for name, rmap in zip(current_names, remappings)
                 if rmap is not None]
    return new_names, remappings


def transformed_row(remappings, row):
    for col, remap in zip(row, remappings):
        if remap is not None:
            yield remap.get(col, col)


def apply_transforms(src_fname, tgt_fname, removed_names, name_to_remapping,
                     current_names=None, limit=None,
                     meter_period=default_meter_period):
    remapped_names = sorted(
        name for name, remap in name_to_remapping.iteritems()
        if len(remap) != 0)
    logging.info("transforming '%s' to '%s'", src_fname, tgt_fname)
    logging.info('removing columns: %s', sorted(removed_names))
    logging.info('modifying columns: %s', remapped_names)
    reader = csv.reader(open(src_fname))
    writer = csv.writer(open(tgt_fname, 'w'))
    if current_names is None:
        current_names = reader.next()
    new_names, remappings = names_and_remappings(
        current_names, removed_names, name_to_remapping)
    writer.writerow(new_names)
    meter = Meter(meter_period, 'total rows written: %d')
    for ridx, row in enumerate(reader):
        if limit is not None and ridx >= limit:
            break
        writer.writerow(tuple(transformed_row(remappings, row)))
        meter.inc(1)
    meter.log()
    logging.info("finished transforming '%s' to '%s'", src_fname, tgt_fname)


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
    elif matches:
        names = zip(*matches)[0]
        print names
    print 'low uniques col count:', len(matches)


def show_anomalies(frame, ratio=None, coverage=None):
    hits, misses = frame.filter(ratio=ratio, coverage=coverage)
    cats = [(name, summ) for name, summ in sorted(misses.iteritems())
            if len(summ.nums == 0)]
    num_misses = [(name, summ) for name, summ in sorted(misses.iteritems())
                  if len(summ.nums > 0)]
    print 'cats'
    for name, summ in cats:
        print name, summ.ratio, summ.coverage, summ.nums, len(summ.cats)
    print
    print 'num hits'
    for name, summ in sorted(hits.iteritems()):
        print name, summ.ratio, summ.coverage, len(summ.nums), summ.cats
    print
    print 'num misses'
    for name, summ in num_misses:
        print name, summ.ratio, summ.coverage, len(summ.nums), len(summ.cats)
