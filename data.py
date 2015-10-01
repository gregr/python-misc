import csv
from itertools import chain
from .dict import countdict, countdict_setops


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

    def __getitem__(self, name):
        self.cols[self.name_to_index[name]]

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

    def zipmap(self, op, frame):
        cols = [op(c0, c1) for c0, c1 in zip(self.cols, frame.cols)]
        return Frame(self.names, cols)

    def union(self, frame):
        return self.zipmap(Column.union, frame)

    def difference(self, frame):
        return self.zipmap(Column.difference, frame)

    def intersection(self, frame):
        return self.zipmap(Column.intersection, frame)

    def difference_symmetric(self, frame):
        return self.zipmap(Column.difference_symmetric, frame)


def summarize(file_name, header=None, limit=None):
    col_name_to_index = {}
    reader = csv.reader(open(file_name))
    if header is None:
        header = reader.next()
    for idx, key in enumerate(header):
        col_name_to_index[key] = idx
    freqs = [countdict() for _ in header]
    for ridx, row in enumerate(reader):
        if limit is not None and ridx >= limit:
            break
        for cidx, col in enumerate(row):
            freqs[cidx][col] += 1
    frame = Frame(header, [Column(ColumnSummary(freq)) for freq in freqs])
    return frame


def transformed_row(transforms, row):
    for col, trans in zip(row, transforms):
        if trans is not None:
            yield trans.get(col, col)


def apply_transforms(src_fname, tgt_fname, names, transforms, limit=None):
    reader = csv.reader(open(src_fname))
    writer = csv.writer(open(tgt_fname, 'w'))
    reader.next()
    writer.writerow(names)
    for ridx, row in enumerate(reader):
        if limit is not None and ridx >= limit:
            break
        writer.writerow(tuple(transformed_row(transforms, row)))


def is_constant(col):
    return len(tuple(col.summary.iterkeys())) <= 1


def constant_cols(frame):
    return [(frame.names[idx], col) for idx, col in enumerate(frame.cols)
            if is_constant(col)]


def show_anomalies(frame, ratio=None, coverage=None):
    hits, misses = frame.filter(ratio=ratio, coverage=coverage)
    print 'hits'
    for idx, summ in sorted(hits.iteritems()):
        print idx, summ.ratio, summ.coverage, len(summ.nums), summ.cats
    print
    print 'misses'
    for idx, summ in sorted(misses.iteritems()):
        print idx, summ.ratio, summ.coverage, summ.nums, len(summ.cats)


# TODO: correlation
