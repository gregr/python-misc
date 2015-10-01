from collections import defaultdict
import csv
from itertools import chain


def validate_sanity(file_name):
    counts = set()
    for row in csv.reader(open(file_name)):
        counts.add(len(row))
    return len(counts) <= 1


class ColumnSummary(object):
    def __init__(self, nums, cats):
        self.cats = cats
        self.nums = nums

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


def col_summary(freqs):
    num_freqs = {}
    for key, count in freqs.items():
        try:
            num_freqs[float(key)] = count
            del freqs[key]
        except ValueError:
            pass
    return ColumnSummary(num_freqs, freqs)


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
                               source,
                               target)
        orig = self.moved_from[source]
        del self.moved_from[source]
        self.moved_from[target] = orig
        self.move[orig] = target
        count = self.summary[source]
        del self.summary[source]
        self.summary[target] += count


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


def summarize(file_name, header=None, limit=None):
    col_name_to_index = {}
    reader = csv.reader(open(file_name))
    if header is None:
        header = reader.next()
    for idx, key in enumerate(header):
        col_name_to_index[key] = idx
    freqs = [defaultdict(int) for _ in header]
    for ridx, row in enumerate(reader):
        if limit is not None and ridx >= limit:
            break
        for cidx, col in enumerate(row):
            freqs[cidx][col] += 1
    frame = Frame(header, [Column(col_summary(freq)) for freq in freqs])
    return frame


# TODO: union, diff, intersection
