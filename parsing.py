from __future__ import absolute_import

class ParseSuccess(object):
    def __init__(self, result, stream):
        self.result = result
        self.stream = stream
    def bind(self, cont): return cont(self.result)(self.stream)
    def case(self, success, failure):
        return success(self)


class ParseFailure(object):
    def __init__(self, message, stream):
        self.message = message
        self.stream = stream
    def bind(self, cont): return self
    def case(self, success, failure):
        return failure(self)


class Parser(object):
    def __init__(self, parse): self.parse = parse
    def __call__(self, stream): return self.parse(stream)

    def __ge__(self, cont):
        def parse(stream): return self(stream).bind(cont)
        return Parser(parse)

    def __mul__(self, p1):
        def c0(f0):
            def c1(r1): return pure(f0(r1))
            return p1 >= c1
        return self >= c0

    def __or__(self, p1):
        def parse(stream):
            def success(succ): return succ
            def failure(f0):
                def success(succ): return succ
                def failure(f1):
                    message = f0.message + " OR " + f1.message
                    return ParseFailure(message, f1.stream)
                return p1(stream).case(success, failure)
            return self(stream).case(success, failure)
        return Parser(parse)

    def __rshift__(self, p1):
        def ignore0(_): return p1
        return self >= ignore0

    def __lshift__(self, p1):
        def ignore1(r0):
            def c1(_): return pure(r0)
            return p1 >= c1
        return self >= ignore1


def pure(result):
    def success(stream): return ParseSuccess(result, stream)
    return Parser(success)


def fail(msg):
    def failure(stream): return ParseFailure(msg, stream)
    return Parser(fail)


def eos(result):
    def parse(stream):
        if stream == '': return ParseSuccess(result, '')
        return ParseFailure('expected end of string', stream)
    return Parser(parse)


def char_pred(err_msg, pred):
    def parse(stream):
        if stream and pred(stream[0]):
            return ParseSuccess(stream[0], stream[1:])
        return ParseFailure(err_msg, stream)
    return Parser(parse)


def char(ch):
    def pred(sch): return ch == sch
    return char_pred("expected character '%s'" % ch, pred)


def char_in(chs):
    def pred(sch): return sch in chs
    return char_pred('expected character from "%s"' % chs, pred)


def string(chs):
    parser = pure('')
    for ch in reversed(chs):
        def sappend(cs0):
            def sappend1(cs1): return cs0 + cs1
            return sappend1
        parser = pure(sappend) * char(ch) * parser
    return parser


def cons(r0):
    def cons1(rs): return [r0] + rs
    return cons1


def tupn(n, acc=()):
    if n == 0: return acc
    def recv(result):
        return tupn(n - 1, acc + (result,))
    return recv


def many0(parser):
    def parse(stream):
        do = pure(cons) * parser * many0(parser)
        def success(succ): return succ
        def failure(fail): return ParseSuccess([], stream)
        return do(stream).case(success, failure)
    return Parser(parse)


def many1(parser):
    return pure(cons) * parser * many0(parser)


def is_wspace(ch): return ch in ' \t\v\n\r'
wspace = many0(char_pred('expected whitespace', is_wspace))


def bracket(lch, rch, p0):
    return wspace >> char(lch) >> wspace >> p0 << wspace << char(rch)


digit = char_in('0123456789')

def to_int(result):
    try: return pure(int(''.join(result)))
    except ValueError, e: return fail(e)
nat = many1(digit) >= to_int

alphabet = ''.join(map(chr, range(ord('a'), 1 + ord('z'))))
alpha_lower = char_in(alphabet)
alpha_upper = char_in(alphabet.upper())
alpha = alpha_lower | alpha_upper
alnum = digit | alpha


# sgf example

#value_text_char = ((char('\\') >> char_pred('expected any character',
                                            #(lambda _: true)))
                   #| char_pred("expected non-']' character",
                               #(lambda ch: ch != ']')))
#value_text = pure(lambda result: ''.join(result)) * many0(value_text_char)
#prop_value = bracket('[', ']', value_text)
#prop_ident = pure(lambda result: ''.join(result)) * many1(alpha_upper)

#basic_property = pure(tupn(2, ('basic',))) * prop_ident * many1(prop_value)
#size_property = pure(tupn(1, ('size',))) * (string('SZ') >> bracket('[', ']', nat))

#point_letter = pure(lambda ch: 1 + (alphabet + alphabet.upper()).index(ch)) * alpha
#point_value_text = pure(tupn(2)) * point_letter * (pure(lambda idx: 20 - idx) * point_letter)
#point_value = bracket('[', ']', point_value_text)

#def add_prop(color):
    #return string('A' + color) >> (pure(tupn(1, ('add', color))) * many1(point_value))
#add_property = add_prop('E') | add_prop('B') | add_prop('W')
#def play_prop(color):
    #return string(color) >> (pure(tupn(1, ('play', color))) * point_value)
#play_property = play_prop('B') | play_prop('W')
#property = wspace >> (play_property | add_property | size_property | basic_property)

#node = wspace >> char(';') >> many0(property)
#sequence = many1(node)
#def parse_game_tree(stream):
    #return bracket('(', ')', pure(tupn(2, ('game-tree',))) * sequence * many0(game_tree))(stream)
#game_tree = Parser(parse_game_tree)
#collection = many1((game_tree << wspace) >= eos)

#sgf = '(;GM[1]FF[4]CA[UTF-8]AP[elm-goban:0]ST[2]SZ[19](;B[pj];W[im];B[kn](;W[oj];B[qo];W[ji];B[ek])(;W[ii];B[jj](;W[mg];B[mj])(;W[mk];B[mj])))(;B[jl];W[fm](;B[fo])(;B[ho])))'
#print collection(sgf).result
