import html
import re


class SentenceSplitter:
    def __init__(self, min_len=2, max_len=20):
        self.min_len = min_len
        self.max_len = max_len
        self.must_split = r"[.。！？!?]+"
        self.maybe_split = r"[,，、.。:：;；(（)）\s]+"

    def _is_ascii(self, s):
        return len(s) == len(s.encode())

    def _get_printable(self, p):
        p = html.unescape(p)  # decode html entity
        p = ''.join(c for c in p if c.isprintable())  # remove unprintable char
        p = ''.join(p.split())  # remove space
        return p

    def _check_p(self, p):
        if (len(p) > self.min_len and  # must longer
                not self._is_ascii(p) and  # must not all english
                # len(re.findall('\s', p)) == 0 and  # must not contain spaces -> likely spam
                '\\x' not in p):  # must not contain bad unicode char
            return True

    def _split(self, p, reg):
        sent = [s for s in re.split(reg, self._get_printable(p)) if s.strip()]
        for s in sent:
            if self._check_p(s):
                if len(s) > self.max_len and reg != self.maybe_split:
                    for ss in self._split(s, self.maybe_split):
                        yield ss
                else:
                    yield s

    def split(self, p):
        return self._split(p, self.must_split)


if __name__ == '__main__':
    ss = SentenceSplitter()
    print(list(ss.split('所述太阳能,   光伏发电,形  或。我是谁，谁是我，电位器，而我，的任务人分为2，所述太阳能,光伏发电,形或。我是谁，谁是我，电位器，而我，的任务人分为2')))
