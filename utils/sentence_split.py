import html
import re


class SentenceSplitter:
    def __init__(self, min_len=5, max_len=20):
        self.min_len = min_len
        self.max_len = max_len
        self.must_split = r"[^.。！？!?]*[.。！？!?]"
        self.maybe_split = r"[^,，、.。:：;；(（)）]*[,，、.。:：;；(（)）]"
        self.html_symbol = {
            '&gt', '&nbsp'
        }

    def _is_ascii(self, s):
        return len(s) == len(s.encode())

    def _get_printable(self, p):
        result = html.unescape(p)
        result = ''.join(c for c in result if c.isprintable())
        return result

    def _check_p(self, p):
        if (len(p) > self.min_len and  # must longer
                not self._is_ascii(p) and  # must not all english
                len(re.findall('\s', p)) == 0 and  # must not contain spaces -> likely spam
                '\\x' not in p):  # must not contain bad unicode char
            return True

    def _split(self, p, reg):
        matches = re.finditer(reg, self._get_printable(p.strip()))
        sent = [match.group() for matchNum, match in enumerate(matches)]
        for s in sent:
            s = s.strip()
            if self._check_p(s):
                if len(s) > self.max_len and reg != self.maybe_split:
                    for ss in self._split(s, self.maybe_split):
                        yield ss
                else:
                    yield s

    def split(self, p):
        return self._split(p, self.must_split)
