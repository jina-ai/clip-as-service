__all__ = ['_py2', '_str', '_buffer', '_unicode']

_py2 = True
_str = basestring
_buffer = buffer


def _unicode(x):
    return x if isinstance(x, unicode) else x.decode('utf-8')
