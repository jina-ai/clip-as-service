__all__ = ['_py2', '_str', '_buffer', '_raise']

_py2 = False
_str = str
_buffer = memoryview


def _raise(t_e, _e):
    raise t_e from _e
