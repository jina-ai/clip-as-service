import sys
import os
import inspect
from jina import Flow

if __name__ == '__main__':
    f = Flow.load_config(
        'torch-flow.yml' if len(sys.argv) == 1 else sys.argv[1],
        extra_search_paths=[os.path.dirname(inspect.getfile(inspect.currentframe()))],
    )
    with f:
        f.block()
