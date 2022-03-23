import sys

from jina import Flow

if __name__ == '__main__':
    f = Flow.load_config('torch-flow.yml' if len(sys.argv) == 1 else sys.argv[1])
    with f:
        f.block()
