import inspect
import os
import sys

if __name__ == '__main__':
    if 'NO_VERSION_CHECK' not in os.environ:
        from .helper import is_latest_version

        is_latest_version(github_repo='clip-as-service')

    from jina import Flow

    f = Flow.load_config(
        'torch-flow.yml' if len(sys.argv) == 1 else sys.argv[1],
        extra_search_paths=[os.path.dirname(inspect.getfile(inspect.currentframe()))],
    )
    with f:
        f.block()
