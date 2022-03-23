import os
import sys

__resources_path__ = os.path.join(
    os.path.dirname(
        sys.modules.get('clip_server').__file__
        if 'clip_server' in sys.modules
        else __file__
    ),
    'resources',
)


def cli_entrypoint():
    print('hello')
