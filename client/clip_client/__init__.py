__version__ = '0.3.5'

import os

from .client import Client

if 'NO_VERSION_CHECK' not in os.environ:
    from .helper import is_latest_version

    is_latest_version(github_repo='clip-as-service')
