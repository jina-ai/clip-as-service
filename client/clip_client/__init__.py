__version__ = '0.4.18'

import os

from clip_client.client import Client

if 'NO_VERSION_CHECK' not in os.environ:
    from clip_server.helper import is_latest_version

    is_latest_version(github_repo='clip-as-service')
