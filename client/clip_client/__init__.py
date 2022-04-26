__version__ = '0.3.1'

from .helper import is_latest_version

is_latest_version(github_repo='clip-as-service')

from .client import Client
