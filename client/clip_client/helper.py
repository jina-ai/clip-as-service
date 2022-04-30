import json
import sys
import threading
from packaging.version import Version
from urllib.request import Request, urlopen

import pkg_resources
from rich import print
from rich.panel import Panel


def _version_check(package: str = None, github_repo: str = None):
    try:

        if not package:
            package = vars(sys.modules[__name__])['__package__']
        if not github_repo:
            github_repo = package

        cur_ver = Version(pkg_resources.get_distribution(package).version)
        req = Request(
            f'https://pypi.python.org/pypi/{package}/json',
            headers={'User-Agent': 'Mozilla/5.0'},
        )
        with urlopen(
            req, timeout=1
        ) as resp:  # 'with' is important to close the resource after use
            j = json.load(resp)
            releases = j.get('releases', {})
            latest_release_ver = max(
                Version(v) for v in releases.keys() if '.dev' not in v
            )
            if cur_ver < latest_release_ver:
                print(
                    Panel(
                        f'You are using [b]{package} {cur_ver}[/b], but [bold green]{latest_release_ver}[/] is available. '
                        f'You may upgrade it via [b]pip install -U {package}[/b]. [link=https://github.com/jina-ai/{github_repo}/releases]Read Changelog here[/link].',
                        title=':new: New version available!',
                        width=50,
                    )
                )
    except Exception:
        # no network, too slow, PyPi is down
        pass


def is_latest_version(package: str = None, github_repo: str = None) -> None:
    """Check if there is a latest version from Pypi, set env `NO_VERSION_CHECK` to disable it.

    :param package: package name if none auto-detected
    :param github_repo: repo name that contains CHANGELOG if none then the same as package name
    """

    threading.Thread(target=_version_check, args=(package, github_repo)).start()
