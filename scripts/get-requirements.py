## under clip-as-service root dir
# python scripts/get-requirments.py $PIP_TAG /path/to/requirements.txt

import sys
from distutils.core import run_setup

result = run_setup('./server/setup.py', stop_after='init')

with open(sys.argv[2], 'w') as fp:
    fp.write('\n'.join(result.install_requires) + '\n')
    if sys.argv[1]:
        fp.write('\n'.join(result.extras_require[sys.argv[1]]) + '\n')
