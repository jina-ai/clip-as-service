import sys
from os import path

from setuptools import find_packages
from setuptools import setup

if sys.version_info < (3, 7, 0):
    raise OSError(f'clip-as-service requires Python >=3.7, but yours is {sys.version}')

try:
    pkg_name = 'clip-client'
    libinfo_py = path.join(
        path.dirname(__file__), pkg_name.replace('-', '_'), '__init__.py'
    )
    libinfo_content = open(libinfo_py, 'r', encoding='utf8').readlines()
    version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][
        0
    ]
    exec(version_line)  # gives __version__
except FileNotFoundError as ex:
    __version__ = '0.0.0'

try:
    with open('../README.md', encoding='utf8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''

setup(
    name=pkg_name,
    packages=find_packages(),
    version=__version__,
    include_package_data=True,
    description='Embed images and sentences into fixed-length vectors via CLIP',
    author='Jina AI',
    author_email='hello@jina.ai',
    license='Apache 2.0',
    url='https://github.com/jina-ai/clip-as-service',
    download_url='https://github.com/jina-ai/clip-as-service/tags',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    zip_safe=False,
    setup_requires=['setuptools>=18.0', 'wheel'],
    install_requires=['jina>=3.2.10', 'docarray[common]>=0.10.3'],
    extras_require={
        'test': [
            'pytest',
            'pytest-timeout',
            'pytest-mock',
            'pytest-asyncio',
            'pytest-cov',
            'pytest-repeat',
            'pytest-reraise',
            'mock',
            'pytest-custom_exit_code',
            'black',
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Unix Shell',
        'Environment :: Console',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Database :: Database Engines/Servers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Internet :: WWW/HTTP :: Indexing/Search',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Multimedia :: Video',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    project_urls={
        'Documentation': 'https://clip-as-service.jina.ai',
        'Source': 'https://github.com/jina-ai/clip-as-service/',
        'Tracker': 'https://github.com/jina-ai/clip-as-service/issues',
    },
    keywords='jina openai clip deep-learning cross-modal multi-modal neural-search',
)
