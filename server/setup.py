from os import path

from setuptools import setup, find_packages

# setup metainfo
libinfo_py = path.join('bert_serving', 'server', '__init__.py')
libinfo_content = open(libinfo_py, 'r').readlines()
version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][0]
exec(version_line)  # produce __version__

setup(
    name='bert_serving_server',
    version=__version__,
    description='Mapping a variable-length sentence to a fixed-length vector using BERT model (Server)',
    url='https://github.com/hanxiao/bert-as-service',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    author='Han Xiao',
    author_email='artex.xh@gmail.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'numpy',
        'six',
        'pyzmq>=17.1.0',
        'GPUtil>=1.3.0',
        'termcolor>=1.1'
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.10.0'],
        'tensorflow_gpu': ['tensorflow-gpu>=1.10.0'],
    },
    classifiers=(
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ),
    scripts=[
        'bin/bert-serving-start',
    ],
    keywords='bert nlp tensorflow machine learning sentence encoding embedding serving',
)
