from os import path

from setuptools import setup, find_packages

# setup metainfo
libinfo_py = path.join('bert_serving', 'client', '__init__.py')
libinfo_content = open(libinfo_py, 'r').readlines()
version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][0]
exec(version_line)  # produce __version__

with open('requirements.txt') as f:
    require_packages = [line[:-1] if line[-1] == '\n' else line for line in f]

setup(
    name='bert_serving_client',
    version=__version__,  # noqa
    description='Mapping a variable-length sentence to a fixed-length vector using BERT model (Client)',
    url='https://github.com/hanxiao/bert-as-service',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    author='Han Xiao',
    author_email='artex.xh@gmail.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=require_packages,
    classifiers=(
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ),
    keywords='bert nlp tensorflow machine learning sentence encoding embedding serving',
)
