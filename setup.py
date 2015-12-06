from distutils.core import setup

setup(
    name='collapsed',
    version='0.1.0',
    author='tllake',
    author_email='thom.l.lake@gmail.com',
    packages=['collapsed', 'collapsed.emission'],
    description='Bayesian Hidden Markov Models and their Mixtures for Python.',
    long_description=open('README.md').read(),
)