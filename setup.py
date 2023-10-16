from setuptools import setup

from evalPM import __version__

with open('README.md') as f:
    long_desc = f.read()

setup(
    name='evalPM',
    version=__version__,

    url='https://github.com/db-tu-dresden/evalPM',
    author='Jonas Deepe, Lucas Woltmann',
    author_email='lucas.woltmann@tu-dresden.de',

    py_modules=['evalPM'],

    install_requires=['pandas','numpy','matplotlib','tensorflow','scikit-learn','deepsig'],

    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],

    description="A framework for creating and evaluating immission models for Particulate Matter",

    long_description=long_desc,

    long_description_content_type='text/markdown'
)
