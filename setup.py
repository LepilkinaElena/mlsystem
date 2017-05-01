from pip.req import parse_requirements
import os
import sys
from setuptools import setup, find_packages

os.chdir(os.path.dirname(os.path.abspath(__file__)))
req_file = "requirements.txt"

install_reqs = parse_requirements(req_file, session=False)

reqs = [str(ir.req) for ir in install_reqs]

setup(
	name = "mlsystem",
    description = "Train ML model",
    keywords = 'machine learning development llvm',
    long_description = """\
*mlsystem*
+++++

About
=====

TODO


Documentation
=============

TODO


Source
======

TODO
""",

    classifiers=[
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        ],

    zip_safe = False,

    packages = find_packages(),

    test_suite = 'tests.test_all',

    entry_points = {
        'console_scripts': [
            'mlsystem = mlsystem.tool:main',
            ],
        },
    install_requires=reqs,
)