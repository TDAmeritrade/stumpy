from setuptools import setup

def readme():
    with open('README.rst') as readme_file:
        return readme_file.read()

configuration = {
    'name' : 'stumpy',
    'version' : '0.1',
    #'python_requires=' : >=3.5',

    'author' : 'Sean M Law',
    'author_email' : 'seanmylaw@gmail.com',
    'description' : 'A powerful and scalable technique that can be used for time series data mining',
    'long_description' : readme(),
    'classifiers' : [
        'Development Status :: 1 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.5',
    ],
    'keywords' : 'time series matrix profile motif discord',
    'url' : 'http://github.com//',
    'maintainer' : 'Sean M. Law',
    'maintainer_email' : 'seanmylaw@gmail.com',
    'license' : 'BSD',
    'packages' : ['stumpy'],
    'install_requires': ['numpy >= 1.13',
                         'scipy >= 1.2.1',
                         'numba >= 0.42.0'],
    'ext_modules' : [],
    'cmdclass' : {},
    'tests_require' : ['pytest'],
    'data_files' : ()
    }

setup(**configuration)
