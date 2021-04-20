------------
Installation
------------

Supported Python and NumPy versions are determined according to the `NEP 29 deprecation policy <https://numpy.org/neps/nep-0029-deprecation_policy.html>`__.

Using conda/pip
===============

Conda install (preferred):

.. code:: bash

    conda install -c conda-forge stumpy

PyPI install with ``pip``:

.. code:: bash

    python -m pip install stumpy


From source
===========

To install stumpy from source, first clone the source repository:

.. code:: bash

    git clone https://github.com/TDAmeritrade/stumpy.git
    cd stumpy

Next, you'll need to install the necessary dependencies. For maximum performance(or if you are installing stumpy for the Apple M1 ARM-based chip), it is recommended that you install all dependencies using `conda`:

.. code:: bash

    conda install -c conda-forge -y numpy scipy numba

Alternatively, but with lower performance, you can also install these dependencies using the requirements.txt file in the root of this repository:

.. code:: bash

    python -m pip install -r requirements.txt

Once the dependencies are installed (stay inside of the ``stumpy`` directory), execute:

.. code:: bash

    python -m pip install .
