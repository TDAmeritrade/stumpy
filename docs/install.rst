------------
Installation
------------

Using conda/pip
===============

Conda install (preferred):

.. code:: bash

    conda install -c conda-forge stumpy

PyPI install with ``pip``:

.. code:: bash

    pip install stumpy


From source
===========

To install stumpy from source, you'll need to install the dependencies above. For maximum performance, it is recommended that you install all dependencies using `conda`:

.. code:: bash

    conda install -y numpy scipy numba

Alternatively, but with lower performance, you can also install these dependencies using the requirements.txt file in the root of this repository:

.. code:: bash

    pip install -r requirements.txt

Once the dependencies are installed (stay inside of the ``stumpy`` directory), execute:

.. code:: bash

    python setup.py install
