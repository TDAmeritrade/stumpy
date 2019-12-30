# 2019-12-30    [ 1.3.0 ]:
--------------------------
* bugfixes
  - Fixed MSTUMP/MSTUMPED input dimensions check
  - Fixed inconsistent MSTUMP/MSTUMPED output
* features
  - Added parallel GPU-STUMP (i.e., multi-GPU support) using Python multiprocessing and file I/O
  - Added Python type hints/type annotations
* tasks
  - Updated performance table and plots with STUMPY.2, GPU-STUMP.1, GPU-STUMP.2, GPU-STUMP.DGX1, and GPU-STUMP.DGX2 results
  - Fixed test function names
  - Added Python script for easier performance timing conversion
* documentation
  - Added window size (m = 50) for performance calculations
  - Fixed various typos
  - Added missing and improved docstrings
  - Replaced Bokeh with Matplotlib
  - Updated GPU-STUMP example with multi-GPU support

# 2019-12-03    [ 1.2.4 ]:
--------------------------
* features
  - Added ability to select GPU device in gpu_stump
* tasks
  - Changed all README hyperlinks to double underscores
  - Added API links to README
  - Added STUMPY circle image (logo)
  - Suppressed pytest junit_family deprecation warning
  - Replaced `python install` with `python -m pip install .`

# 2019-11-26    [ 1.2.3 ]:
--------------------------
* bugfixes
  - Fixed incorrect GPU output for self joins
* features
  - Added array check for NaNs
  - Improved test script for CI
* tasks
  - Added discourse group at stumpy.discourse.group
  - Updated formatting and added newline before logo
  - Made logo a hyperlink to Github repo
  - Added custom CSS to sphinx theme

# 2019-11-03    [ 1.2.2 ]:
--------------------------
* bugfixes
  - Fixed Python AST utf8 file reading bug on Windows

# 2019-11-03    [ 1.2.1 ]:
--------------------------
* bugfixes
  - Fixed driver not found function when no GPU present
  - Fixed gpu_stump docstring in RTD

# 2019-11-02    [ 1.2.0 ]:
--------------------------
* bugfixes
  - Fixed transposed Pandas DataFrame issue #66
* features
  - Added GPU support (NVIDIA Only)
* tasks
  - Added CHANGELOG
  - Added GPU tests for GPU-STUMP
  - Added CPU tests (via CUDA simulator) for GPU-STUMP
  - Added additional STUMPY logos
  - Disabled bidirectional and left indices for FLOSS
  - Added STUMPY to Python 3 Statement
* documentation
  - Added reference to semantic segmentation
  - Added gpu_stump API to RTF
  - Added gpu_stump example to README
  - Improved docstring for mstump(ed)
  - Improved time series chains tutorial
  - Added FLUSS to README
  - Added missing docstring for FLOSS
  - Added FLOSS and FLUSS to RTD API
  - Updated docstring DOIs with URLs to primary references
  - Added motif discovery to exisiting tutorial

# 2019-08-03    [ 1.1.0 ]:
--------------------------
* bugfixes
  - Removed incorrect compatibility with Py35
  - Split Numba JIT tests tests to ensure proper Dask PyTest teardown 
* features
  - Added FLUSS and FLOSS
  - Added Pandas Series/DataFrame support
* tasks
  - Added conda-forge support
  - Set up CI with Azure Pipelines
  - Added black and flake8 checks
  - 100% test coverage
  - Added badges
  - Added __version__ attr
  - Added performance graph
  - Published in JOSS
  - Added STUMPY logo suite
  - Added CHANGELOG
* documentation
  - Added RTD documentation
  - Added Tutorials
  - Added Binder support
  - Added NABDConf presentation

# 2019-05-12    [ 1.0.0 ]:
--------------------------
* Initial release
