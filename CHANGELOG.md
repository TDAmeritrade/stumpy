# 2020-06-15    [ 1.4.0 ]:
--------------------------
* bugfixes
  - Fixed bad chunking in `compute_mean_std`
* features
  - Added parallelized `scrump`
  - Added NaN/inf support to `scrump`
  - Added `prescrump` (and, consequently, scrump++)
  - Added AB-join to `scrump`
  - Added unit test for `scrump`
  - Added constrained inclusion motif search for `mstump`/`mstumped`
  - Added discord support for `mstump`/`mstumped`
  - Changed sorting to pure numpy in `mstump`/`mstumped` for better performance
  - Added NaN/inf support to `gpu_stump`
  - Added new `core.preprocess` function
  - Added NaN/inf support to `core.mass`
  - Added `core.apply_exclusion_zone` for consistent exclusion zone across functions
  - Added `stumpi` for incrementally updating matrix profiles with streaming data
  - Added `stumpi` unit tests
  - Added NaN/inf support for `stumpi`
  - Converted `floss.floss` generator to `class`
* tasks
  - Added Python 3.8 to Azure Pipelines for unit testing
  - Moved `stomp` to `_stomp` to prevent public usage
  - Fixed numerous typos
  - Added several `np.asarray` calls to input arrays
  - Split some unit tests out into separate files
  - Updated distributed teststo use context manager
  - Refactored `_calculate_squared_distance`
  - Added `core.py` to JIT-compiled unit test section
  - Remove mypy config file
  - Corrected cuda.jit signature for `gpu_stump`
  - Shortened time series length for faster GPU tests
  - Removed link to discourse from documentation
  - Added global variables for controlling chunking in `compute_mean_std`
  - Replaced name of `naive_mass` with `naive_stamp`
  - Removed redundant "caption" in RTD ToC
  - Renamed `stamp.mass` to private `stamp._mass_PI`
  - Added flake8-docstrings checking
  - Renamed `utils.py` to `naive.py` and updated corresponding function calls
  - Added `stumpi` and `scrump` to STUMPY API (RTD) 
* documentation
  - Initialized shapelet discovery tutorial (WIP)
  - Updated `check_window_size` docstring
  - Added `gpu_stump` to tutorial
  - Added `scrump` tutorial for Fast Approximate Matrix Profiles
  - Updated string formatting to conform to flake8
  - Added `stumpi` tutorial
  - Improved `mstump` tutorial (WIP)
  - Added additional references to original matrix profile papers

# 2020-03-27    [ 1.3.1 ]:
--------------------------
* bugfixes
  - Fixed MSTUMP/MSTUMPED input dimensions check
  - Fixed inconsistent MSTUMP/MSTUMPED output
* features
  - Added support for constant subsequences and added unit tests
  - Improved GPU memory consumption for self-join
  - Added ability to handle NaN and inf values in all matrix profile algorithms (except gpu_stump)
* tasks
  - Updated performance table with new performance results, better color scheme, intuitive hardware grouping
  - Re-organized ndarray input verification steps
  - Added more unit tests and rearranged test order
  - Removed Python type hints or type annotations
  - Split failing dask unit tests into multiple test files
  - Added PR template
  - Updated Mac OS X image for Azure Pipelines
  - Replaced stddev computation with a memory efficient rolling chunked stddev 
  - Modified exclusion zone to be symmetrical
  - Refactored multi-dimensional mass
  - Fixed scenarios where subsequence contains zero mean
  - Added explicit PR trigger to Azure Pipelines
  - Updated installation instructions to use conda-forge channel
  - Fixed time series chains all_c test to handle differences in Python set order
* documentation
  - Fixed various typos
  - Refactored tutorials for clarity

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
