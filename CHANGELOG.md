# 2021-01-19    [ 1.7.1 ]:
--------------------------
* bugfixes
  - None
* features
  - None
* tasks
  - Bumped minimum NumPy version to use `np.array_equal`
* documentation
  - None


# 2021-01-17    [ 1.7.0 ]:
--------------------------
* bugfixes
  - None
* features
  - Added maximum window size check
  - Added window size checking to preprocessing
* tasks
  - Replaced array comparison in `core.are_arrays_equal` with `np.array_equal`
  - Added Github Actions and badge
  - Improved stability for integration with other packages
* documentation
  - Fixed typo
  - Replaced NABDConf motivation video with PyData Global video


# 2020-12-30    [ 1.6.1 ]:
--------------------------
* bugfixes
  - Fixed inconsistent `mstump`/`mstumped` output to match description in published work
* features
  - Added `subspace` function for compute multi-dimensional matrix profile subspace
  - Added `include` and `discords` handling to `subspace`
  - Added `mdl` building blocks
  - Added `ostinato` Dask distributed, GPU, and `aamp` variants
  - Added fast rolling min/max functions
* tasks
  - Updated Azure Pipelines CI coverage tests to use latest images
  - Added `mstump` tutorial to Binder
  - Fixed bad reference to `aamp_gpu_ostinato`
  - Fixed `aamp_ostinato` unit test coverage
  - Converted `conda.sh` to `bash`
  - Moved all private functions out of `__init__.py`
  - Added commands to remove `build/`, `dist/`, and `stumpy.egg/` after PyPI upload
  - Added twine to environment setup script
  - Fixed incorrect date in `CHANGELOG.md`
* documentation
  - Added `mstump` tutorial to RTD
  - Updated various function docstrings
  - Added Github Discussions to README and RTD
  - Updated API list on RTD


# 2020-12-10    [ 1.6.0 ]:
--------------------------
* bugfixes
  - Fixed incorrect cancelling of Dask data that was being scattered with `hash=False`
  - Fixed floating point imprecision in computing distances with `mass_absolute`
* features
  - Added `ostinato` function for computing consensus motifs
  - Added new approach for retrieving the most central consensus motif from `ostinato`
  - Added `mpdist` function for computing the MPdist distance measure
  - Added `mpdisted` function for computing the MPdist distance measure
  - Added `gpu_mpdist` function for computing the MPdist distance measure
  - Added `aampdist` function for computing the MPdist distance measure
  - Added `aampdisted` function for computing the MPdist distance measure
  - Added `gpu_aampdist` function for computing the MPdist distance measure
  - Changed `np.convolve` to the faster `scipy.signal.convolve` for FFT
  - Added matrix profile subspace for multi-dimensional motif discovery
  - Replaced existing rolling functions with fast Welford nanstd and nanvar functions
* tasks
  - Updated Azure Pipelines CI to use latest images
  - Fixed tutorial typos
  - Added ostinato paper to README References
  - Added `mamba` to speed up Python environment installation
  - Added `ostinato` tutorial
  - Added a series of new unit tests (maintained at 100% test coverage)
  - Updated all tutorials to use Zenodo links for data retrieval
  - Removed tutorial plotting function that set default plotting conditions
  - Added AB-join tutorial
  - Replaced rolling window isfinite with a much faster function
  - Updated Github PR template to use conda-forge channel
  - Updated Welford corrected-sum-of-squares derivation
  - Updated Binder environment to use STUMPY release in Binder postBuild
* documentation
  - Fixed GPU function signatures that were being displayed on RTD
  - Fixed incorrect docstring indentation
  - Added STUMPY docs and Github code repo to Resources section of tutorials
  - Added default values to docstrings


# 2020-10-19    [ 1.5.1 ]:
--------------------------
* bugfixes
  - Fixed AB-join so that it now matches the published definition (previously, BA-join)
* features
  - Added nan/inf support to FLOSS/FLUSS
* tasks
  - Removed Pandas series in GPU tests to improve unit test speed in CI
  - Identify operating system prior to installing cuda toolkit
  - Changed `left`/`right` keywords in all unit tests to `ref`/`comp`
  - Split up unit tests and coverage testing 
  - Updated `displayNames` in Azure Pipelines
  - `test.sh` now accepts `unit`, `custom`, and `coverage` keywords 
  - Fixed typos
  - Added pattern searching (MASS) tutorial
  - Added `Contribute` notebook to RTD table of contents for first time contributors
  - Refactored `_compute_diagonal` for speed improvements
  - Replaced `np.roll` with slice indexing in `stumpy.floss`
  - Refactored and improved `aampi` and `stumpi` update performance
  - Added `lxml` to environment.yml  
* documentation
  - Added `plt.show()` to code figures in tutorials
  - Updated `stumpi` tutorial with `egress=False`


# 2020-08-31    [ 1.5.0 ]:
--------------------------
* bugfixes
  - Fixed warning and check when time series has inappropriate dtype
  - Fixed scenarios where identical subsequences produce non-zero distances
* features
  - For interactive data science work, matrix profile calcs are 10-15x faster
  - Added `aamp` with non-normalized Euclidean distance (i.e., no z-normalization)
  - Added `aamped`
  - Added `aampi`
  - Added `gpu_aamp`
  - Added egress for `stumpi` and egress is now the default behavior
  - Added a `mass_absolute` function for non-normalized distance calculation with FFT convolution
  - Added diagonal pre-processing function to `core.py`
  - Added centered-sum-of-products and Pearson correlation in place of sliding dot products
  - Added left and right matrix profile indices to `scrump` and converted to Pearson correlation
* tasks
  - Removed Pandas series in GPU tests to improve unit test speed in CI
  - Updated to latest version of black for better formatting
  - Refactored redundant test section
  - Added unit test for inappropriate dtype inputs
  - Corrected absolute stumpy import to be relative import
  - Replaced `._illegal` attribute with a more obvious `._T_isfinite` attribute
  - Moved common diagonal functions to `core.py`
  - Replaced `order` variable with the more obvious `diag` name
  - Added environment.yml for easier installation of dependencies
  - Removed random print statement in code
  - Moved STUMPY thresholds to global parameters in `config.py`
  - Refactored left/right matrix profile indices
  - Refactored NaN checking
  - Check for Linux OS and add TBB dynamically especially for CI
* documentation
  - Added `aamp` reference to README
  - Update docstrings to be less verbose for API documentation
  - Fixed some typos
  - Replaced `sep="\s+"` with `sep="\\s+"` in tutorials
  - Added notes and derivations for Pearson correlation and centered-sum-of-products
  - Renamed tutorials with underscores for consistency
  - Added all `aamp`-like functions to API reference
  - Replaced MS Word docs with LaTeX notebooks


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
