# 2024-07-08    [ 1.13.0 ]:
---------------------------
* bugfixes
  - Fixed Ostinato overwriting original time series
* features
  - Added Ray support
  - Added `numpy` 2.0 support
  - Added named attributes to matrix profile array
  - Added Python 3.12 support
  - Migrated setup.py, setuptools to pyproject.toml
* tasks
  - Added version mismatch checker
  - Added `copy` param to `preprocess`-related functions
  - Disabled bokeh dashboard in dask
  - Added `numba` channel to environment.yml
  - Replace `np.INF` with `-np.infg`
  - Fixed inability to import packages in tutorials
  - Added matplotlib to RTD requirements
  - Removed unnecessary comments in code
  - Added `tests/__pycache__` to `clean_up` process
  - Added option to display current dev environment in `test.sh`
  - Added keyword `test.sh` to only execute `gpu` containing tests
  - Added "upgrade pip" to Github Actions workflow
  - Removed Twitter and Zenodo badges, added NumFOCUS badge
  - Added Github Discussions link
  - Updated codecove version for Github Actions
  - Removed `pkg_resources` as a dependency
  - Added codeowners file
* documentation
  - Improved syntax highlighting
  - Updated class docstrings
  - Removed napoleon extension
  - Switched to Myst
  - Relocated notebooks/tutorials
  - Fixed named attributes being displayed incorrectly
  - Fixed typos


# 2023-08-20    [ 1.12.0 ]:
---------------------------
* bugfixes
  - Fixed prescrump, scrump, and scraamp miscalculation of AB-joins
  - Fixed bug in `is False` by converting to `bool` type
  - Fixed bug in `snippet` that caused loss in precision, add unit tests
  - Fixed RTD incompatibility with urllib3
  - Fixed loss of precision in distances computed for self-matches
  - Fixed libiomp5.dylib Github Actions location
  - Fixed incorrect post-processing in naive.mass_PI
  - Fixed incorrect number of contiguous windows in snippets
* features
  - Improved matrix profile performance (15-20%) with uint64 indexing
  - Added top-k nearest neighbor feature
  - Added rolling_isconstant function
  - Added `subseq_isconstant` to API for transparent handling of constant time series subsequences
  - Added parallelized rolling_nanstd
  - Added abstraction layer for distributed client functions (e.g., Dask, Ray, etc)
  - Added initial support for `numba` function caching
  - Added Python 3.11 support
  - Added `query_idx` to improve distances computed for self-matches in `motifs` function
  - Added `mmotifs` for multi-dimensional motif discovery
  - Added function `process_isconstant`
* tasks
  - Refactored mpdist
  - Added MERLIN notebook reproducer
  - Added negative index checks to mmotifs
  - Fixed mybinder badge and links
  - Renamed nave.mass to naive.mass_PI
  - Added `row_wise` parameter to naive.stump
  - Added input type (list) check to ostinato
  - Added ability to return fully filled bfs indices
  - Updated setup.py to enable Github dependency graph tracking
  - Improved stability of prescrimp
  - Split coverage tests for more verbosity
  - Added more features to custom test function
  - Removed explicit cancellation of dask futures
  - Removed max `dask`/`distributed` version requirement
  - Optimized stumpi and aampi class init
  - Improve aampi update behavior with constant sequences returning nan
  - Added pytest notebook link checking
  - Refactored `match` function
  - Added warnings to motifs and aamp_motifs
  - Added `numba -s` step in Github Actions workflow
  - Added explicit link to OpenMP for MacOS Github Actions workflow
  - Moved to Actions/Checkout V3 in Github Actions
  - Improved numba function signatures
  - Added '--editable' install mode to setup.sh
  - Updated Github Action codecov/codecov-action@v1 to v3
  - Maintained 100% code coverage
  - Used unittest.mock.patch to prevent overwriting of config variables during testing
  - Added Python 3.11 to test matrix
  - Added ability to detect missing parameter definitions in docstrings (docstring.py)
  - Updated pytest flags to report skips and added additional summary
  - Replaced logging.warning with warnings.warn
  - Improved multi-line warnings
  - Ensured `bfs_indices` are sent to correct GPU device
  - Refactored window size check in mass/mass_absolute functions
  - Added animated GIF to README
  - Updated minimum black version
  - Removed `_parallel_rolling_func` as it conflicted with `numba` caching
  - Removed mamba timeout
  - Improved warnings
  - Added missing p-norm param to idx_to_mp and floss functions
  - Added test failure when coverage is below 100%
  - Added boolean array test for rolling_isfinite
  - Added isort, resolved circular dependencies, updated examples
  - Only build HTML for RTD
  - Added `mp` param to stumpi to allow pre-computed matrix profile as input
  - Added various unit tests
  - Removed codecov as dependency
  - Added ability to test the execution of tutorial notebooks
  - Added check for negative matrix profile indices
  - Added `test_precision.py` for all issues related to loss-of-precision
  - Fixed tls deprecation warning
  - Replaced np.int with np.int64
  - Specified fastmath flags to include nan/inf values in inputs/outputs
  - Replaced bool dtype with np.bool_
  - Replaced np.newaxis with np.expand_dims
  - Added check for docstring and parameter mismatch
  - Update URLs for minimum version references
  - Added explicit shell declaration in Github Actions workflow
  - Show OpenMP libraries in Github Actions workflow
  - Added `pip.sh` script for setting up dev environment using `pip`
  - Removed parallel=True in `core._compute_multi_PI`
  - Updated coverage testing to include all modules
  - Improved code consistency for `T_A` and `T_A` definitions
  - Refactored `test.sh` and include missing test files in unit tests
  - Added minimum dependency compatibility script (min.py)
  - Updated minimum dependency bumping instructions
  - Bumped minimum Python version to 3.8
  - Update PyPI downloads badge
* documentation
  - Improved/updated various docstrings
  - Added shapelet discovery tutorial
  - Clarified unanchored chain description
  - Fixed typos
  - Improved `core._get_QT docstring`
  - Fixed imbalanced tree representation in docstring
  - Improved scrump documentation
  - Made light mode default and remove theme switcher from header nav bar
  - Improved dataframe layout display in tutorials
  - Added multi-dimensional motif and match tutorial
  = Added T_subseq_isfinite to docstring
  - Added tutorial for "Discovering motifs under uniform scaling"
  - Updated docs for using a dask client
  - Fixed malformed link in floss docstring
  - Added missing parameter section in various docstrings
  - Added Minkowski docstring for Euclidean distance
  - Added missing parameters for GPU functions in docstrings
  - Improved documentation for `P` in motifs function


# 2022-03-31    [ 1.11.1 ]:
---------------------------
* bugfixes
  - Fixed #582 Allow 1D mean/stddev inputs for `stumpy.match`
* features
  - N/A
* tasks
  - Added mmotifs and aamp_mmotifs to __init__.py
* documentation
  - Added mmotifs docstring in RTD API


# 2022-03-21    [ 1.11.0 ]:
---------------------------
* bugfixes
  - Fixed #576 Incorrect stimp, stimped, gpu_stimp normalize rerouting
  - Fixed bad index in `naive.stimp` normalization method
  - Fixed unit tests
  - Fixed `cutoff=np.inf` edge case in `_motifs` function
* features
  - Added `mmotifs` and `aamp_mmotifs` functions for multi-dimensional motif discovery and unit tests
  - Added all AAMP/p-norm `stimp` implementations
  - Added `atol` parameter to `motifs` function
  - Added parallelized `prescraamp`
  - Added parallelized `prescrump`
  - Added `_get_ranges` function
  - Added `shoelace` formula for computing total diagonal ndists
  - Added `p_norm` support
  - Added `_preprocess` function
  - Added minimum description length function, `mdl`
  - Added Python 3.10 support
  - Added AAMP/p-norm support for pan matrix profiles
* tasks
  - Fixed typos
  - Added `gpu_aamp_stimp` driver error
  - Increased minimum dependencies
  - Updated coverage testings to include all unit tests and additional modules
  - Added `pip.sh` script for setting dev environment
  - Replaced `argsort` with `argmin`/`argmax`
  - Refactored redundant preprocessing steps
  - Replaced deprecated `scipy.ndimage.filters` module
  - Updated conda environment installation steps in `conda.sh` script
  - Updated `subspace` vs `subspaces` definition
  - Replaced `py.test` with `pytest`
  - Updated minimum `black` version to 22.1.0
  - Replaced elbow method with `mdl` in multi-dimensional motif tutorial 
  - Converted `float`/`int` type to `np.float64`/`np.int64`
  - Added `dtype` check and fill value to `apply_excl_zone` function
  - Added note on anti-correlated subsequences
* documentation
  - Updated README and various docstrings
  - Added MPdist tutorial draft
  - Added annotation vector tutorial
  - Added geometric time series chains tutorial draft
  - Added top-K motif section to motif discovery tutorial


# 2021-12-15    [ 1.10.2 ]:
---------------------------
* bugfixes
  - Fixed #501 Allow `max_distance = np.inf` in match function
* features
  - Added `atol=1e-8` parameter to match and motifs functions 
* tasks
  - Fixed typos
  - Removed conda download badge, updated PyPI download badge
  - Added removal __pycache__ in test clean up phase
* documentation
  - Updated pan matrix profile tutorial
  - Updated docstring for match and motifs functions


# 2021-12-15    [ 1.10.1 ]:
---------------------------
* bugfixes
  - Fixed #495 Reduce import time by removing Numba NJIT signatures 
* features
  - Added multi_distance_profile function
* tasks
  - Refactored _query_mstump_profile (see multi_distance_profile)
  - Added SVG STUMPY logo
* documentation
  - Refactored tutorial


# 2021-11-02    [ 1.10.0 ]:
---------------------------
* bugfixes
  - Raise TypeError for non np.float64 input arrays
  - Fixed NumbaPerformanceWarning backwards compatibility
* features
  - Converted all dtypes to np.float64/np.int64
  - Added explicit NJIT function signatures
  - Added Python 3.9 support
  - Added core._idx_to_mp convenience function
  - Added Nan/Inf support for motifs and matches functions
* tasks
  - Removed "in-tree" build flag for pip
  - Added pypi stats query for bigquery
  - Set explicit sphinx version for RTD
  - Fixed typos
  - Replaced direct installation via setup.py
  - Added numpydoc
  - Removed ipywidgets from requirements
  - Removed Azure Pipelines from CI
  - Used official OSI name in the license metadata, added license name in classifiers section
  - Added Github Citation BIB format
  - Fixed broken tutorial links
* documentation
  - Fixed multi-dimensional matrix profile description
  - Added clear warning in discord section of MSTUMP tutorial
  - Added API examples to all docstrings
  - Updated docstrings to use numpy.ndarray
  - Added interactive threshold example for STIMP
  - Updated matplotlib style sheet to use URL for all tutorials


# 2021-07-28    [ 1.9.2 ]:
--------------------------
* bugfixes
  - Fixed cutoff parameter not being used in motifs.py
* features
  - Unified motif discovery and pattern matching tools
* tasks
  - Added binder link to tutorial
* documentation
  - Updated pattern matching tutorial
  - Added Pearson correlation notebook
  - Fixed missing sphinx docstring for Python class


# 2021-07-20    [ 1.9.1 ]:
--------------------------
* tasks
  Bumped version

# 2021-07-20    [ 1.9.0 ]:
--------------------------
* bugfixes
  - Fixed scenarios where n_chunks == 0 in _get_array_ranges and fixed empty input array edge case
* features
  - Added `normalize` to `core.mass`
  - Added motif discovery (`stumpy.motifs` and `stumpy.match`)
  - Added snippets for identifying regimes (`stumpy.snippets`)
  - Added pan matrix profile (`stumpy.stimp`, `stumpy.stimped`, `stumpy.gpu_stimp`)
  - Added `excl_zone` parameter to config.py (`config.STUMPY_EXCL_ZONE_DENOM`)
* tasks
  - Aggregate or Refactor Dask Unit Tests
  - Added script for testing latest Numba release candidate
  - Converted bash scripts to [[ ... ]] construct
  - Updated Python class declaration
  - Updated to RTD to PyData Sphinx Theme
  - Updated conda installation environment
  - Refactored test files and added check to ensure that naive implementations always come ahead of tests
* documentation
  - Corrected binder badges to point to "main"
  - Updated tutorial to discover motif/discord indices
  - Added missing docstrings and fixed minor typos
  - Added missing logo file and favicon to _static directory
  - Updated source installation instructions
  - Added instructions for Apple M1 chip
  - Updated Contributor guide
  - Added include/discords tutorial example to subspace
  - Added bonus content on interpreting the columns of a matrix profile
  - Added syntax highlighting to tutorials
  - Replaced matplotlib params with style file
  - Added Annotation Vectors Tutorial
  - Added Binder links to top of tutorials
  - Added introduction to Snippets Tutorial


# 2021-02-04    [ 1.8.0 ]:
--------------------------
* bugfixes
  - Fixed chunk size for `scrump` and `scraamp` when time series are short
* features
  - Added `maamp` and `maamped` functions
  - Added `scraamp` function
  - Added a new `core.non_normalized` decorator that re-routes normalized functions to non-normalized functions
  - All z-normalized functions now accept a `normalize` parameter
* tasks
  - Renamed `main` branch
  - Removed Azure pipelines badge
  - Refactored `subspace`
  - Refactored non-normalized functions
  - Added non-normalized support to `floss`
* documentation
  - Updated README with `if __name__ == "__main__"` for Dask and Jupyter notebooks
  - Removed all `aamp` references as `normalize=False` should be used instead
  - Fixed function docstrings and typos in API docs


# 2021-01-20    [ 1.7.2 ]:
--------------------------
* bugfixes
  - None
* features
  - Added the NEP 29 policy
* tasks
  - Added CI for minimum version dependencies
* documentation
  - Updated README to conver NEP 29


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
