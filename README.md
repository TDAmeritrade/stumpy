# Matrix Profile: Automatically Finding Patterns in Time Series or Sequential Data

## What is it?

## Main Features

## Where to get it

```PyPI
# From source
pip install matrix_profile
```

## Dependencies
- [NumPy]()
- [Numba]()
- [SciPy]()

## Installation from sources

To install matrix_profile from source, you'll need to install the dependencies above. For maximum performance, it is recommended that you install all dependencies using `conda`:

```sh
conda install -y numpy
conda install -y scipy
conda install -y numba
```

Alternatively, but with lower performance, you can also install these dependencies using the requirements.txt file (found in the `matrix_profile` directory (same directory where you found this file after cloning the git repo)):

```sh
pip install -r requirements.txt
```
Once the dependencies are installed (stay inside of the `matrix_profile` directory), execute:

```sh
pip install .
```

## Running Tests

Tests are written in the tests directory and processed using [PyTest](). Tests can be executed with:

```sh
./test.sh
```

## License
[BSD 3](License)

## Documentation

## Getting Help

## References
