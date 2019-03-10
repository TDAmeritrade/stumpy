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

## Performance

We tested the performance using the Numba JIT compiled version of the code on data with various lengths.

Hardware: 16 CPUs in Total - 16x Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz processors on a single server.

i: 6  n: 64  time: 00:00:00.00
i: 7  n: 128  time: 00:00:00.00
i: 8  n: 256  time: 00:00:00.01
i: 9  n: 512  time: 00:00:00.02
i: 10  n: 1024  time: 00:00:00.04
i: 11  n: 2048  time: 00:00:00.09
i: 12  n: 4096  time: 00:00:00.19
i: 13  n: 8192  time: 00:00:00.41
i: 14  n: 16384  time: 00:00:00.99
i: 15  n: 32768  time: 00:00:02.39
i: 16  n: 65536  time: 00:00:06.42
i: 17  n: 131072  time: 00:00:19.52
i: 18  n: 262144  time: 00:01:08.44
i: 19  n: 524288  time: 00:03:56.82
i: 20  n: 1048576  time: 00:19:54.75
i: 21  n: 2097152  time: 03:05:07.64
i: 22  n: 4194304  time: 10:37:51.21
i: 23  n: 8388608  time: 38:42:51.42

Hardware: 128 CPUs in Total - 8x Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz processors x 16 servers.

i: 6  n: 64  time: 00:00:05.77
i: 7  n: 128  time: 00:00:05.93
i: 8  n: 256  time: 00:00:05.95
i: 9  n: 512  time: 00:00:05.97
i: 10  n: 1024  time: 00:00:05.69
i: 11  n: 2048  time: 00:00:05.60
i: 12  n: 4096  time: 00:00:06.26
i: 13  n: 8192  time: 00:00:06.29
i: 14  n: 16384  time: 00:00:06.24
i: 15  n: 32768  time: 00:00:06.48
i: 16  n: 65536  time: 00:00:07.33
i: 17  n: 131072  time: 00:00:09.75
i: 18  n: 262144  time: 00:00:33.38
i: 19  n: 524288  time: 00:01:35.27
i: 20  n: 1048576  time: 00:04:37.15
i: 21  n: 2097152  time: 00:13:36.51
i: 22  n: 4194304  time: 00:55:44.43
i: 23  n: 8388608  time: 03:33:30.53
i: 24  n: 16777216  time: 13:03:43.86

Hardware: 256 CPUs in Total - 8x Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz processors x 32 servers.



## License
[BSD 3](License)

## Documentation

## Getting Help

## References
