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

|    i     |  n = 2^i  | GPU-STOMP    | STUMP.16    | STUMPED.128 | STUMPED.256 |
| -------- | ----------| ------------ | ----------- | ----------- | ----------- |
| 6        | 64        | 00:00:10.00  | 00:00:00.00 | 00:00:05.77 | 00:00:06.08 |
| 7        | 128       | 00:00:10.00  | 00:00:00.00 | 00:00:05.93 | 00:00:07.29 |
| 8        | 256       | 00:00:10.00  | 00:00:00.01 | 00:00:05.95 | 00:00:07.59 |
| 9        | 512       | 00:00:10.00  | 00:00:00.02 | 00:00:05.97 | 00:00:07.47 |
| 10       | 1024      | 00:00:10.00  | 00:00:00.04 | 00:00:05.69 | 00:00:07.64 |
| 11       | 2048      | NaN          | 00:00:00.09 | 00:00:05.60 | 00:00:07.83 |
| 12       | 4096      | NaN          | 00:00:00.19 | 00:00:06.26 | 00:00:07.90 |
| 13       | 8192      | NaN          | 00:00:00.41 | 00:00:06.29 | 00:00:07.73 |
| 14       | 16384     | NaN          | 00:00:00.99 | 00:00:06.24 | 00:00:08.18 |
| 15       | 32768     | NaN          | 00:00:02.39 | 00:00:06.48 | 00:00:08.29 |
| 16       | 65536     | NaN          | 00:00:06.42 | 00:00:07.33 | 00:00:09.01 |
| 17       | 131072    | 00:00:10.00  | 00:00:19.52 | 00:00:09.75 | 00:00:10.53 |
| 18       | 262144    | 00:00:18.00  | 00:01:08.44 | 00:00:33.38 | 00:00:24.07 |
| 19       | 524288    | 00:00:46.00  | 00:03:56.82 | 00:01:35.27 | 00:03:43.66 |
| 20       | 1048576   | 00:02:30.00  | 00:19:54.75 | 00:04:37.15 | 00:03:01.16 |
| 21       | 2097152   | 00:09:15.00  | 03:05:07.64 | 00:13:36.51 | 00:08:47.47 |
| 22       | 4194304   | NaN          | 10:37:51.21 | 00:55:44.43 | 00:32:06.70 |
| 23       | 8388608   | NaN          | 38:42:51.42 | 03:33:30.53 | 02:00:49.37 |
| 24       | 16777216  | NaN          | NaN         | 13:03:43.86 | 07:13:47.12 |
| NaN      | 17729800  | 09:16:12.00  | NaN         | NaN         | NaN         |
| 25       | 33554432  | NaN          | NaN         | NaN         | 28:58:09.19 |
| 26       | 67108864  | NaN          | NaN         | NaN         | 111:17:08.22 |
| NaN      | 100000000 | 291:07:12.00 | NaN         | NaN         | NaN         |
| 27       | 134217728 | NaN          | NaN         | NaN         | NaN         |

GPU-STOMP: NVIDIA Tesla K80 (contains 2 GPUs) 
    
STUMP.16: 16 CPUs in Total - 16x Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz processors parallelized with Numba on a single server without Dask.

STUMPED.128: 128 CPUs in Total - 8x Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz processors x 16 servers, parallelized with Numba, and distributed with Dask Distributed.

STUMPED.256: 256 CPUs in Total - 8x Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz processors x 32 servers, parallelized with Numba, and distributed with Dask Distributed.

## License
[BSD 3](License)

## Documentation

## Getting Help

## References

Yeh, Chin-Chia Michael, et al. (2016) Matrix Profile I: All Pairs Similarity Joins for Time Series: A Unifiying View that Includes Motifs, Discords, and Shapelets. ICDM:1317-1322. [Link](https://ieeexplore.ieee.org/abstract/document/7837992)

Zhu, Yan, et al. (2016) Matrix Profile II: Exploiting a Novel Algorithm and GPUs to Break the One Hundred Million Barrier for TIme Series Motifs and Joins. ICDM:739-748. [Link](https://ieeexplore.ieee.org/abstract/document/7837898)

Zhu, Yan, et al. (2017) Matrix Profile VII: Time Series Chains: A New Primitive for Time Series Data Mining. ICDM:695-704. [Link](https://ieeexplore.ieee.org/abstract/document/8215542)
