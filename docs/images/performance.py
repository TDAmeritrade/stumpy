#!/usr/bin/env python

import datetime
import math
import re


def seconds_to_time(x):
    x = float(x)
    x = round(x, 2)
    milliseconds, seconds = math.modf(x)
    milliseconds = math.ceil(milliseconds * 1000)
    seconds = int(seconds)

    dt = datetime.timedelta(seconds=int(seconds), milliseconds=int(milliseconds))

    out = str(dt)[:-4]
    if re.match(r"^\d:", out):
        out = "0" + out

    return out


def perf_to_time(text):
    for line in text.split("\n"):
        if re.match(r"^\s*$", line):
            continue
        line = line.strip()
        words = line.split()
        words[-1] = seconds_to_time(words[-1])
        out = " ".join(words)
        print(out)


if __name__ == "__main__":
    perf = """
           64 0.02545022964477539
           128 0.03863406181335449
           256 0.0771641731262207
           512 0.1278393268585205
           1024 0.24474358558654785
           2048 0.5341148376464844
           4096 1.03592848777771
           8192 1.9745631217956543
           16384 3.689000129699707
           32768 7.450819492340088
           65536 14.890256881713867
           131072 29.96510910987854
           262144 59.616419076919556
           524288 116.67402338981628
           1048576 306.47734475135803
           2097152 1227.9421191215515
           4194304 4872.334391593933
           100000000 123456.78910
           """

    perf_to_time(perf)
