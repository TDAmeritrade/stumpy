#!/usr/bin/env python

import requests
from packaging.version import Version

classifiers = (
    requests.get("https://pypi.org/pypi/ray/json").json().get("info").get("classifiers")
)

versions = []
for c in classifiers:
    x = c.split()
    if "Python" in x:
        versions.append(x[-1])

versions.sort(key=Version)
print(versions[-1])
