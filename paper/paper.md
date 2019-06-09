---
title: 'STUMPY: A Powerful and Scalable Python Library for Time Series Data Mining'
tags:
  - time series analysis
  - data science
  - anomaly detection
  - pattern matching
  - machine learning
authors:
 - name: Sean M. Law
   orcid: 0000-0003-2594-1969
   affiliation: 1
affiliations:
 - name: TD Ameritrade
   index: 1
date: 3 June 2019
bibliography: paper.bib
---

# Summary

![STUMPY Logo](stumpy_logo_small.png)

Direct visualization, summary statistics (i.e., minimum, maximum, mean, standard deviation), ARIMA models, anomaly detection, forecasting, clustering, and deep learning are all popular techniques for analyzing and understanding time series data. However, the simplest and most intuitive  approach of comparing all of the pairwise distances between each subsequence within a time series (i.e., a self-similarity join) has not seen much progress due to its inherent computational complexities. For a time series with length *n* and a subsequence comparison length *m*, the brute force self-similarity join for this sqeuence would have a computational complexity of  *O(n^2^m)*. To put this into perspective, assuming that each distance calculation took 0.0000001 seconds, a time series of length *n = 100,000,000* would require roughly 1,585.49 years to compute all of the pairwise distances in a brute force manner. The ability to accurately and efficiently compute the exact similarity join would enable, amongst other things, time series motif and time series discord discovery. While approximate methods exist, they are often inexact, lead to false positives or false dismissals, and do not generalize well to other time series data. Novel research for computing the exact similarity join has significantly improved the scalability for exploring larger datasets without compromise.

Leveraging this work, we present STUMPY, a powerful and scalable library that efficiently computes something called the matrix profile (a vector that represents the distances between all subsequences within a time series and their nearest neighbors) [@yeh2016], [@zhu2016], which can be used for a variety of time series data mining tasks such as:

* pattern/motif (approximately repeated subsequences within a longer time series) discovery
* anomaly/novelty (discord) discovery
* shapelet discovery
* semantic segmentation
* density estimation
* time series chains (temporally ordered set of subsequence patterns)
* and more ...

![Matrix Profile](matrix_profile.jpeg)

The library also includes support for parallel and distributed computation, multi-dimensional motif discovery [@yeh2017], and time series chains [@zhu2017]. Whether you are an academic, data scientist, software developer, or time series enthusiast, STUMPY is straightforward to install and allows you to compute the matrix profile in the most efficient way. The goal of STUMPY is to allow you to get to your time series insights faster.

# References
  
