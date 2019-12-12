# log-determinant

Techniques for calculating log-determinant of a matrix.

## Header

```cpp
#include <mathtoolbox/log-determinant.hpp>
```

## Log-Determinant of a Symmetric Positive-Definite Matrix

Let $ \mathbf{K} \in \mathbb{R}^{n \times n} $ be a symmetric positive-definite matrix such a *covariance matrix*. The goal is to calculate the log of its determinant:
$$
\log(\det(\mathbf{K})).
$$
This calculation often appears when handling a log-likelihood of some Gaussian-related event.

A naive way is to calculate the determinant explicitly and then calculate its log. However, this way is known for its numerical instability (i.e., likely to go to negative infinity).

This module offers a function to calculate log-determinant much more stably.

## Useful Resources

- Compute the log-determinant of a matrix - The DO Loop. <https://blogs.sas.com/content/iml/2012/10/31/compute-the-log-determinant-of-a-matrix.html>.
