# matrix-inversion

Matrix inversion techniques.

## Header

```cpp
#include <mathtoolbox/matrix-inversion.hpp>
```

## Block Matrix Inversion

### About

Inverse of a block (partitioned) matrix

$$
\mathbf{X}
=
\begin{bmatrix}
  \mathbf{A} & \mathbf{B} \\
  \mathbf{C} & \mathbf{D}
\end{bmatrix},
$$

where $\mathbf{A}$ and $\mathbf{D}$ are square matrices, can be calculated as

$$
\begin{bmatrix}
  \mathbf{A} & \mathbf{B} \\
  \mathbf{C} & \mathbf{D}
\end{bmatrix}^{-1}
=
\begin{bmatrix}
  \mathbf{A}^{-1} ( \mathbf{I} + \mathbf{B} ( \mathbf{D} - \mathbf{C} \mathbf{A}^{-1} \mathbf{B} )^{-1} \mathbf{C} \mathbf{A}^{-1} ) &
  - \mathbf{A}^{-1} \mathbf{B} ( \mathbf{D} - \mathbf{C} \mathbf{A}^{-1} \mathbf{B} )^{-1} \\
  - ( \mathbf{D} - \mathbf{C} \mathbf{A}^{-1} \mathbf{B} )^{-1} \mathbf{C} \mathbf{A}^{-1} &
  ( \mathbf{D} - \mathbf{C} \mathbf{A}^{-1} \mathbf{B} )^{-1}
\end{bmatrix}
$$

This technique is useful particularly when $\mathbf{A}$ is relatively large (compared to $\mathbf{D}$) and $\mathbf{A}^{-1}$ is known.

### API

This module provides the following function:
```cpp
Eigen::MatrixXd GetInverseUsingUpperLeftBlockInverse(const Eigen::MatrixXd& matrix,
                                                     const Eigen::MatrixXd& upper_left_block_inverse);
```
where `upper_left_block_inverse` corresponds to $\mathbf{A}^{-1}$.

### Performance (Casual Comparison)

When $\mathbf{X}$ was a random matrix, and the size of $\mathbf{X}$ was 3,000 and that of $\mathbf{A}$ was 2,999, a naive approach (i.e., the LU decomposition from Eigen) took 5847 milliseconds to obtain $\mathbf{X}^{-1}$ while the block inversion approach took only 121 milliseconds.

## Useful Resources

- Block matrix - Wikipedia. <https://en.wikipedia.org/wiki/Block_matrix>.
