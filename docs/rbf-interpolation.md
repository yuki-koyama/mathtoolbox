# rbf-interpolation

Radial basis function (RBF) network for scattered data interpolation and approximation.

## Math

Given input data:

$$
\{ (\mathbf{x}_i, y_i) \}_{i = 1, \ldots, N},
$$

this technique calculates an interpolated value $$ y $$ for a specified point $$ \mathbf{x} $$ by

$$
y = f(\mathbf{x}) = \sum_{i = 1}^{N} w_{i} \phi( \| \mathbf{x} - \mathbf{x}_{i} \|),
$$

where $$ \phi(\cdot) $$ is an RBF and $$ w_1, \cdots, w_N $$ are weights.

## Useful Resources

- Ken Anjyo, J. P. Lewis, and Frédéric Pighin. 2014. Scattered data interpolation for computer graphics. In ACM SIGGRAPH 2014 Courses (SIGGRAPH '14). ACM, New York, NY, USA, Article 27, 69 pages. DOI: <https://doi.org/10.1145/2614028.2615425>
