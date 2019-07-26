# bfgs

The BFGS method (BFGS) is a numerical optimization algorithm that is one of the most popular choices among quasi-Newton methods.

## Header

```cpp
#include <mathtoolbox/bfgs.hpp>
```

## Internal Dependencies

- [strong-wolfe-conditions-line-search](../strong-wolfe-conditions-line-search/)

## Math and Algorithm

We follow [Nocedal and Wright (2006)](https://doi.org/10.1007/978-0-387-40065-5) (Chapter 6).

### Inverse Hessian Initialization

This implementation adopts the strategy described in Equation 6.20:

$$
\mathbf{H}_0 \leftarrow \frac{\mathbf{y}_k^T \mathbf{s}_k}{\mathbf{y}_k^T \mathbf{y}_k} \mathbf{I}.
$$

See the book for details.

### Line Search

This implementation uses [strong-wolfe-conditions-line-search](../strong-wolfe-conditions-line-search) to find an appropriate step size.

## Useful Resources

- Jorge Nocedal and Stephen J. Wright. 2006. Numerical optimization (2nd ed.). Springer. DOI: <https://doi.org/10.1007/978-0-387-40065-5>
