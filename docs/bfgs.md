# bfgs

The BFGS method (BFGS) is a numerical optimization algorithm that is one of the most popular choices among quasi-Newton methods.

## Header

```
#include <mathtoolbox/bfgs.hpp>
```

## Math and Algorithm

We follow [Nocedal and Wright (2006)](https://doi.org/10.1007/978-0-387-40065-5) (Chapter 8).

### Inverse Hessian Initialization

This implementation adopts the strategy described in Equation 8.20.

### Line Search

This implementation uses [strong-wolfe-conditions-line-search](./strong-wolfe-conditions-line-search) to find an appropriate step size.

## Useful Resources

- Jorge Nocedal and Stephen J. Wright. 2006. Numerical optimization (2nd ed.). Springer. DOI: <https://doi.org/10.1007/978-0-387-40065-5>
