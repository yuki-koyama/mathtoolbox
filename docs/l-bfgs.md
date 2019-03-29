# l-bfgs

The Limited-memory BFGS method (L-BFGS) is a numerical optimization algorithm that is one of the most popular choices among quasi-Newton methods.

## Header

```
#include <mathtoolbox/l-bfgs.hpp>
```

## Math and Algorithm

We follow [Nocedal and Wright (2006)](https://doi.org/10.1007/978-0-387-40065-5) (Chapter 9).

### Inverse Hessian Initialization

This implementation adopts the strategy described in Equation 9.6.

### Line Search

To find an appropriate step size (called _alpha_), this implementation uses Algorithm 3.2, which considers both the _sufficient decrease condition_ (also known as the _Armijo condition_) and the _curvature condition_ (collectively known as the _strong Wolfe conditions_).

## Useful Resources

- Jorge Nocedal and Stephen J. Wright. 2006. Numerical optimization (2nd ed.). Springer. DOI: <https://doi.org/10.1007/978-0-387-40065-5>
