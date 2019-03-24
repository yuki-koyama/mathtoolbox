# bfgs

BFGS method (BFGS) is a numerical optimization algorithm that is one of the most popular choices in quasi-Newton methods.

## Header

```
#include <mathtoolbox/bfgs.hpp>
```

## Math and Algorithm

To find an appropriate step size (called _alpha_), this implementation uses _Backtracking Line Search_ procedure considering the _sufficient decrease condition_ (also known as _Armijo condition_).

## Useful Resources

- Jorge Nocedal and Stephen J. Wright. 2006. Numerical optimization (2nd ed.). Springer. DOI: <https://doi.org/10.1007/978-0-387-40065-5>
