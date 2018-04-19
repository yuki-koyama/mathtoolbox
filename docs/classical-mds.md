# classical-mds

Classical multi-dimensional scaling (MDS) for dimensionality reduction and low-dimensional embedding.

## Math

Given a distance (or dissimilarity) matrix of $$ n $$ elements

$$
\mathbf{D} \in \mathbb{R}^{n \times n},
$$

this technique calculates a set of $$ m $$-dimensional coordinates for them:

$$
\mathbf{X} = \begin{bmatrix} \mathbf{x}_1 & \cdots & \mathbf{x}_n \end{bmatrix}.
$$

If the elements are originally defined in a $$ m' $$-dimensional space ($$ m < m' $$) and Euclidian distance is used for calculating the distance matrix, then this is considered dimensionality reduction (or low-dimensional embedding).

## Useful Resources

- Josh Wills, Sameer Agarwal, David Kriegman, and Serge Belongie. 2009. Toward a perceptual space for gloss. ACM Trans. Graph. 28, 4, Article 103 (September 2009), 15 pages. DOI: <https://doi.org/10.1145/1559755.1559760>

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>