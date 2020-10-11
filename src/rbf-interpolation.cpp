#include <Eigen/LU>
#include <cmath>
#include <mathtoolbox/rbf-interpolation.hpp>

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::PartialPivLU;
using Eigen::VectorXd;
using std::vector;

mathtoolbox::RbfInterpolator::RbfInterpolator(const std::function<double(const double)>& rbf_kernel)
    : m_rbf_kernel(rbf_kernel)
{
}

void mathtoolbox::RbfInterpolator::SetData(const Eigen::MatrixXd& X, const Eigen::VectorXd& y)
{
    assert(y.rows() == X.cols());
    this->m_X = X;
    this->m_y = y;
}

void mathtoolbox::RbfInterpolator::CalcWeights(const bool use_regularization, const double lambda)
{
    const int dim = m_y.rows();

    MatrixXd Phi = MatrixXd::Zero(dim, dim);
    for (int i = 0; i < dim; ++i)
    {
        for (int j = i; j < dim; ++j)
        {
            const double value = CalcRbfValue(m_X.col(i), m_X.col(j));

            Phi(i, j) = value;
            Phi(j, i) = value;
        }
    }

    const MatrixXd A = use_regularization ? Phi.transpose() * Phi + lambda * MatrixXd::Identity(dim, dim) : Phi;
    const VectorXd b = use_regularization ? Phi.transpose() * m_y : m_y;

    m_w = PartialPivLU<MatrixXd>(A).solve(b);
}

double mathtoolbox::RbfInterpolator::CalcValue(const VectorXd& x) const
{
    assert(x.rows() == m_X.rows());

    const int dim = m_w.rows();

    // Calculate the distance for each data point via broadcasting
    const auto norms = (m_X.colwise() - x).colwise().norm();

    // Calculate the RBF value associated with each data point
    // TODO: This part can be further optimized for performance by vectorization
    Eigen::VectorXd rbf_values{dim};
    for (int i = 0; i < dim; ++i)
    {
        rbf_values(i) = m_rbf_kernel(norms(i));
    }

    // Calculate the weighted sum using dot product
    const double value = m_w.dot(rbf_values);

    return value;
}

double mathtoolbox::RbfInterpolator::CalcRbfValue(const VectorXd& xi, const VectorXd& xj) const
{
    assert(xi.rows() == xj.rows());

    return m_rbf_kernel((xj - xi).norm());
}
