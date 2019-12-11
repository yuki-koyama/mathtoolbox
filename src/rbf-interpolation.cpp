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
    const int dim = m_w.rows();

    double result = 0.0;
    for (int i = 0; i < dim; ++i)
    {
        result += m_w(i) * CalcRbfValue(x, m_X.col(i));
    }

    return result;
}

double mathtoolbox::RbfInterpolator::CalcRbfValue(const VectorXd& xi, const VectorXd& xj) const
{
    assert(xi.rows() == xj.rows());

    return m_rbf_kernel((xj - xi).norm());
}
