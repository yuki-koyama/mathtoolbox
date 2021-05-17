#include <Eigen/LU>
#include <cmath>
#include <mathtoolbox/rbf-interpolation.hpp>

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::PartialPivLU;
using Eigen::VectorXd;
using std::function;
using std::vector;

mathtoolbox::RbfInterpolator::RbfInterpolator(const function<double(const double)>& rbf_kernel,
                                              const bool                            use_polynomial_term)
    : m_rbf_kernel(rbf_kernel), m_use_polynomial_term(use_polynomial_term)
{
}

void mathtoolbox::RbfInterpolator::SetData(const MatrixXd& X, const VectorXd& y)
{
    assert(y.rows() == X.cols());

    this->m_X = X;
    this->m_y = y;
}

void mathtoolbox::RbfInterpolator::CalcWeights(const bool use_regularization, const double lambda)
{
    const int num_data = m_y.rows();

    // Construct the symmetric matrix of RBF values
    MatrixXd Phi{num_data, num_data};
    for (int i = 0; i < num_data; ++i)
    {
        for (int j = i; j < num_data; ++j)
        {
            const double value = m_rbf_kernel((m_X.col(i) - m_X.col(j)).norm());

            Phi(i, j) = value;
            Phi(j, i) = value;
        }
    }

    if (m_use_polynomial_term)
    {
        const int dim = m_X.rows();

        MatrixXd P{num_data, dim + 1};

        P.block(0, 0, num_data, 1)   = MatrixXd::Ones(num_data, 1);
        P.block(0, 1, num_data, dim) = m_X.transpose();

        MatrixXd A = MatrixXd::Zero(num_data + dim + 1, num_data + dim + 1);

        A.block(0, 0, num_data, num_data)       = Phi;
        A.block(0, num_data, num_data, dim + 1) = P;
        A.block(num_data, 0, dim + 1, num_data) = P.transpose();

        VectorXd b = VectorXd::Zero(num_data + dim + 1);

        b.segment(0, num_data) = m_y;

        VectorXd solution;
        if (use_regularization)
        {
            const auto I = MatrixXd::Identity(num_data + dim + 1, num_data + dim + 1);

            solution = PartialPivLU<MatrixXd>(A.transpose() * A + lambda * I).solve(A.transpose() * b);
        }
        else
        {
            solution = PartialPivLU<MatrixXd>(A).solve(b);
        }

        m_w = solution.segment(0, num_data);
        m_v = solution.segment(num_data, dim + 1);
    }
    else
    {
        const auto     I = MatrixXd::Identity(num_data, num_data);
        const MatrixXd A = use_regularization ? Phi.transpose() * Phi + lambda * I : Phi;
        const VectorXd b = use_regularization ? Phi.transpose() * m_y : m_y;

        m_w = PartialPivLU<MatrixXd>(A).solve(b);
    }
}

double mathtoolbox::RbfInterpolator::CalcValue(const VectorXd& x) const
{
    assert(x.rows() == m_X.rows());

    const int num_data = m_w.rows();
    const int dim      = x.rows();

    // Calculate the distance for each data point via broadcasting
    const Eigen::VectorXd norms = (m_X.colwise() - x).colwise().norm();

    // Calculate the RBF value associated with each data point
    // TODO: This part can be further optimized for performance by vectorization
    VectorXd rbf_values{num_data};
    for (int i = 0; i < num_data; ++i)
    {
        rbf_values(i) = m_rbf_kernel(norms(i));
    }

    // Calculate the weighted sum using dot product
    const double rbf_term = m_w.dot(rbf_values);

    if (m_use_polynomial_term)
    {
        const double polynomial_term = m_v(0) + x.transpose() * m_v.segment(1, dim);

        return rbf_term + polynomial_term;
    }
    else
    {
        return rbf_term;
    }
}
