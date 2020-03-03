#include <cassert>
#include <mathtoolbox/classical-mds.hpp>
#include <utility>
#include <vector>

// Extract the N-largest eigen values and eigen vectors
inline void ExtractNLargestEigens(unsigned n, Eigen::VectorXd& S, Eigen::MatrixXd& V)
{
    // Note: m is the original dimension
    const unsigned m = S.rows();

    // Copy the original matrix
    const Eigen::MatrixXd original_V = V;

    // Sort by eigenvalue
    constexpr double                         epsilon = 1e-16;
    std::vector<std::pair<double, unsigned>> index_value_pairs(m);
    for (unsigned i = 0; i < m; ++i)
    {
        index_value_pairs[i] = std::make_pair(std::max(S(i), epsilon), i);
    }
    std::partial_sort(index_value_pairs.begin(),
                      index_value_pairs.begin() + n,
                      index_value_pairs.end(),
                      std::greater<std::pair<double, unsigned>>());

    // Resize matrices
    S.resize(n);
    V.resize(m, n);

    // Set values
    for (unsigned i = 0; i < n; ++i)
    {
        S(i)     = index_value_pairs[i].first;
        V.col(i) = original_V.col(index_value_pairs[i].second);
    }
}

Eigen::MatrixXd mathtoolbox::ComputeClassicalMds(const Eigen::MatrixXd& D, const unsigned target_dim)
{
    assert(D.rows() == D.cols());
    assert(D.rows() >= target_dim);

    const auto n    = D.rows();
    const auto ones = Eigen::VectorXd::Ones(n);
    const auto I    = Eigen::MatrixXd::Identity(n, n);

    const auto H = I - (1.0 / static_cast<double>(n)) * ones * ones.transpose();
    const auto K = -0.5 * H * D.cwiseAbs2() * H;

    const Eigen::EigenSolver<Eigen::MatrixXd> solver(K);

    Eigen::VectorXd S = solver.eigenvalues().real();
    Eigen::MatrixXd V = solver.eigenvectors().real();

    ExtractNLargestEigens(target_dim, S, V);

    const Eigen::MatrixXd X = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(S.cwiseSqrt()) * V.transpose();

    return X;
}
