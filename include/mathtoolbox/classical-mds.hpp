#ifndef CLASSICAL_MDS_HPP
#define CLASSICAL_MDS_HPP

#include <vector>
#include <utility>
#include <cassert>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace mathtoolbox
{
    // This function computes low-dimensional embedding by using classical multi-dimensional scaling (MDS)
    // - Input:  A distance (dissimilarity) matrix and a target dimension for embedding
    // - Output: A coordinate matrix whose i-th column corresponds to the embedded coordinates of the i-th entry
    extern inline Eigen::MatrixXd ComputeClassicalMds(const Eigen::MatrixXd& D, unsigned dim);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // This function extract the N-largest eigen values and eigen vectors
    inline void ExtractNLargestEigens(unsigned n, Eigen::VectorXd& S, Eigen::MatrixXd& V)
    {
        // Note: m is the original dimension
        const unsigned m = S.rows();

        // Copy the original matrix
        const Eigen::MatrixXd original_V = V;

        // Sort by eigenvalue
        constexpr double epsilon = 1e-16;
        std::vector<std::pair<double, unsigned>> index_value_pairs(m);
        for (unsigned i = 0; i < m; ++ i) index_value_pairs[i] = std::make_pair(std::max(S(i), epsilon), i);
        std::partial_sort(index_value_pairs.begin(), index_value_pairs.begin() + n, index_value_pairs.end(), std::greater<std::pair<double, unsigned>>());

        // Resize matrices
        S.resize(n);
        V.resize(m, n);

        // Set values
        for (unsigned i = 0; i < n; ++ i)
        {
            S(i)     = index_value_pairs[i].first;
            V.col(i) = original_V.col(index_value_pairs[i].second);
        }
    }

    inline Eigen::MatrixXd ComputeClassicalMds(const Eigen::MatrixXd& D, unsigned dim)
    {
        assert(D.rows() == D.cols());
        assert(D.rows() >= dim);
        const unsigned n = D.rows();
        const Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n) - (1.0 / static_cast<double>(n)) * Eigen::VectorXd::Ones(n) * Eigen::VectorXd::Ones(n).transpose();
        const Eigen::MatrixXd K = - 0.5 * H * D.cwiseAbs2() * H;
        const Eigen::EigenSolver<Eigen::MatrixXd> solver(K);
        Eigen::VectorXd S = solver.eigenvalues().real();
        Eigen::MatrixXd V = solver.eigenvectors().real();
        ExtractNLargestEigens(dim, S, V);
        const Eigen::MatrixXd X = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(S.cwiseSqrt()) * V.transpose();
        return X;
    }
}

#endif // CLASSICAL_MDS_HPP
