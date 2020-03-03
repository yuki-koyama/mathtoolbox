#ifndef MATHTOOLBOX_CLASSICAL_MDS_HPP
#define MATHTOOLBOX_CLASSICAL_MDS_HPP

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace mathtoolbox
{
    /// \brief Compute low-dimensional embedding by using classical multi-dimensional scaling (MDS)
    ///
    /// \param D Distance (dissimilarity) matrix
    ///
    /// \param target_dim Target dimensionality
    ///
    /// \return Coordinate matrix whose i-th column corresponds to the embedded coordinates of the i-th entry
    Eigen::MatrixXd ComputeClassicalMds(const Eigen::MatrixXd& D, const unsigned taregt_dim);
} // namespace mathtoolbox

#endif // MATHTOOLBOX_CLASSICAL_MDS_HPP
