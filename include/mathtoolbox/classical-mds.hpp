#ifndef MATHTOOLBOX_CLASSICAL_MDS_HPP
#define MATHTOOLBOX_CLASSICAL_MDS_HPP

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace mathtoolbox
{
    /// \brief This function computes low-dimensional embedding by using classical multi-dimensional scaling (MDS)
    /// \param D Distance (dissimilarity) matrix and a target dimension for embedding
    /// \param dim Target dimension
    /// \return Coordinate matrix whose i-th column corresponds to the embedded coordinates of the i-th entry
    Eigen::MatrixXd ComputeClassicalMds(const Eigen::MatrixXd& D, unsigned dim);
} // namespace mathtoolbox

#endif // MATHTOOLBOX_CLASSICAL_MDS_HPP
