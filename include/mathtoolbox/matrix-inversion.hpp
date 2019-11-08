#ifndef MATHTOOLBOX_MATRIX_INVERSION_HPP
#define MATHTOOLBOX_MATRIX_INVERSION_HPP

#include <Eigen/Core>

namespace mathtoolbox
{
    Eigen::MatrixXd GetInverseUsingUpperLeftBlockInverse(const Eigen::MatrixXd& matrix,
                                                         const Eigen::MatrixXd& upper_left_block_inverse);
} // namespace mathtoolbox

#endif // MATHTOOLBOX_MATRIX_INVERSION_HPP
