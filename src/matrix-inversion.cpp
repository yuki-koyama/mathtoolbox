#include <Eigen/LU>
#include <mathtoolbox/matrix-inversion.hpp>

using Eigen::MatrixXd;

MatrixXd mathtoolbox::GetInverseUsingUpperLeftBlockInverse(const MatrixXd& matrix,
                                                           const MatrixXd& upper_left_block_inverse)
{
    const int size       = matrix.rows();
    const int block_size = upper_left_block_inverse.rows();

    assert(block_size > 0);
    assert(size > block_size);

    assert(matrix.cols() == size);
    assert(upper_left_block_inverse.cols() == block_size);

    const Eigen::MatrixXd& A_inv = upper_left_block_inverse;
    const Eigen::MatrixXd& B     = matrix.block(0, block_size, block_size, size - block_size);
    const Eigen::MatrixXd& C     = matrix.block(block_size, 0, size - block_size, block_size);
    const Eigen::MatrixXd& D     = matrix.block(block_size, block_size, size - block_size, size - block_size);

    const Eigen::MatrixXd E     = D - C * A_inv * B;
    const Eigen::MatrixXd E_inv = E.inverse();

    assert((E * E_inv).isApprox(MatrixXd::Identity(E.rows(), E.cols())));

    const Eigen::MatrixXd I = MatrixXd::Identity(block_size, block_size);

    Eigen::MatrixXd result(size, size);

    result.block(0, 0, block_size, block_size)                                 = A_inv * (I + B * E_inv * C * A_inv);
    result.block(0, block_size, block_size, size - block_size)                 = -A_inv * B * E_inv;
    result.block(block_size, 0, size - block_size, block_size)                 = -E_inv * C * A_inv;
    result.block(block_size, block_size, size - block_size, size - block_size) = E_inv;

    assert((matrix * result).isApprox(MatrixXd::Identity(size, size)));

    return result;
}
