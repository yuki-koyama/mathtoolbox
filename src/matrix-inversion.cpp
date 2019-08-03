#include <Eigen/LU>
#include <mathtoolbox/matrix-inversion.hpp>

using Eigen::MatrixXd;

MatrixXd mathtoolbox::GetInverseUsingUpperLeftBlockInverse(const MatrixXd& matrix,
                                                           const MatrixXd& upper_left_block_inverse)
{
    const int size       = matrix.rows();
    const int block_size = upper_left_block_inverse.rows();
    const int rest_size  = size - block_size;

    assert(block_size > 0);
    assert(size > block_size);

    assert(matrix.cols() == size);
    assert(upper_left_block_inverse.cols() == block_size);

    const Eigen::MatrixXd& A_inv = upper_left_block_inverse;
    const Eigen::MatrixXd& B     = matrix.block(0, block_size, block_size, rest_size);
    const Eigen::MatrixXd& C     = matrix.block(block_size, 0, rest_size, block_size);
    const Eigen::MatrixXd& D     = matrix.block(block_size, block_size, rest_size, rest_size);

    const Eigen::MatrixXd E     = D - C * A_inv * B;
    const Eigen::MatrixXd E_inv = E.inverse();

    assert((E * E_inv - MatrixXd::Identity(E.rows(), E.cols())).cwiseAbs().maxCoeff() < 1e-10);

    Eigen::MatrixXd result(size, size);

    result.block(0, 0, block_size, block_size)                 = A_inv + (A_inv * B) * E_inv * (C * A_inv);
    result.block(0, block_size, block_size, rest_size)         = -A_inv * B * E_inv;
    result.block(block_size, 0, rest_size, block_size)         = -E_inv * C * A_inv;
    result.block(block_size, block_size, rest_size, rest_size) = E_inv;

    assert((matrix * result - MatrixXd::Identity(size, size)).cwiseAbs().maxCoeff() < 1e-10);

    return result;
}
