#ifndef MATHTOOLBOX_LOG_DETERMINANT_HPP
#define MATHTOOLBOX_LOG_DETERMINANT_HPP

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace mathtoolbox
{
    double CalcLogDetOfSymmetricPositiveDefiniteMatrix(const Eigen::MatrixXd& matrix);
    double CalcLogDetOfSymmetricPositiveDefiniteMatrix(const Eigen::LLT<Eigen::MatrixXd>& matrix_llt);
} // namespace mathtoolbox

#endif // MATHTOOLBOX_LOG_DETERMINANT_HPP
