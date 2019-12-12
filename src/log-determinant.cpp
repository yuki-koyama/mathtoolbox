#include <mathtoolbox/log-determinant.hpp>

double mathtoolbox::CalcLogDetOfSymmetricPositiveDefiniteMatrix(const Eigen::MatrixXd& matrix)
{
    return CalcLogDetOfSymmetricPositiveDefiniteMatrix(Eigen::LLT<Eigen::MatrixXd>(matrix));
}

double mathtoolbox::CalcLogDetOfSymmetricPositiveDefiniteMatrix(const Eigen::LLT<Eigen::MatrixXd>& matrix_llt)
{
    return 2.0 * matrix_llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
}
