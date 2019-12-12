#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <cmath>
#include <iostream>
#include <mathtoolbox/log-determinant.hpp>

double CalcCovariance(const Eigen::VectorXd& x_1, const Eigen::VectorXd& x_2)
{
    return std::exp(-(x_1 - x_2).squaredNorm());
}

int main(int argc, char** argv)
{
    constexpr int num_points = 150;
    constexpr int num_dims   = 3;

    const Eigen::MatrixXd points = Eigen::MatrixXd::Random(num_dims, num_points);

    Eigen::MatrixXd covariance_matrix(num_points, num_points);
    for (int i = 0; i < num_points; ++i)
    {
        for (int j = i; j < num_points; ++j)
        {
            const double covariance = CalcCovariance(points.col(i), points.col(j));

            covariance_matrix(i, j) = covariance;
            covariance_matrix(j, i) = covariance;
        }
    }

    const double log_det       = mathtoolbox::CalcLogDetOfSymmetricPositiveDefiniteMatrix(covariance_matrix);
    const double log_det_naive = std::log(covariance_matrix.determinant());

    assert(!std::isnan(log_det));

    std::cout << "log(det(K)) = " << log_det << std::endl;
    std::cout << "log(det(K)) = " << log_det_naive
              << " (calculated by a naive approach, which may suffer from numerical instability)" << std::endl;

    return 0;
}
