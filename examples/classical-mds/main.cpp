#include <Eigen/Core>
#include <iostream>
#include <mathtoolbox/classical-mds.hpp>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main()
{
    // Generate test data (5-dimensional points)
    std::vector<VectorXd> points(10, VectorXd(5));
    points[0] << 0.0, 2.0, 0.0, 3.0, 4.0;
    points[1] << 1.0, 0.0, 2.0, 4.0, 3.0;
    points[2] << 0.0, 1.0, 4.0, 2.0, 0.0;
    points[3] << 0.0, 4.0, 1.0, 0.0, 2.0;
    points[4] << 4.0, 3.0, 0.0, 1.0, 0.0;
    points[5] << 3.0, 4.0, 2.0, 0.0, 1.0;
    points[6] << 0.0, 2.0, 4.0, 1.0, 0.0;
    points[7] << 2.0, 0.0, 1.0, 4.0, 0.0;
    points[8] << 0.0, 1.0, 0.0, 3.0, 4.0;
    points[9] << 1.0, 0.0, 2.0, 0.0, 3.0;

    // Generate a distance matrix
    MatrixXd D(10, 10);
    for (unsigned i = 0; i < 10; ++i)
    {
        for (unsigned j = i; j < 10; ++j)
        {
            const double d = (points[i] - points[j]).norm();

            D(i, j) = d;
            D(j, i) = d;
        }
    }

    // Compute metric MDS (embedding into a 2-dimensional space)
    const MatrixXd X = mathtoolbox::ComputeClassicalMds(D, 2);

    // Show the result
    std::cout << X.format(Eigen::IOFormat(3)) << std::endl;

    return 0;
}
