#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <mathtoolbox/metric-mds.hpp>

using Eigen::VectorXd;
using Eigen::MatrixXd;

int main()
{
    // Generate test data (5-dimensional points)
    std::vector<VectorXd> points(10);
    points[0] = VectorXd(5); points[0] << 0.0, 2.0, 0.0, 3.0, 4.0;
    points[1] = VectorXd(5); points[1] << 1.0, 0.0, 2.0, 4.0, 3.0;
    points[2] = VectorXd(5); points[2] << 0.0, 1.0, 4.0, 2.0, 0.0;
    points[3] = VectorXd(5); points[3] << 0.0, 4.0, 1.0, 0.0, 2.0;
    points[4] = VectorXd(5); points[4] << 4.0, 3.0, 0.0, 1.0, 0.0;
    points[5] = VectorXd(5); points[5] << 3.0, 4.0, 2.0, 0.0, 1.0;
    points[6] = VectorXd(5); points[6] << 0.0, 2.0, 4.0, 1.0, 0.0;
    points[7] = VectorXd(5); points[7] << 2.0, 0.0, 1.0, 4.0, 0.0;
    points[8] = VectorXd(5); points[8] << 0.0, 1.0, 0.0, 3.0, 4.0;
    points[9] = VectorXd(5); points[9] << 1.0, 0.0, 2.0, 0.0, 3.0;
    
    // Generate a distance matrix
    MatrixXd D(10, 10);
    for (unsigned i = 0; i < 10; ++ i) for (unsigned j = i; j < 10; ++ j)
    {
        const double d = (points[i] - points[j]).squaredNorm();
        D(i, j) = d;
        D(j, i) = d;
    }
    
    // Compute metric MDS (embedding into a 2-dimensional space)
    const MatrixXd X = mathtoolbox::ComputeMetricMds(D, 2);
    
    // Show the result
    std::cout << X << std::endl;
    
    return 0;
}
