#include <mathtoolbox/kernel-functions.hpp>

using Eigen::VectorXd;

double
mathtoolbox::GetArdSquaredExponentialKernel(const VectorXd& x_a, const VectorXd& x_b, const VectorXd& hyperparameters)
{
    assert(x_a.rows() == x_b.rows());
    assert(x_a.rows() == hyperparameters.rows() + 1);

    const int dim = x_a.rows();

    const double&          sigma_squared_f = hyperparameters[0];
    const Eigen::VectorXd& length_scales   = hyperparameters.segment(1, dim);

    const double sum = [&]() {
        double sum = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            const double r = x_a(i) - x_b(i);
            sum += (r * r) / (length_scales(i) * length_scales(i));
        }
        return sum;
    }();

    return sigma_squared_f * std::exp(-0.5 * sum);
}

