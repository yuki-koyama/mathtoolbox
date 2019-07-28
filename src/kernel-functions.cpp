#include <cmath>
#include <mathtoolbox/kernel-functions.hpp>

using Eigen::VectorXd;

double mathtoolbox::GetArdSquaredExpKernel(const VectorXd& x_a, const VectorXd& x_b, const VectorXd& theta)
{
    assert(x_a.size() == x_b.size());
    assert(x_a.size() == theta.size() - 1);

    const int       dim             = x_a.size();
    const double&   sigma_squared_f = theta[0];
    const VectorXd& length_scales   = theta.segment(1, dim);

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

VectorXd
mathtoolbox::GetArdSquaredExpKernelThetaDerivative(const VectorXd& x_a, const VectorXd& x_b, const VectorXd& theta)
{
    assert(x_a.size() == x_b.size());
    assert(x_a.size() == theta.size() - 1);

    const int       dim           = x_a.size();
    const VectorXd& length_scales = theta.segment(1, dim);

    VectorXd derivative(theta.size());

    derivative(0) = [&]() {
        double sum = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            const double r = x_a(i) - x_b(i);
            sum += (r * r) / (length_scales(i) * length_scales(i));
        }
        return std::exp(-0.5 * sum);
    }();

    const double k = GetArdSquaredExpKernel(x_a, x_b, theta);

    for (int i = 0; i < dim; ++i)
    {
        const double r    = x_a(i) - x_b(i);
        derivative(i + 1) = k * (r * r) / (length_scales(i) * length_scales(i) * length_scales(i));
    }

    return derivative;
}

double mathtoolbox::GetArdSquaredExpKernelThetaIDerivative(const VectorXd& x_a,
                                                           const VectorXd& x_b,
                                                           const VectorXd& theta,
                                                           const int       index)
{
    assert(x_a.size() == x_b.size());
    assert(x_a.size() == theta.size() - 1);

    const int       dim           = x_a.size();
    const VectorXd& length_scales = theta.segment(1, dim);

    if (index == 0)
    {
        double sum = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            const double r = x_a(i) - x_b(i);
            sum += (r * r) / (length_scales(i) * length_scales(i));
        }
        return std::exp(-0.5 * sum);
    }
    else
    {
        const int    i = index - 1;
        const double k = GetArdSquaredExpKernel(x_a, x_b, theta);
        const double r = x_a(i) - x_b(i);

        return k * (r * r) / (length_scales(i) * length_scales(i) * length_scales(i));
    }
}
