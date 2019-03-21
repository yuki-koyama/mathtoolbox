#include <iostream>
#include <Eigen/Core>
#include <mathtoolbox/bfgs.hpp>
#include <optimization-test-functions.hpp>

int main()
{
    mathtoolbox::Setting setting;
    setting.f = [](const Eigen::VectorXd& x)
    {
        return otf::GetValue(x, otf::FunctionType::Sphere);
    };
    setting.f_grad = [](const Eigen::VectorXd& x)
    {
        return otf::GetGrad(x, otf::FunctionType::Sphere);
    };

    const mathtoolbox::Result result = mathtoolbox::RunOptimization(setting);

    std::cout << "solution: " << result.x_star << std::endl;

    return 0;
}
