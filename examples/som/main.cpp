#include <iostream>
#include <mathtoolbox/som.hpp>

int main()
{
    constexpr int    num_data_dims        = 3;
    constexpr int    num_points           = 30;
    constexpr int    num_latent_dims      = 2;
    constexpr int    resolution           = 10;
    constexpr double init_var             = 0.20;
    constexpr double min_var              = 0.01;
    constexpr double var_decreasing_speed = 20.0;
    constexpr bool   normalize_data       = true;

    Eigen::MatrixXd data = Eigen::MatrixXd::Random(num_data_dims, num_points);

    mathtoolbox::Som som(data, num_latent_dims, resolution, init_var, min_var, var_decreasing_speed, normalize_data);

    for (int i = 0; i < 50; ++i)
    {
        const auto temp = som.GetDataSpaceNodePositions();

        som.Step();

        std::cout << "#iter: " << i + 1 << ", delta: " << (temp - som.GetDataSpaceNodePositions()).norm() << std::endl;
    }

    return 0;
}
