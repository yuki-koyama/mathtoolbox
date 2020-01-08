#include <iostream>
#include <mathtoolbox/som.hpp>

int main()
{
    constexpr int  num_data_dims   = 3;
    constexpr int  num_points      = 20;
    constexpr int  num_latent_dims = 2;
    constexpr int  resolution      = 20;
    constexpr bool normalize_data  = false;

    Eigen::MatrixXd data = Eigen::MatrixXd::Random(num_data_dims, num_points);

    mathtoolbox::Som som(data, num_latent_dims, resolution, normalize_data);

    for (int i = 0; i < 50; ++i)
    {
        auto temp = som.GetDataNodePositions();

        som.Step();

        std::cout << "#iter: " << i + 1 << ", update: " << (temp - som.GetDataNodePositions()).norm() << std::endl;
    }

    return 0;
}
