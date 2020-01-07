#include <mathtoolbox/som.hpp>

int main()
{
    constexpr int  num_data_dims   = 3;
    constexpr int  num_points      = 10;
    constexpr int  num_latent_dims = 2;
    constexpr int  resolution      = 10;
    constexpr bool normalize_data  = false;

    Eigen::MatrixXd data = Eigen::MatrixXd::Random(num_data_dims, num_points);

    mathtoolbox::Som som(data, num_latent_dims, resolution, normalize_data);

    return 0;
}
