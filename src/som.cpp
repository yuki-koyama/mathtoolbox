#include <mathtoolbox/som.hpp>

mathtoolbox::Som::Som(const Eigen::MatrixXd& m_data,
                      const int              latent_num_dims,
                      const int              resolution,
                      const bool             normalize_data)
    : m_latent_num_dims(latent_num_dims), m_resolution(resolution), m_X(m_data)
{
    if (normalize_data)
    {
        this->NormalizeData();
    }

    this->PerformInitialization();
}

void mathtoolbox::Som::NormalizeData()
{
    // TODO: Implement this function
    assert(false);
}

void mathtoolbox::Som::PerformInitialization()
{
    // TODO: Implement this function
    assert(false);
}
