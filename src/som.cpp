#include <iostream>
#include <mathtoolbox/som.hpp>

namespace
{
    /// \details The computational complexity is O(#data * #nodes)
    Eigen::VectorXi FindBestMatchingUnits(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
    {
        assert(X.rows() == Y.rows());

        const int num_data  = X.cols();
        const int num_nodes = Y.cols();

        Eigen::VectorXi node_indices(num_data);

        for (int i = 0; i < num_data; ++i)
        {
            Eigen::VectorXd squared_norms(num_nodes);

            for (int j = 0; j < num_nodes; ++j)
            {
                squared_norms(j) = (X.col(i) - Y.col(j)).squaredNorm();
            }

            squared_norms.minCoeff(&(node_indices(i)));
        }

        return node_indices;
    }
} // namespace

mathtoolbox::Som::Som(const Eigen::MatrixXd& data,
                      const int              latent_num_dims,
                      const int              resolution,
                      const bool             normalize_data)
    : m_latent_num_dims(latent_num_dims), m_resolution(resolution), m_X(data)
{
    if (normalize_data)
    {
        this->NormalizeData();
    }

    this->PerformInitialization();

    const Eigen::VectorXi best_matching_units = FindBestMatchingUnits(m_X, m_Y);

    std::cout << best_matching_units << std::endl;
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
