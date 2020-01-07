#include <iostream>
#include <mathtoolbox/som.hpp>

namespace
{
    int GetNumNodes(const int resolution, const int latent_num_dims)
    {
        assert(latent_num_dims == 1 || latent_num_dims == 2);

        if (latent_num_dims == 1)
        {
            return resolution;
        }
        else
        {
            return resolution * resolution;
        }
    }

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

    /// \return Posititons in the latent space [0, 1]^{#dims x #nodes}
    Eigen::MatrixXd GetLatentSpacePositions(const int resolution, const int latent_num_dims)
    {
        assert(latent_num_dims == 1 || latent_num_dims == 2);
        assert(resolution >= 2);

        const int num_nodes = GetNumNodes(resolution, latent_num_dims);

        Eigen::MatrixXd positions(latent_num_dims, num_nodes);

        if (latent_num_dims == 1)
        {
            for (int x_index = 0; x_index < num_nodes; ++x_index)
            {
                positions(0, x_index) = static_cast<double>(x_index) / static_cast<double>(resolution - 1);
            }
        }
        else
        {
            for (int y_index = 0; y_index < resolution; ++y_index)
            {
                for (int x_index = 0; x_index < resolution; ++x_index)
                {
                    const int index = y_index * resolution + x_index;

                    const double x_offset = static_cast<double>(x_index) / static_cast<double>(resolution - 1);
                    const double y_offset = static_cast<double>(y_index) / static_cast<double>(resolution - 1);

                    positions.col(index) = Eigen::Vector2d(x_offset, y_offset);
                }
            }
        }

        return positions;
    }

    Eigen::MatrixXd CalculateNeighborhoodMatrix(const int resolution, const int latent_num_dims)
    {
        // TODO: Implement this function
        assert(false);
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

    const Eigen::MatrixXd latent_node_positions = GetLatentSpacePositions(m_resolution, m_latent_num_dims);

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
    const int num_data_dims = m_X.rows();
    const int num_nodes     = GetNumNodes(m_resolution, m_latent_num_dims);

    m_Y = Eigen::MatrixXd::Random(num_data_dims, num_nodes);
}
