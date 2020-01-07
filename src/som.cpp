#include <Eigen/SparseCore>
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

    Eigen::SparseMatrix<double> ConvertBestMatchingUnitsIntoMat(const Eigen::VectorXi& best_matching_units,
                                                                const int              num_nodes)
    {
        const int num_data = best_matching_units.size();

        Eigen::SparseMatrix<double> B(num_nodes, num_data);

        for (int data_index = 0; data_index < num_data; ++data_index)
        {
            B.insert(best_matching_units(data_index), data_index) = 1.0;
        }

        return B;
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

    Eigen::MatrixXd CalcNeighborhoodMat(const Eigen::MatrixXd& latent_node_positions, int iter_count)
    {
        const double init_var = 0.5;
        const double min_var  = 0.1;

        constexpr double tau     = 20.0;
        constexpr double inv_tau = 1.0 / tau;

        const int    num_nodes = latent_node_positions.cols();
        const double var       = std::max(init_var * std::exp(-inv_tau * iter_count), min_var);

        Eigen::MatrixXd H(num_nodes, num_nodes);

        for (int i = 0; i < num_nodes; ++i)
        {
            for (int j = i; j < num_nodes; ++j)
            {
                const double squared_dist = (latent_node_positions.col(i) - latent_node_positions.col(j)).squaredNorm();
                const double value        = std::exp(-(1.0 / 2.0 * var) * squared_dist);

                H(i, j) = value;
                H(j, i) = value;
            }
        }

        return H;
    }
} // namespace

mathtoolbox::Som::Som(const Eigen::MatrixXd& data,
                      const int              latent_num_dims,
                      const int              resolution,
                      const bool             normalize_data)
    : m_latent_num_dims(latent_num_dims),
      m_resolution(resolution),
      m_latent_node_positions(GetLatentSpacePositions(resolution, latent_num_dims)),
      m_iter_count(0),
      m_X(data)
{
    if (normalize_data)
    {
        this->NormalizeData();
    }

    this->PerformInitialization();

    constexpr int max_iter_count = 1;

    for (int i = 0; i < max_iter_count; ++i)
    {
        this->Step();
    }
}

void mathtoolbox::Som::Step()
{
    const int num_nodes = GetNumNodes(m_resolution, m_latent_num_dims);

    const Eigen::VectorXi best_matching_units = FindBestMatchingUnits(m_X, m_Y);

    const Eigen::SparseMatrix<double> B = ConvertBestMatchingUnitsIntoMat(best_matching_units, num_nodes);

    const Eigen::MatrixXd H = CalcNeighborhoodMat(m_latent_node_positions, m_iter_count);

    const Eigen::MatrixXd BX = B * m_X.transpose();

    std::cout << H * BX << std::endl;

    ++m_iter_count;
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
