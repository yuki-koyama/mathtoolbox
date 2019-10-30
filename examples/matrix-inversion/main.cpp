#include <Eigen/LU>
#include <mathtoolbox/matrix-inversion.hpp>
#include <memory>
#include <timer.hpp>

using Eigen::MatrixXd;

inline bool CheckInvertibleness(const MatrixXd& matrix)
{
    const Eigen::FullPivLU<MatrixXd> lu(matrix);
    return lu.isInvertible();
}

int main(int argc, char** argv)
{
    constexpr int size = 500;

    std::shared_ptr<timer::Timer> t;

#if 1
    // Generate a test matrix
    const MatrixXd random_matrix = [](const int size) {
        while (true)
        {
            const MatrixXd candidate = MatrixXd::Random(size, size);

            if (CheckInvertibleness(candidate))
            {
                return candidate;
            }
        }
    }(size);

    constexpr int block_size = 495;

    std::cout << "Matrix size: " << size << std::endl;
    std::cout << "Block size: " << block_size << std::endl;

    // Prepare the inverse of the upper left block
    const Eigen::MatrixXd block_inv = random_matrix.block(0, 0, block_size, block_size).inverse();

    // Perform matrix inversion in the naive direct approach
    t = std::make_shared<timer::Timer>("Naive approach");

    const MatrixXd naive_result = random_matrix.inverse();

    // Perform matrix inversion in the block matrix approach
    t = std::make_shared<timer::Timer>("Block approach");

    const MatrixXd block_result = mathtoolbox::GetInverseUsingUpperLeftBlockInverse(random_matrix, block_inv);

    // Stop the timer
    t = nullptr;

    // Error check
    if (!naive_result.isApprox(block_result, 1e-06))
    {
        throw std::runtime_error("The results are not consistent.");
    }
#else
    const int start_size = 100;

    MatrixXd test_matrix     = MatrixXd::Random(start_size, start_size);
    MatrixXd test_matrix_inv = test_matrix.inverse();
    for (int i = test_matrix.rows(); i < size; ++i)
    {
        const MatrixXd new_test_matrix = [&]() {
            MatrixXd mat(i + 1, i + 1);

            mat.block(0, 0, i, i) = test_matrix; // A
            while (true)
            {
                mat.block(0, i, i, 1) = MatrixXd::Random(i, 1); // B
                mat.block(i, 0, 1, i) = MatrixXd::Random(1, i); // C
                mat.block(i, i, 1, 1) = MatrixXd::Random(1, 1); // D

                const MatrixXd sub_matrix =
                    mat.block(i, i, 1, 1) - mat.block(i, 0, 1, i) * test_matrix_inv * mat.block(0, i, i, 1);

                if (CheckInvertibleness(sub_matrix))
                {
                    break;
                }
            }

            return mat;
        }();

        std::cout << "Matrix size: " << std::to_string(i + 1) << std::endl;

        // Perform matrix inversion in the naive direct approach
        t = std::make_shared<timer::Timer>("Naive approach");

        const MatrixXd naive_result = new_test_matrix.inverse();

        // Perform matrix inversion in the block matrix approach
        t = std::make_shared<timer::Timer>("Block approach");

        const MatrixXd block_result =
            mathtoolbox::GetInverseUsingUpperLeftBlockInverse(new_test_matrix, test_matrix_inv);

        // Stop the timer
        t = nullptr;

        // Error check
        if (!naive_result.isApprox(block_result, 1e-06))
        {
            throw std::runtime_error("The results are not consistent.");
        }

        // Store the results for the next step
        test_matrix     = new_test_matrix;
        test_matrix_inv = naive_result;
    }
#endif

    return 0;
}
