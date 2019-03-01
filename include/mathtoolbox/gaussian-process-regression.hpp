#ifndef gaussian_process_regression_hpp
#define gaussian_process_regression_hpp

#include <Eigen/Core>

namespace mathtoolbox
{
    class GaussianProcessRegression
    {
    public:
        // Construction with input data
        GaussianProcessRegression(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

        // Estimation methods
        double EstimateY(const Eigen::VectorXd& x) const;
        double EstimateVariance(const Eigen::VectorXd& x) const;

        // Hyperparameters setup methods
        void SetHyperparameters(double sigma_squared_f,
                                double sigma_squared_n,
                                const Eigen::VectorXd& length_scales);
        void PerformMaximumLikelihood(double sigma_squared_f_initial,
                                      double sigma_squared_n_initial,
                                      const Eigen::VectorXd& length_scales_initial);

        // Getter methods
        const Eigen::MatrixXd& GetX() const { return X; }
        const Eigen::VectorXd& GetY() const { return y; }

    private:

        // Data points
        Eigen::MatrixXd X;
        Eigen::VectorXd y;

        // Derivative data
        Eigen::MatrixXd K;
        Eigen::MatrixXd K_inv;

        // Hyperparameters
        double          sigma_squared_f;
        double          sigma_squared_n;
        Eigen::VectorXd length_scales;
    };
}

#endif /* gaussian_process_regression_hpp */
