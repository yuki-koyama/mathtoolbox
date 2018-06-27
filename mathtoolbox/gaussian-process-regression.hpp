#ifndef gaussian_process_regression_hpp
#define gaussian_process_regression_hpp

#include <Eigen/Core>

namespace mathtoolbox
{
    class GaussianProcessRegression
    {
    public:
        
        GaussianProcessRegression(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
        
        // Estimation methods
        double EstimateY(const Eigen::VectorXd& x) const;
        double EstimateS(const Eigen::VectorXd& x) const;
        
        // Hyperparameters setup methods
        void SetHyperparameters(double s_f_squared, double s_n_squared, const Eigen::VectorXd& l);
        void PerformMaximumLikelihood(double s_f_squared_initial, double s_n_squared_initial, const Eigen::VectorXd& l_initial);

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
        double          s_f_squared;
        double          s_n_squared;
        Eigen::VectorXd l;
    };
}

#endif /* gaussian_process_regression_hpp */
