#ifndef RBF_INTERPOLATION_HPP
#define RBF_INTERPOLATION_HPP

#include <vector>

namespace mathtoolbox
{
    enum class FunctionType
    {
        Gaussian,         // f(r) = exp(-(epsilon * r)^2)
        ThinPlateSpline,  // f(r) = (r^2) * log(r)
        InverseQuadratic, // f(r) = (1 + (epsilon * r)^2)^(-1)
        BiharmonicSpline, // f(r) = r
    };
    
    class Interpolator
    {
    public:
        Interpolator(FunctionType functionType = FunctionType::BiharmonicSpline, const double epsilon = 2.0);
        
        // API
        void   reset();
        void   addCenterPoint(const double y, const std::vector<double>& x);
        void   computeWeights(const bool useRegularization = false, const double lambda = 0.1);
        double getInterpolatedValue(const std::vector<double>& x) const;
        
        // Getter methods
        const std::vector<double>&              getYs() const { return ys; }
        const std::vector<std::vector<double>>& getXs() const { return xs; }
        const std::vector<double>&              getW()  const { return w;  }
        
    private:
        
        // Function type
        FunctionType functionType;
        
        // A control parameter used in some kernel functions
        double epsilon;
        
        // Data points
        std::vector<double>              ys;
        std::vector<std::vector<double>> xs;
        
        // Weights
        std::vector<double>              w;
        
        // Returns f(r)
        double getRbfValue(const double r) const;
        
        // Returns f(||xj - xi||)
        double getRbfValue(const std::vector<double>& xi, const std::vector<double>& xj) const;
    };
}

#endif // RBF_INTERPOLATION_HPP
