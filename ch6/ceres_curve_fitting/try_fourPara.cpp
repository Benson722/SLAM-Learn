#include "iostream"
#include "opencv2/core/core.hpp"
#include "ceres/ceres.h"
#include "chrono"

using namespace std;

struct CostFunctor
{
    CostFunctor(double x, double y): _x(x), _y(y) {}

    template <typename T>
    bool operator()(const T* const abcd, T* residual) const{
        // y - exp(a * x^3 + b * x^2 + c * x + d)
        residual[0] = T(_y) - ceres::exp(abcd[0] * T(_x) * T(_x)* T(_x) + abcd[1] * T(_x) * T(_x) + abcd[2] * T(_x) + abcd[3] );
        return true;
    }

    const double _x, _y;
};

int main(int argc, char** argv)
{
    google::InitGoogleLogging( argv[0] );

    //Initialize the parameter
    double initial_a = 4.0, initial_b = 3.0, initial_c = 2.0, initial_d = 1.0;
    double true_a    = 1.0, true_b     = 2.0,  true_c   = 3.0,    true_d = 4.0;
    int MaxIterationTimes = 100;
    int Nums = 100;
    double w_sigma = 1.0;
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;

    //Generate the data
    vector<double> x_data,y_data; 
    double abcd[4] = {initial_a, initial_b, initial_c, initial_d};

    for (int num = 0; num < Nums; num++)
    {
        double x = num / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(initial_a *  x * x * x + initial_b * x * x + initial_c * x + initial_d) + rng.gaussian(w_sigma * w_sigma));
    }

    //Build the problem
    ceres::Problem problem;

    //Set up the only cost function (also known as residual ). This uses
    //auto-differentiation to obtain the derivative (Jocobian).
    for (int num = 0; num < Nums; num++ )
    {
        ceres::CostFunction* cost_function = 
            new ceres::AutoDiffCostFunction<CostFunctor, 1, 4>(
                new CostFunctor(x_data[num], y_data[num]));
        problem.AddResidualBlock(cost_function, nullptr, abcd);
    }

    //Run the Solver! 
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<
        chrono::duration<double>> (t2 - t1);

    cout << "The time cost = " << time_used.count() << " seconds. " << endl;
    cout << summary.BriefReport() <<endl;
    cout << "Estimated Parameters a, b, c, d = "; 
    for (auto para:abcd ) cout << para << " " ;           
    cout << endl;

    return 0;
}