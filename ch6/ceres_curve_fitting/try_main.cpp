#include "iostream"
#include "opencv2/core/core.hpp"
#include "ceres/ceres.h"
#include "chrono"

using namespace std;

//Define the calculation model
struct CURVE_FITTING_COST
{
    CURVE_FITTING_COST ( double x, double y ) : _x( x ), _y( y ){}
    // Calculate the residual
    template <typename T>
    bool operator()(
        const T* const abc,
        T* residual) const
    {
        residual[0] = T(_y) - ceres::exp( abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }
    const double _x,_y; 
};

int main()
{
    double at = 1.0, bt = 2.0, ct = 1.0; // True parameter
    int N = 100;                                        // data point
    double w_sigma = 1.0;                 // noise_sigma
    cv::RNG rng;                                      // OpenCV Random 
    double abc[3] = {0,0,0};                // Estimated parameter

    vector<double> x_data, y_data;

    cout << "Generate Dataset ..." << endl;

    for (int i = 0 ; i < N ; i++)
    {
        double x = i / 100.0;
        x_data.push_back( x );
        y_data.push_back(
            exp(at * x * x + bt * x + ct) + rng.gaussian(w_sigma * w_sigma)
        );
        // cout << x_data[i] << "" << y_data[i] << endl;
    }

    // Build the minimize Least Square method
    ceres::Problem problem;
    for (int i = 0 ; i < N ; i++ )
    {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3> (
                new CURVE_FITTING_COST ( x_data[i], y_data[i])
            ),
            nullptr,
            abc
        );
    }

    //Set Solver Machine
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR; 
    options.minimizer_progress_to_stdout = true; // Output to cout

    ceres::Solver::Summary summary; 

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve (options, &problem, &summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2-t1);
    cout << "Solve time cost = " << time_used.count() << " seconds. " << endl;

    //Output the Result
    cout << summary.BriefReport() << endl;
    cout << "Estimated a, b, c = ";
    for (auto a:abc)
    {
        cout << a << " ";
    }
    cout << endl;

    return 0;
}