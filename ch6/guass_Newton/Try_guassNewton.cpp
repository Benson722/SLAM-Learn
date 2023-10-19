#include "iostream"
#include "chrono"
#include "opencv2/opencv.hpp"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"

using namespace std;
using namespace Eigen;

int main()
{
    //Goal Function: y = exp(a * x^2 + b * x + c) . Value: a = 2, b = 2, c = 1
    //Use Guass-Newton method to solve the parameters

    //1. Get the no-noise value
    vector<double> x_true, y_noise, error;
    double guass_error;
    double sigma = 1.0;
    double inv_sigma = 1.0 / sigma;

    double a_true = 1, b_true = 2, c_true = 1;
    double a_error  = 2, b_error = -1, c_error = 5;

    cv::RNG rng;

    int N = 100; // Number of point

    for (int i = 0 ; i < N ; i++) // Initiate the dataset
    {
        double x = i/100.0; // Choose 0~1
        x_true.push_back(x);

        guass_error = rng.gaussian(sigma * sigma);
        y_noise.push_back(exp(a_true * x * x + b_true * x + c_true) + guass_error);
    }

    int iterations = 100; //inerate times
    double cost = 0, lastCost = 0; //本次迭代的cost和上一次迭代的cost  cost表示本次迭代的代价，lastCost表示上次迭代的代价

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    for (int iter = 0 ; iter < iterations ; iter++) // Guass-Newton Method
    {
        cost = 0;
        Matrix3d H = Matrix3d::Zero();
        Vector3d g = Vector3d::Zero();
        
        for (int i = 0 ; i < N ; i++) //Go through all points
        {
            Vector3d J; // 雅可比矩阵
            J[0] = -x_true[i] * x_true[i]    * exp(a_error * x_true[i] * x_true[i] + b_error * x_true[i] + c_error);  // de/da 视觉slam十四讲p133式6.40 第一个
            J[1] = -x_true[i]                          * exp(a_error * x_true[i] * x_true[i] + b_error * x_true[i] + c_error);  // de/db 视觉slam十四讲p133式6.40 第二个
            J[2] = -1                                         * exp(a_error * x_true[i] * x_true[i] + b_error * x_true[i] + c_error);  // de/dc 视觉slam十四讲p133式6.40 第三个

            double error = y_noise[i] - exp(a_error * x_true[i] * x_true[i] + b_error * x_true[i] + c_error); 

            H = H + J * inv_sigma * inv_sigma * J.transpose();
            g = g + (-J * inv_sigma * inv_sigma * error); 

            cost = cost + error * error; // 残差平方和
        }

        // Use Cholesky to solve H * delta(x) = g
        Vector3d dx = H.ldlt().solve(g);

        //Judge the existance of dx
        if(isnan(dx[0]))
        {
            cout << "No result in Cholesky! " << endl;
            break; 
        }
        if(iter > 0 && cost >= lastCost)
        {
            // iteration times >  0  and if the cost > lastcost, do NOT update 
            cout << "cost: " << cost << " >= last cost: " << lastCost << ", break." << endl;
            break;
        }

        //Updata the parameters and error
        a_error = a_error + dx[0];
        b_error = b_error + dx[1];
        c_error = c_error + dx[2];

        lastCost = cost;

        cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
         "\t\testimated params: " << a_error << "," << b_error << "," << c_error << endl;
    }
    
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);

    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
    cout << "estimated abc = " << a_error << ", " << b_error << ", " << c_error << endl;

    return 0;
}