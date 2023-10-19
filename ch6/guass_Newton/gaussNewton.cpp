#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>//Eigen核心模块
#include <Eigen/Dense>//Eigen稠密矩阵运算模块
 
using namespace std;
using namespace Eigen;
//高斯牛顿法拟合曲线y = exp(a * x^2 + b * x + c)
int main(int argc, char **argv) {
  double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
  double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值,并赋初始值
  int N = 100;                                 // 数据点
  double w_sigma = 1.0;                        // 噪声Sigma值
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng;                                 // OpenCV随机数产生器 RNG为OpenCV中生成随机数的类，全称是Random Number Generator
 
  vector<double> x_data, y_data;      // double数据x_data, y_data
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;//相当于x范围是0-1
    x_data.push_back(x);//x_data存储的数值
    y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));//rng.gaussian(w_sigma * w_sigma)为opencv随机数产生高斯噪声
   //rng.gaussian(val)表示生成一个服从均值为0，标准差为val的高斯分布的随机数  视觉slam十四讲p133式6.38上面的表达式
  }
 
  // 开始Gauss-Newton迭代 求ae，be和ce的值，使得代价最小
  int iterations = 300;    // 迭代次数
  double cost = 0, lastCost = 0;  // 本次迭代的cost和上一次迭代的cost  cost表示本次迭代的代价，lastCost表示上次迭代的代价
   //cost = error * error，error表示测量方程的残差
 
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//std::chrono是c++11引入的日期处理库，其中包含三种时钟（system_clock,steady_clock,high_resolution_clock）
  //t1表示steady_clock::time_point类型
  for (int iter = 0; iter < iterations; iter++) {
 
    Matrix3d H = Matrix3d::Zero(); // Hessian = J^T W^{-1} J in Gauss-Newton 将矩阵H初始化为3*3零矩阵，表示海塞矩阵，H = J * (sigma * sigma).transpose() * J.transpose() 
    //（视觉slam十四讲p133式6.41左边）
    Vector3d b = Vector3d::Zero(); // bias 将b初始化为3*1零向量，b = -J * (sigma * sigma).transpose() * error，error表示测量方程的残差（视觉slam十四讲p133式6.41右边）
    cost = 0;
    //遍历所有数据点计算H，b和cost
    for (int i = 0; i < N; i++) {
      double xi = x_data[i], yi = y_data[i];  // 第i个数据点
      double error = yi - exp(ae * xi * xi + be * xi + ce);//视觉slam十四讲p133式6.39
      Vector3d J; // 雅可比矩阵
      J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/da 视觉slam十四讲p133式6.40 第一个
      J[1] = -xi * exp(ae * xi * xi + be * xi + ce);  // de/db 视觉slam十四讲p133式6.40 第二个
      J[2] = -exp(ae * xi * xi + be * xi + ce);  // de/dc 视觉slam十四讲p133式6.40 第三个
 
      H += inv_sigma * inv_sigma * J * J.transpose();//视觉slam十四讲p133式6.41左边 求和
      b += -inv_sigma * inv_sigma * error * J;//视觉slam十四讲p133式6.41右边 求和
 
      cost += error * error;//残差平方和
    }
 
    // 求解线性方程 Hx=b
    Vector3d dx = H.ldlt().solve(b); //ldlt()表示利用Cholesky分解求dx
    if (isnan(dx[0]))//isnan()函数判断输入是否为非数字，是非数字返回真，nan全称为not a number
     {
      cout << "result is nan!" << endl;
      break;
    }
 
    if (iter > 0 && cost >= lastCost) //因为iter要大于0，第1次迭代(iter = 0, cost > lastCost)不执行！
    {
      cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
      break;
    }
   //更新优化变量ae，be和ce！
    ae += dx[0];
    be += dx[1];
    ce += dx[2];
 
    lastCost = cost; //更新上一时刻代价
 
    cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
         "\t\testimated params: " << ae << "," << be << "," << ce << endl;
  }
 
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
 
  cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
  return 0;
}
