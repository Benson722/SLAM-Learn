// Copyright [2023] <Yiqing Zhang 583032099@qq.com>

#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include "extra.h" // use this if in OpenCV2

// using std declarations
using std::cout;
using std::endl;
using std::max;
using std::vector;

using cv::DescriptorExtractor;
using cv::DescriptorMatcher;
using cv::drawKeypoints;
using cv::DMatch;
using cv::FeatureDetector;
using cv::imread;
using cv::imshow;
using cv::KeyPoint;
using cv::Mat;
using cv::Mat_;
using cv::ORB;
using cv::Point2d;
using cv::Point2f;
using cv::Ptr;
using cv::RANSAC;
using cv::DrawMatchesFlags;
using cv::Scalar;

/****************************************************
 * 本程序演示了如何使用2D-2D的特征匹配估计相机运动
 * **************************************************/

void find_feature_matches(const Mat& img_1, const Mat& img_2,
                          vector<KeyPoint>& keypoints_1,
                          vector<KeyPoint>& keypoints_2,
                          vector<DMatch>& matches);

void pose_estimation_2d2d(vector<KeyPoint> keypoints_1,
                          vector<KeyPoint> keypoints_2, vector<DMatch> matches,
                          Mat& R, Mat& t);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d& p, const Mat& K);

int main(int argc, char** argv) {
  if (argc != 3) {
    cout << "usage: pose_estimation_2d2d img1 img2" << endl;
    return 1;
  }
  //-- 读取图像
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  //-- 估计两张图像间运动


  //-- 验证E=t^R*scale

  //-- 验证对极约束

  return 0;
}

void find_feature_matches(const Mat& img_1, const Mat& img_2,
                          std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2,
                          std::vector<DMatch>& matches) {
  //-- 初始化
  vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptor_1, descriptor_2;
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher =
    DescriptorMatcher::create("BruteForce-Hamming");

  //-- 第一步:检测 Oriented FAST 角点位置
  clock_t begin, finish;
  double costSeconds;
  begin = clock();
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步: 1)根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptor_1);
  descriptor->compute(img_2, keypoints_2, descriptor_2);
  finish = clock();
  costSeconds = static_cast<double>((finish - begin)/CLOCKS_PER_SEC);
  cout << "Extract ORB cost = "<< costSeconds << " seconds. " << endl;

  // 2) 在图像中显示特征点
  Mat out_img1;
  drawKeypoints(img_1, keypoints_1, out_img1,
    Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  imshow("ORB features", out_img1);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> matches;
  begin = clock();
  matcher->match(descriptor_1, descriptor_2, matches);
  finish = clock();
  costSeconds = static_cast<double>((finish - begin)/CLOCKS_PER_SEC);
  cout << "Match ORB cost = "<< costSeconds << " seconds. " << endl;

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  // 找出所有匹配之间的最小距离和最大距离,
  // 即是最相似的和最不相似的两组点之间的距离

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  // 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
}

Point2d pixel2cam(const Point2d& p, const Mat& K) {}

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                          std::vector<KeyPoint> keypoints_2,
                          std::vector<DMatch> matches, Mat& R, Mat& t) {
  // 相机内参,TUM Freiburg2

  //-- 把匹配点转换为vector<Point2f>的形式

  //-- 计算基础矩阵

  //-- 计算本质矩阵

  //-- 计算单应矩阵

  //-- 从本质矩阵中恢复旋转和平移信息.
}
