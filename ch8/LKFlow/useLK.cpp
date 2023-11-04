// Copyright [2023] <Yiqing Zhang>

#include <ctime>
#include <fstream>
#include <iostream>
#include <list>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "usage: useLK path_to_dataset" << std::endl;
    return 1;
  }
  std::string path_to_dataset = argv[1];
  std::string associate_file = path_to_dataset + "/associate.txt";

  std::ifstream fin(associate_file);
  if (!fin) {
    std::cerr << "I cann't find associate.txt!" << std::endl;
    return 1;
  }

  std::string rgb_file, depth_file, time_rgb, time_depth;
  std::list<cv::Point2f> keypoints;  // 因为要删除跟踪失败的点，使用list
  cv::Mat color, depth, last_color;

  for (int index = 0; index < 100; index++) {
    fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
    color = cv::imread(path_to_dataset + "/" + rgb_file);
    depth = cv::imread(path_to_dataset + "/" + depth_file, -1);
    if (index == 0) {
      // 对第一帧提取FAST特征点
      std::vector<cv::KeyPoint> kps;
      cv::Ptr<cv::FastFeatureDetector> detector =
          cv::FastFeatureDetector::create();
      detector->detect(color, kps);
      for (auto kp : kps) keypoints.push_back(kp.pt);
      last_color = color;
      continue;
    }
    if (color.data == nullptr || depth.data == nullptr) continue;
    // 对其他帧用LK跟踪特征点
    std::vector<cv::Point2f> next_keypoints;
    std::vector<cv::Point2f> prev_keypoints;
    for (auto kp : keypoints) prev_keypoints.push_back(kp);
    std::vector<unsigned char> status;
    std::vector<float> error;

    clock_t start = clock();

    cv::calcOpticalFlowPyrLK(last_color, color, prev_keypoints, next_keypoints,
                             status, error);

    clock_t finish = clock();

    double consumeTime = static_cast<double>
        (finish-start)/CLOCKS_PER_SEC;  // 注意转换为double的位置

    std::cout << "LK Flow use time: " <<
        consumeTime << " seconds." << std::endl;
    // 把跟丢的点删掉
    int i = 0;
    for (auto iter = keypoints.begin(); iter != keypoints.end(); i++) {
      if (status[i] == 0) {
        iter = keypoints.erase(iter);
        continue;
      }
      *iter = next_keypoints[i];
      iter++;
    }
    std::cout << "tracked keypoints: " << keypoints.size() << std::endl;
    if (keypoints.size() == 0) {
      std::cout << "all keypoints are lost." << std::endl;
      break;
    }
    // 画出 keypoints
    cv::Mat img_show = color.clone();
    for (auto kp : keypoints)
      cv::circle(img_show, kp, 10, cv::Scalar(0, 240, 0), 1);
    cv::imshow("corners", img_show);
    cv::waitKey(0);
    last_color = color;
  }
  return 0;
}
