// Copyright [2023] <Yiqing Zhang>
// 获得ORB特征 -> 手写PnP用高斯牛顿法迭代

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "sophus/se3.hpp"
#include "ctime"
#include "iostream"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

#define YELLOW  "\033[33m"      /* Yellow */
#define END_YELLOW  "\033[0m"      /* END_Yellow */

// 获得特征点和匹配子
void find_feature_matches_hand(
    const cv::Mat& img_1,
    const cv::Mat& img_2,
    std::vector<cv::KeyPoint>& keypoints_1,
    std::vector<cv::KeyPoint>& keypoints_2,
    std::vector<cv::DMatch>& matches
);

// 手写高斯牛顿法迭代求解PnP
void bundleAdjustmentGuassNewton(
        const std::vector<cv::Point3f>& points_3d,
        const std::vector<cv::Point2f>& points_2d,
        const cv::Mat &K,
        Eigen::Matrix<double, 3, 3>& R,
        Eigen::Matrix<double, 3, 1>& t);

// 像素坐标系转到相机坐标系
cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K) {
  return cv::Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                 (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

// 主函数
int main(int argc, char** argv) {
    if (argc != 5) {
        std::cout << "usage: pose_estimation_hand_3d2d img1 img2 depth1 depth2"
                  << std::endl;
        return 1;
    }
    // 读取图像
    cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;

    find_feature_matches_hand(img_1, img_2, keypoints_1, keypoints_2, matches);
    std::cout << "一共找到了" << matches.size() << "组特征点" << std::endl;

    // 下面手写求解PnP的方法，使用牛顿高斯法
    // 第1步：读入深度数据
    cv::Mat depth1 = cv::imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);
    // 相机内参
    cv::Mat K = (cv::Mat_<double>(3, 3)
                        << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    std::vector<cv::Point3f> pts_3f;
    std::vector<cv::Point2f> pts_2f;
    Sophus::SE3d poses;

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    // std::cout << "RotationMatrix : \n" << R << std::endl;

    Eigen::Matrix<double, 3, 1> t;

    for ( cv::DMatch m : matches ) {
        // 这一步应该是取得匹配点的深度，
        // queryIdx查询描述子索引，pt关键点的坐标
        // std::cout << "输出索引为：" << m.queryIdx << std::endl;
        uint16_t  d = depth1.ptr<uint16_t> (
            static_cast<int>(keypoints_1[m.queryIdx].pt.y))[
                static_cast<int>(keypoints_1[m.queryIdx].pt.x)];
        if ( d == 0 )   // bad depth
            continue;
        float dd = static_cast<float>(d/1000.0);
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        // No.1 相机坐标系
        pts_3f.push_back(cv::Point3f(static_cast<float>(p1.x*dd),
            static_cast<float>(p1.y*dd), dd));
        pts_2f.push_back(keypoints_2[m.trainIdx].pt);
    }

    std::cout << "3d-2d pairs: " << pts_3f.size() << std::endl;

    cv::Mat r1, t1;
    solvePnP(pts_3f, pts_2f, K, cv::Mat(), r1, t1,
           false);  // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    cv::Mat R1;
    cv::Rodrigues(r1, R1);  // r为旋转向量形式，用Rodrigues公式转换为矩阵

    std::cout << "R =" << std::endl << R1 << std::endl;
    std::cout << "t =" << std::endl << t1 << std::endl;
    std::cout << "Over. " << std::endl;

    bundleAdjustmentGuassNewton(pts_3f, pts_2f, K, R, t);

    return 0;
}

// 获得特征点和匹配子
void find_feature_matches_hand(const cv::Mat& img_1, const cv::Mat& img_2,
        std::vector<cv::KeyPoint>& keypoints_1,
        std::vector<cv::KeyPoint>& keypoints_2,
        std::vector<cv::DMatch>& matches) {
    // 初始化
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher =
        cv::DescriptorMatcher::create("BruteForce-Hamming");

    // 第1步：检测Oriented FAST角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // 第2步：根据角点位置计算BRIEF描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    // 第3步：根据描述子进行匹配，使用Hamming距离
    std::vector<cv::DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    // 第4步：筛选匹配对
    // double max_dist = 10000, min_dist = 0;

    // 找出所有匹配之间的最小和最大距离
    // 即是最相似和最不相似的两组点之间的距离
    // for (int i = 0; i < descriptors_1.rows; i++) {
    //     double dist = match[i].distance;
    //     if (dist < min_dist) min_dist = dist;
    //     if (dist > max_dist) max_dist = dist;
    // }

    auto min_max = minmax_element(match.begin(), match.end(),
        [](const cv::DMatch &m1, const cv::DMatch &m2)
        {return
            m1.distance < m2.distance;
        });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    std::cout << "---Max dist is " << max_dist <<std::endl;
    std::cout << "---Min dist is " << min_dist <<std::endl;

    // 当描述子之间的距离大于2倍的min_dist时，
    // 认为是误匹配，min_dist的值不小于30.0（工程值）
    for (int i = 0; i < descriptors_1.rows ; i++) {
        if (match[i].distance <= std::max(2 * min_dist , 30.0)) {
            matches.push_back(match[i]);
        }
    }
    }

void bundleAdjustmentGuassNewton(
        const std::vector<cv::Point3f>& points_3d,
        const std::vector<cv::Point2f>& points_2d,
        const cv::Mat &K,
        Eigen::Matrix<double, 3, 3>& R,
        Eigen::Matrix<double, 3, 1>& t) {
    // 创建模板Vector6d
    typedef Eigen::Matrix<double, 6, 1> Vector6d;

    // 初始化
    const int iterations = 10;
    double cost = 0, lastCost = 0;
    double f_x = K.at<double>(0, 0);
    double f_y = K.at<double>(1, 1);
    double c_x = K.at<double>(0, 2);
    double c_y = K.at<double>(1, 2);

    // 建立迭代模型，跳出迭代条件：
    // 1. LDLT无解
    // 2. 本次迭代重投影误差大于上次迭代误差
    // 3. 得出的delta_x 小于1e-6，结束迭代
    for (int iter = 0; iter < iterations; iter++) {
        // 第1步：定义增量方程中的H，b矩阵，同时令cost归零
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();
        cost = 0;

        // 第2步：计算cost, H, b
        for (int i = 0; i < points_3d.size(); i++) {
            // 获得相机坐标系下的坐标
            Eigen::Vector3d pc;
            pc[0] =  points_3d[i].x;
            pc[1] =  points_3d[i].y;
            pc[2] =  points_3d[i].z;

            double inv_z = 1.0 / pc[2];  // 为雅可比矩阵作准备
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(f_x * pc[0] / pc[2] + c_x,
                f_y * pc[1] / pc[2] + c_y);
            Eigen::Vector2d e = Eigen::Vector2d(points_2d[i].x,
                points_2d[i].y) - proj;  // 误差

            cost = cost + e.squaredNorm();  // 高斯牛顿法的右端的误差部分

            Eigen::Matrix<double, 2, 6> J;
            J << -f_x * inv_z,
            0,
            f_x * pc[0] * inv_z2,
            f_x * pc[0] * pc[1] * inv_z2,
            -f_x - f_x * pc[0] * pc[0] * inv_z2,
            f_x * pc[1] * inv_z,
            0,
            -f_y * inv_z,
            f_y * pc[1] * inv_z2,
            f_y + f_y * pc[1] * pc[1] * inv_z2,
            - f_y * pc[0] * pc[1] * inv_z2,
            - f_y * pc[0] * inv_z;

            H += J.transpose() * J;  // 高斯牛顿法的左端
            b += - J.transpose() * e;  // 高斯牛顿法的右端
        }

        Vector6d dx;  // 高斯牛顿法的解
        dx = H.ldlt().solve(b);

        // 跳出迭代1. LDLT无解
        if (std::isnan(dx[0])) {
            std::cout << "No solution in Guass-Newton! " << std::endl;
            break;
        }

        // 2. 本次迭代重投影误差大于上次迭代误差
        if (iter > 0 && cost >= lastCost) {
            std::cout <<  YELLOW << "cost is " << cost <<
                        " .LastCost is " << lastCost << ". "
                            << END_YELLOW << std::endl;
            break;
        }

        // 更新本次cost和本次姿态
        // std::cout << "RotationMatrix : \n" << R << std::endl;
        R = Sophus::SE3d::exp(dx).rotationMatrix() * R;
        t  = Sophus::SE3d::exp(dx).translation() + t;
        lastCost = cost;

        // 保留小数点后12位
        std::cout << "iteration " << iter
            << " cost = " << std::cout.precision(12) << cost << std::endl;

        // 3. 得出的delta_x 小于1e-6，结束迭代
        if (dx.norm() < 1e-8) {
            // Converge
            break;
        }
    }

    // 输出位姿
    std::cout << "RotationMatrix : \n" << R << std::endl;
    std::cout << "TranslationMatrix : \n" << t << std::endl;
}
