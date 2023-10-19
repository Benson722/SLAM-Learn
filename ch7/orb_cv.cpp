#include "iostream"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "chrono"

using namespace std;
using namespace cv;

int main (int argc, char** argv)
{
    if( argc != 3)
    {
        cout << "usage: feature_extraction img1 img2 " << endl;
        return 1;
    }

    //Read the pics
    Mat img_1 = imread( argv[1] , CV_LOAD_IMAGE_ANYCOLOR);
    Mat img_2 = imread( argv[2] , CV_LOAD_IMAGE_ANYCOLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    //Initialization
    std::vector<cv::KeyPoint> key_point1, key_point2;
    Mat descriptor_1, descriptor_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor > descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // Step 1: Detect the corner position (Oriented FAST)
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector ->detect(img_1, key_point1);
    detector ->detect(img_2, key_point2);

    //Step 2: Calculate the BRIEF descriptor Accoring to corner position
    detector ->compute(img_1, key_point1, descriptor_1);
    detector ->compute(img_2, key_point2, descriptor_2);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
    cout << "Extract ORB cost = " << time_used.count() << " seconds. " <<endl;

    Mat outimg_1;
    drawKeypoints(img_1, key_point1, outimg_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB features", outimg_1);

    //Step 3: Match the BRIEF in 2 pics, using Hamming distance
    vector<DMatch> matches;
    t1 = chrono::steady_clock::now();
    matcher->match(descriptor_1, descriptor_2, matches);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
    cout << "Match ORB cost = " << time_used.count() << " seconds. " <<endl;

    //Step 4: 匹配点对筛选
    //计算最小距离和最大距离
    auto min_max = minmax_element(matches.begin(), matches.end(),
        [](const DMatch &m1, const DMatch &m2) {return m1.distance < m2.distance ;} );
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f  \n", max_dist);
    printf("-- Min dist : %f  \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时，即认为匹配有误。但有时最小距离非常小，
    //所以设置经验值30作为下限。
    vector<DMatch> good_matches;
    for (int i = 0; i < descriptor_1.rows; i++)
    {
        if(matches[i].distance <= max(2 * min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    std::cout << "good_matches Num: "  << good_matches.size() << std::endl;

    //Step 5: Draw the result
    Mat img_match;
    Mat img_goodMatch;
    drawMatches(img_1, key_point1, img_2, key_point2, matches, img_match);
    drawMatches(img_1, key_point1, img_2, key_point2, good_matches, img_goodMatch);
    imshow("all matches", img_match);
    imshow("good matches", img_goodMatch);
    waitKey(0);

    return 0;
}