#include "iostream"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "chrono"

using namespace std;
using namespace cv;

// compute the descriptor
typedef vector<uint32_t> DescType;
void ComputeORB(const cv::Mat &img, vector<cv::KeyPoint> &keypoints, vector<DescType> & descriptors)
{
    const int half_patch_size = 8;
    const int half_boundary = 16;
    int bad_points = 0;
    for (auto &kp:keypoints)
    {
        if (kp.pt.x < half_boundary || kp.pt.y < half_boundary || 
        kp.pt.x >= img.cols - half_boundary || kp.pt.y >= img.rows - half_boundary)
        {
            // outsides
            bad_points++;
            descriptors.push_back({});
            continue;
        }

        float m01 = 0, m10 = 0;
        for(int dx = -half_patch_size ; dx < half_patch_size ; dx++)
        {
            for (int dy = - half_patch_size ; dy < half_patch_size ; dy++)
            {
                uchar pixel = img.at<uchar>(kp.pt.y + dy , kp.pt.x + dx);
                m01 += dx * pixel;
                m10 += dy * pixel;
            }
        }

        // angle should be arc tan(m01/m10)
        

    }


}


int main( int argc, char** argv)
{
    if( argc != 3)
    {
        cout << "usage: feature_extraction img1 img2 " << endl;
        return 1;
    }



    return 0;
}