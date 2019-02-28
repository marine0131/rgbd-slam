#include <iostream>
#include "slamBase.h"
using namespace std;

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

//#define DEBUG

int main(int argc, char** argv)
{
    FRAME f1, f2;
    f1.rgb = cv::imread("./data/rgb1.png");
    f2.rgb = cv::imread("./data/rgb2.png");
    f1.depth = cv::imread("./data/depth1.png", -1);
    f2.depth = cv::imread("./data/depth2.png", -1);

    // create orb detector and dexcriptor
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    // create and detect keypoints for two image
    orb->detectAndCompute(f1.rgb, cv::Mat(), f1.kp, f1.desp);
    orb->detectAndCompute(f2.rgb, cv::Mat(), f2.kp, f2.desp);

#ifdef DEBUG
    // print and show key points
    cout << "Key points of two image: " << f1.kp.size()<<", "<<f2.kp.size() <<endl;
    cv::Mat imgShow;
    cv::drawKeypoints(f1.rgb, f1.kp, imgShow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("keypoints", imgShow);
    cv::waitKey(0);
#else
#endif

    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();

    //solve pnp
    RESULT_OF_PNP pnp_result = estimateMotion(f1, f2, camera);

    cout << "inliers = " << pnp_result.inliers << endl;
    cout << "rvec = " << pnp_result.rvec << endl;
    cout << "tvec = " << pnp_result.tvec << endl;
    return 0;
}
