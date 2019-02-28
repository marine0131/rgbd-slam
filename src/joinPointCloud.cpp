#include <iostream>
//eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/eigen.hpp>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

#include "slamBase.h"

int main(int argc, char** argv)
{
    ParamReader pr;
    FRAME f1, f2;

    f1.rgb = cv::imread("./data/rgb1.png");
    f1.depth = cv::imread("./data/depth1.png");
    f2.rgb = cv::imread("./data/rgb2.png");
    f2.depth = cv::imread("./data/depth2.png");


    cout << "extracting feature ..." << endl;
    // create orb detector and dexcriptor                                       
    cv::Ptr<cv::ORB> orb = cv::ORB::create();  
    // create and detect keypoints for two image                                
    orb->detectAndCompute(f1.rgb, cv::Mat(), f1.kp, f1.desp);  
    orb->detectAndCompute(f2.rgb, cv::Mat(), f2.kp, f2.desp);  

    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();

    cout << "solving pnp ..." << endl;
    RESULT_OF_PNP result = estimateMotion(f1, f2, camera);
    cout<<result.rvec<<endl<<result.tvec<<endl;

    //vec --> rotation matrix --> transform matrix
    cv::Mat R;
    cv::Rodrigues(result.rvec, R);
    Eigen::Matrix3d r; //3x3 matrix
    cv::cv2eigen(R, r);

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity(); //4x4 transform matrix
    Eigen::AngleAxisd rotate_vec(r); //3x1 rotation vec 
    T.rotate(r); //fill rotate matrix
    T(0,3) = result.tvec.at<double>(0,0);
    T(1,3) = result.tvec.at<double>(0,1);
    T(2,3) = result.tvec.at<double>(0,2);

    //transform pointcloud
    cout << "converting image to cloud" << endl;
    PointCloud::Ptr cloud1 = image2PointCloud(f1.rgb, f1.depth, camera);
    PointCloud::Ptr cloud2 = image2PointCloud(f2.rgb, f2.depth, camera);

    //merge point cloud
    cout << "merging cloud" << endl;
    PointCloud::Ptr output(new PointCloud);
    pcl::transformPointCloud(*cloud1, *output, T.matrix());
    *output += *cloud2;
    pcl::io::savePCDFile("data/output.pcd", *output);
    cout << "pcd file saved" << endl;

    pcl::visualization::CloudViewer viewer("view");
    viewer.showCloud(output);
    while(!viewer.wasStopped())
        ;

    return 0;
}
