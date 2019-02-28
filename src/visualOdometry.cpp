#include <iostream>
#include <fstream>
#include <sstream>

#include "slamBase.h"

// read a frame from file
FRAME readFrame(int index, ParamReader& pr);
// measure norm of transform
double normofTransform(cv::Mat rvec, cv::Mat tvec);

int main(int argc, char** argv)
{
    ParamReader pr;
    int startIndex = atoi(pr.getData("start_index").c_str());
    int endIndex = atoi(pr.getData("end_index").c_str());
    int min_inliers = atoi(pr.getData("min_inliers").c_str());
    double max_norm = atof(pr.getData("max_norm").c_str());
    bool visualize = pr.getData("visualize_pointcloud")==string("yes");
    cout << "params: " << "\n\tmin_inliers: " << min_inliers <<"\n\tmax_norm: " << max_norm << endl;


    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    // first frame
    FRAME lastFrame = readFrame(startIndex, pr);
    orb->detectAndCompute(lastFrame.rgb, cv::Mat(), lastFrame.kp, lastFrame.desp);

    // cloud
    PointCloud::Ptr cloud = image2PointCloud(lastFrame.rgb, lastFrame.depth, camera);
    // view
    pcl::visualization::CloudViewer viewer("viewer");

    for (int index = startIndex+1; index <=endIndex; index ++)
    {
        cout << "processing: image " << index << endl;
        FRAME currFrame = readFrame(index, pr);
        // create and detect keypoints
        orb->detectAndCompute(currFrame.rgb, cv::Mat(), currFrame.kp, currFrame.desp);
        //solve pnp
        RESULT_OF_PNP result = estimateMotion(lastFrame, currFrame, camera);

        cout << "inliers: " << result.inliers << endl;

        // not enough inliers
        if(result.inliers < min_inliers)
            continue;

        // move too fast
        double norm = normofTransform(result.rvec, result.tvec);
        if(norm > max_norm)
            continue;

        // to transform matrix in eigen
        Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec);

        // merge cloud
        cloud = joinPointCloud(cloud, currFrame, T, camera);

        // visualize
        if (visualize == true)
            viewer.showCloud(cloud);

        // next frame
        lastFrame = currFrame;
    }
    pcl::io::savePCDFile("data/result.pcd", *cloud);
    return 0;
}
FRAME readFrame(int index, ParamReader& pr)
{
    FRAME f;
    string rgbDir = pr.getData("rgb_dir");
    string depthDir = pr.getData("depth_dir");

    string rgbExt = pr.getData("rgb_extension");
    string depthExt = pr.getData("depth_extension");

    stringstream ss;
    // read rgb
    ss << rgbDir << index << rgbExt;
    string filename;
    ss >> filename;
    f.rgb = cv::imread(filename);

    // read depth
    ss.clear();
    filename.clear();
    ss << depthDir << index << depthExt;
    ss >> filename;
    f.depth = cv::imread(filename, -1);

    return f;
}

double normofTransform(cv::Mat rvec, cv::Mat tvec)
{
    return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec))) + fabs(cv::norm(tvec));
}
