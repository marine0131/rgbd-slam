#ifndef SLAM_BASE_H
#define SLAM_BASE_H

#include <fstream>
#include <vector>
#include <map>
using namespace std;

// Eigen 
#include <Eigen/Core>
#include <Eigen/Geometry>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

//pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

//camera inner params
struct CAMERA_INTRINSIC_PARAMETERS
{
    double cx, cy, fx, fy, scale;
};

// frame
struct FRAME
{
    int frameID;
    cv::Mat rgb, depth;
    cv::Mat desp; //description
    vector<cv::KeyPoint> kp; 
};

//result of pnp
struct RESULT_OF_PNP
{
    cv::Mat rvec, tvec;
    int inliers;
};

//rgb to pointcloud
PointCloud::Ptr image2PointCloud(cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera);

//point 2d to 3d
cv::Point3f point2dTo3d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera);

// compute rotation and translation
RESULT_OF_PNP estimateMotion(FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera);

// cvmat2eigen
Eigen::Isometry3d cvMat2Eigen(cv::Mat& rvec, cv::Mat& tvec);

//joinPointCloud
PointCloud::Ptr joinPointCloud(PointCloud::Ptr origin, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMETERS& camera);

class ParamReader
{
    public:
        ParamReader(string filename="./data/params.txt")
        {
            ifstream fin(filename.c_str());
            if(!fin)
            {
                cerr << "params file does not exist" << endl;
                return;
            }
            while(!fin.eof())
            {
                string str;
                getline(fin, str);
                if(str[0] == '#')
                {
                    continue;
                }
                int pos = str.find("=");
                if(pos == -1)
                    continue;
                string key = str.substr(0, pos);
                string value = str.substr(pos+1, str.length());
                data[key] = value;

                if(!fin.good())
                    break;
            }
        }

        string getData(string key)
        {
            map<string, string>::iterator iter = data.find(key);
            if(iter == data.end())
            {
                cerr << "param " << key << "not found" << endl;
                return string("NOT_FOUND");
            }
            return iter->second; //return value

        }
    public:
        map<string, string> data;
};

static inline CAMERA_INTRINSIC_PARAMETERS getDefaultCamera()
{
    ParamReader pr;
    CAMERA_INTRINSIC_PARAMETERS c;
    c.fx = atof(pr.getData("camera.fx").c_str());
    c.fy = atof(pr.getData("camera.fy").c_str());
    c.cx = atof(pr.getData("camera.cx").c_str());
    c.cy= atof(pr.getData("camera.cx").c_str());
    c.scale = atof(pr.getData("camera.scale").c_str());
    return c;
}

// color
#define RESET "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m" /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m" /* Cyan */
#define WHITE "\033[37m" /* White */
#define BOLDBLACK "\033[1m\033[30m" /* Bold Black */
#define BOLDRED "\033[1m\033[31m" /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m" /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m" /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m" /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m" /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m" /* Bold White */

#endif
