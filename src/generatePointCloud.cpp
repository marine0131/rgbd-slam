#include <iostream>
#include <string>
using namespace std;

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;


const double camera_factor = 1000; //mm
const double camera_cx = 325.5;
const double camera_cy = 253.5;
const double camera_fx = 518.0;
const double camera_fy = 519.0;


int main( int argc, char** argv)
{
    // read 2 picture and transfer to point cloud
    cv::Mat rgb, depth;
    rgb = cv::imread("./data/rgb.png");
    depth = cv::imread("./data/depth.png");

    PointCloud::Ptr cloud(new PointCloud);

    for (int m=0; m < depth.rows; m ++)
        for (int n = 0; n < depth.cols; n ++)
        {
            ushort d = depth.ptr<ushort>(m)[n];
            if(d==0)
                continue;

            PointT p;
            p.z = double(d)/camera_factor;
            p.x = (n - camera_cx) * p.z / camera_fx;
            p.y = (m - camera_cy) * p.z / camera_fy;

            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];
        
            cloud->points.push_back(p);
        }

    //save cloud
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cout << "pointcloud size="<<cloud->points.size()<<endl;
    cloud->is_dense=false;
    pcl::io::savePCDFile("./data/pointcloud.pcd", *cloud);
    cloud->points.clear();
    cout << "pointcloud saved" << endl;
    return 0;

}

