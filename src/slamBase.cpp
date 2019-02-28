#include "slamBase.h"
#include <time.h>

// image2PointCloud
// input: rgb image, depth image
// output: cloud
PointCloud::Ptr image2PointCloud(cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    PointCloud::Ptr cloud(new PointCloud);

    for (int m=0; m < depth.rows; m ++)
        for (int n = 0; n < depth.cols; n ++)
        {
            ushort d = depth.ptr<ushort>(m)[n];
            if(d==0)
                continue;

            PointT p;
            p.z = double(d)/camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;

            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];
        
            cloud->points.push_back(p);
        }

    //save cloud
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense=false;

    return cloud;
}

// point2d to 3d
// input: point in image but has depth
// output: point3d in camera coor
cv::Point3f point2dTo3d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    cv::Point3f p;
    p.z = double(point.z)/camera.scale;
    p.x = (point.x - camera.cx) * p.z / camera.fx;
    p.y = (point.y - camera.cy) * p.z / camera.fy;
    return p;
}

// estimatemotion
// input: frame1 and frame 2
// output: rvec and tvec
RESULT_OF_PNP estimateMotion(FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    static ParamReader param;
    RESULT_OF_PNP result;
    vector<cv::DMatch> matches;
    cv::BFMatcher matcher;
    matcher.match(frame1.desp, frame2.desp, matches);
    // cout << "Find matches: " << matches.size() << endl;

#ifdef DEBUG
    // show matches
    cv::Mat imgMatches;
    cv::drawMatches(frame1.rgb, frame1.kp, frame2.rgb, frame2.kp, matches, imgMatches);
    cv::imshow("match", imgMatches);
    cv::imwrite("./data/matches.png", imgMatches);
    cv::waitKey(0);
#else
#endif

    // filter matches
    vector <cv::DMatch> goodMatches;
    double minDis = 9999;  //init minDis
    double good_match_threshold = atof(param.getData("good_match_threshold").c_str());
    for (size_t i =0 ; i<matches.size(); i++)
    {
        if(matches[i].distance < minDis)
            minDis=matches[i].distance;
    }
    for (size_t i = 0 ; i < matches.size(); i++)
    {
        if (matches[i].distance < good_match_threshold * minDis)
            goodMatches.push_back(matches[i]);
    }

    // cout << "good matches: " << goodMatches.size() << endl;
#ifdef DEBUG
    // show good matches
    cv::Mat imgMatches;
    cv::drawMatches(frame1.rgb, frame1.kp, frame2.rgb, frame2.kp, goodMatches, imgMatches);
    cv::imshow("good match", imgMatches);
    cv::imwrite("./data/good_matches.png", imgMatches);
    cv::waitKey(0);
#else
#endif
    // good matches not enough for pnp
    if (goodMatches.size() <= 5)
    {
        result.inliers = -1;
        return result;
    }

    // calc rotation and translation between two image
    vector<cv::Point3f> pts_obj; //3d points of first image
    vector<cv::Point2f> pts_img; //2d points of second image

    for(size_t i = 0; i < goodMatches.size(); i++)
    {
        cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;
        ushort d = frame1.depth.ptr<ushort>(int(p.y))[int(p.x)];
        if (d == 0)
            continue;
        // (u,v,d) -->(x, y, z) of frame1
        cv::Point3f pt (p.x, p.y, d);
        cv::Point3f pd = point2dTo3d(pt, camera);
        pts_obj.push_back(pd); //3d points

        // get (u,v) of frame2
        pts_img.push_back(cv::Point2f(frame2.kp[goodMatches[i].trainIdx].pt)); //2d points in image
    }
    // pts are not enough for pnp
    if (pts_img.size()< 4 || pts_obj.size() < 4)
    {
        result.inliers = -1;
        return result;
    }

    double camera_matrix_data[3][3] = {
        {camera.fx, 0 , camera.cx},
        {0, camera.fy, camera.cy},
        {0, 0, 1}
    };

    // create camera matrix
    cv::Mat cameraMatrix(3, 3, CV_64F, camera_matrix_data);
    cv::Mat rvec, tvec, inliers;
    //solve pnp
    cv::solvePnPRansac(pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 0.99, inliers);
    result.inliers = inliers.rows;
    result.rvec = rvec;
    result.tvec = tvec;

#ifdef DEBUG
    //draw inliers
    vector<cv::DMatch > matchesShow;
    for (size_t i = 0; i < pnp_result.inliers.rows; i++)
    {
        matchesShow.push_back(goodMatches[pnp_result.inliers.ptr<int>(i)[0]]);
    }
    cv::drawMatches(f1.rgb, f1.kp, f2.rgb, f2.kp, matchesShow, imgMatches);
    cv::imshow("inlier matches", imgMatches);
    cv::imwrite("./data/inliers.png", imgMatches);
    cv::waitKey(0);
#else
#endif

    return result;
}

// cvMat2Eigen
// input: R, t in cv format
// output: T in eigen format
Eigen::Isometry3d cvMat2Eigen(cv::Mat& rvec, cv::Mat& tvec)
{
    cv::Mat R; //rotation matrix
    cv::Rodrigues(rvec, R); //r vec to rotation matrix
    //R in cv to R in eigen format
    Eigen::Matrix3d r;
    for(size_t i = 0; i <  3; i ++)
        for(size_t j = 0; j< 3; j++)
            r(i,j) = R.at<double>(i,j);

    // R + t to T
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(r); //get rotate value
    T(0,3) = tvec.at<double>(0,0);
    T(1,3) = tvec.at<double>(1,0);
    T(2,3) = tvec.at<double>(2,0);

    return T;
}


// joinPointCloud
// input: cloud, new frame and its pose
// output: updated cloud
PointCloud::Ptr joinPointCloud(PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    PointCloud::Ptr newCloud = image2PointCloud(newFrame.rgb, newFrame.depth, camera);

    // merge cloud
    PointCloud::Ptr output(new PointCloud());
    pcl::transformPointCloud(*original, *output, T.matrix());
    *newCloud += *output;

    // vocel grid filter
    static pcl::VoxelGrid<PointT> voxel;
    static ParamReader pr;
    double gridsize = atof(pr.getData("voxel_grid").c_str());
    voxel.setLeafSize(gridsize, gridsize, gridsize);
    voxel.setInputCloud(newCloud);
    PointCloud::Ptr tmp(new PointCloud());
    voxel.filter(*tmp);
    return tmp;
}
