#include <iostream>
#include <fstream>
#include <sstream>

#include "slamBase.h"
#include <pcl/filters/passthrough.h>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

// read a frame from file
FRAME readFrame(int index, ParamReader& pr);
// measure norm of transform
double normofTransform(cv::Mat rvec, cv::Mat tvec);
// result of comparison of two frame
enum CHECK_RESULT {NOT_MATCHED=0, TOO_FAR_AWAY, TOO_CLOSE, KEYFRAME};
//check key frame
CHECK_RESULT checkKeyframes(FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops=false);
// nearby loop closure
void checkNearbyLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti);
// random loop closure
void checkRandomLoops(vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti);

// g2o def
// typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> SlamBlockSolver;
typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

int main(int argc, char** argv)
{
    ParamReader pr;
    int startIndex = atoi(pr.getData("start_index").c_str());
    int endIndex = atoi(pr.getData("end_index").c_str());
    bool check_loop_closure = pr.getData("check_loop_closure")==string("yes");
    bool visualize = pr.getData("visualize_pointcloud")==string("yes");
    double gridsize = atof(pr.getData("voxel_grid").c_str());

    // key frames
    vector<FRAME> keyframes;

    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    // first frame
    FRAME lastFrame = readFrame(startIndex, pr);
    orb->detectAndCompute(lastFrame.rgb, cv::Mat(), lastFrame.kp, lastFrame.desp);

    // cloud
    PointCloud::Ptr cloud = image2PointCloud(lastFrame.rgb, lastFrame.depth, camera);

    // initialize g2o
    unique_ptr<SlamLinearSolver> linearSolver (new SlamLinearSolver());
    linearSolver->setBlockOrdering(false);
    unique_ptr<SlamBlockSolver> blockSolver (new SlamBlockSolver(move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(move(blockSolver));

    g2o::SparseOptimizer globalOptimizer;
    globalOptimizer.setAlgorithm(solver);
    globalOptimizer.setVerbose(false);

    // add first Vertex(point)
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId(startIndex);
    v->setEstimate(Eigen::Isometry3d::Identity()); // estimate a identity
    v->setFixed(true); // first vertex need no optimization
    globalOptimizer.addVertex(v);

    // add frist keyframe
    keyframes.push_back(lastFrame);

    int lastIndex = startIndex;

    for (int index = startIndex+1; index < endIndex; index ++)
    {
        cout << RESET"processing: image " << index << endl;
        FRAME currFrame = readFrame(index, pr);
        // create and detect keypoints
        orb->detectAndCompute(currFrame.rgb, cv::Mat(), currFrame.kp, currFrame.desp);

        // check key frame
        CHECK_RESULT result = checkKeyframes(keyframes.back(), currFrame, globalOptimizer, false);
        switch(result){
            case NOT_MATCHED:
                // not match, return
                cout << RED"not enough inliers" << endl;
                break;
            case TOO_FAR_AWAY:
                // too far,break
                cout << RED"Too far away" << endl;
                break;
            case TOO_CLOSE:
                // too close
                cout << RED"too close" << endl;
                break;
            case KEYFRAME:
                cout << GREEN"it is a new keyframe" << endl;

                if(check_loop_closure)
                {
                    checkNearbyLoops(keyframes, currFrame, globalOptimizer);
                    checkRandomLoops(keyframes, currFrame, globalOptimizer);
                }
                keyframes.push_back(currFrame);
                break;
            default:
                break;
        }
    }

    // optimize all vertex
    cout << "optimizing pose graph..., vertices: " <<globalOptimizer.vertices().size()<<endl;
    globalOptimizer.save("data/result_before.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize(100);
    globalOptimizer.save("data/result_after.g2o");
    cout << "optimize done!" << endl;

    // merge cloud
    cout << "merging cloud ..."<<endl;
    PointCloud::Ptr output(new PointCloud);  //global map
    PointCloud::Ptr tmp(new PointCloud);
    // set filter
    pcl::VoxelGrid<PointT> voxel_filter;
    pcl::PassThrough<PointT> pass_filter;
    voxel_filter.setLeafSize(gridsize, gridsize, gridsize);
    pass_filter.setFilterFieldName("z");
    pass_filter.setFilterLimits(0.0, 4.0);

    for(size_t i = 0; i < keyframes.size(); i ++)
    {
        //get a frame from g2o
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex(keyframes[i].frameID));
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose = vertex->estimate(); //get optimized pose, this pose is relative to last frame
        PointCloud::Ptr newCloud = image2PointCloud(keyframes[i].rgb, keyframes[i].depth, camera); //get cloud
        // filter
        voxel_filter.setInputCloud(newCloud);
        voxel_filter.filter(*tmp);
        pass_filter.setInputCloud(tmp);
        pass_filter.filter(*newCloud);
        //transform
        pcl::transformPointCloud(*newCloud, *tmp, pose.matrix());
        *output += *tmp;

        tmp->clear();
        newCloud->clear();
    }

    // global filter
    voxel_filter.setInputCloud(output);
    voxel_filter.filter(*tmp);


    // save
    pcl::io::savePCDFile("./data/result.pcd", *tmp);
    cout << "final map saved" << endl;

    if(visualize)
    {
        pcl::visualization::CloudViewer viewer("viewer");
        viewer.showCloud(tmp);
        while(!viewer.wasStopped())
            ;
    }

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

    f.frameID = index;
    return f;
}

double normofTransform(cv::Mat rvec, cv::Mat tvec)
{
    return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec))) + fabs(cv::norm(tvec));
}

// check key frame
// input: two frame and optimizer
CHECK_RESULT checkKeyframes(FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops)
{
    static ParamReader pr;
    static int min_inliers = atoi(pr.getData("min_inliers").c_str());
    static double max_norm = atof(pr.getData("max_norm").c_str());
    static double min_norm = atof(pr.getData("keyframe_threshold").c_str());
    static double max_norm_lp = atof(pr.getData("max_norm_lp").c_str());
    static CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();

    //solve pnp
    RESULT_OF_PNP result = estimateMotion(f1, f2, camera);
    // not enough inliers
    if (result.inliers < min_inliers)
        return NOT_MATCHED;

    double norm = normofTransform(result.rvec, result.tvec);

    if(is_loops == false)
    {
        if(norm >= max_norm)
            // move too fast
            return TOO_FAR_AWAY;
    }
    else
    {
        if(norm >= max_norm_lp)
            // move too fast
            return TOO_FAR_AWAY;
    }

    // too close
    if(norm < min_norm)
        return TOO_CLOSE;

    Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec);
    // not loop then add new vetex
    if(is_loops == false)
    {
        g2o::VertexSE3* v = new g2o::VertexSE3();
        v->setId(f2.frameID);
        v->setEstimate(Eigen::Isometry3d::Identity());
        opti.addVertex(v);
    }
    // add edge
    g2o::EdgeSE3* e = new g2o::EdgeSE3();
    e->setVertex(0, opti.vertex(f1.frameID));
    e->setVertex(1, opti.vertex(f2.frameID));
    e->setRobustKernel(new g2o::RobustKernelHuber());
    // infomation matrix
    // 6*6 because pose is 3d and angle is 3d
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
    // if precision of pose and angle are all 0.1, then co-var matrix is 0.01, information matrix is 100
    information(0,0) = information(1,1) = information(2,2) = 100;
    information(3,3) = information(4,4) = information(5,5) = 100;
    e->setInformation(information);

    // set edge measurement, result of pnp
    e->setMeasurement(T.inverse()); //inverse??????????????
    // add edge to graph
    opti.addEdge(e);

    return KEYFRAME;
}

// check nearby loops
// input: frames, current frame
// 
void checkNearbyLoops(vector<FRAME>& keyframes, FRAME& currFrame, g2o::SparseOptimizer& globalOptimizer)
{
    static ParamReader pr;
    static int nearby_loops = atoi(pr.getData("nearby_loops").c_str());

    //check current frame and last N frames in keyframes
    if(keyframes.size() < nearby_loops)
    {
        //not enough keyframes,then check everyone
        for(size_t i=0; i < keyframes.size(); i ++)
        {
            checkKeyframes(keyframes[i], currFrame, globalOptimizer, true);
        }
    }
    else
    {
        // check last N frames
        for(size_t i = keyframes.size()-nearby_loops; i < keyframes.size(); i ++)
        {
            checkKeyframes(keyframes[i], currFrame, globalOptimizer, true);
        }
    }

}

// check random loops
// input: frames, current frame
void checkRandomLoops(vector<FRAME>& keyframes, FRAME& currFrame, g2o::SparseOptimizer& globalOptimizer)
{
    static ParamReader pr;
    static int random_loops = atoi(pr.getData("random_loops").c_str());
    srand((unsigned int)time(NULL));

    if(keyframes.size() <= random_loops)
    {
        // not enough frames, check everyone
        for(size_t i=0; i < keyframes.size(); i ++)
        {
            checkKeyframes(keyframes[i], currFrame, globalOptimizer, true);
        }
        
    }
    else
    {
        // randomly check frame
        for(int i = 0; i < random_loops; i++)
        {
            int index = rand()%keyframes.size();
            checkKeyframes(keyframes[index], currFrame, globalOptimizer, true);
        }
    }
}
