#include <iostream>
#include <fstream>
#include <sstream>

#include "slamBase.h"
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

// read a frame from file
FRAME readFrame(int index, ParamReader& pr);
// measure norm of transform
double normofTransform(cv::Mat rvec, cv::Mat tvec);

// g2o def
// typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> SlamBlockSolver;
typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

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

    // initialize g2o
    unique_ptr<SlamLinearSolver> linearSolver (new SlamLinearSolver());
    linearSolver->setBlockOrdering(false);
    unique_ptr<SlamBlockSolver> blockSolver (new SlamBlockSolver(move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(move(blockSolver));

    g2o::SparseOptimizer globalOptimizer;
    globalOptimizer.setAlgorithm(solver);
    // no debug message
    globalOptimizer.setVerbose(false);

    // add first Vertex(point)
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId(startIndex);
    v->setEstimate(Eigen::Isometry3d::Identity()); // estimate a identity
    v->setFixed(true); // first vertex need no optimization
    globalOptimizer.addVertex(v);

    int lastIndex = startIndex;
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

        // add vertex
        g2o::VertexSE3* v = new g2o::VertexSE3();
        v->setId(index);
        v->setEstimate(Eigen::Isometry3d::Identity());
        globalOptimizer.addVertex(v);
        // add edge
        g2o::EdgeSE3* edge = new g2o::EdgeSE3();
        edge->vertices()[0] = globalOptimizer.vertex(lastIndex);
        edge->vertices()[1] = globalOptimizer.vertex(index);
        // infomation matrix
        // 6*6 because pose is 3d and angle is 3d
        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
        // if precision of pose and angle are all 0.1, then co-var matrix is 0.01, information matrix is 100
        information(0,0) = information(1,1) = information(2,2) = 100;
        information(3,3) = information(4,4) = information(5,5) = 100;
        edge->setInformation(information);

        // set edge measurement, result of pnp
        edge->setMeasurement(T);
        // add edge to graph
        globalOptimizer.addEdge(edge);

        // next frame
        lastIndex = index;
        lastFrame = currFrame;
    }

    // optimize all vertex
    cout << "optimize pose graph, vertices: " <<globalOptimizer.vertices().size()<<endl;
    globalOptimizer.save("data/result_defore.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize(100);
    globalOptimizer.save("data/result_after.g2o");
    globalOptimizer.clear();
    cout << "optimize done!" << endl;
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
