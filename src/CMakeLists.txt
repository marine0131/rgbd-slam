FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

FIND_PACKAGE(PCL REQUIRED COMPONENTS common io filters visualization)
ADD_DEFINITIONS(${PCL_DEFINITIONS})
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
LINK_LIBRARIES(${PCL_LIBRARY_DIRS})

# eigen3
FIND_PACKAGE(Eigen3)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

# g2o
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
SET(G2O_ROOT /usr/local/include/g2o)
FIND_PACKAGE(G2O)
INCLUDE_DIRECTORIES(${G2O_INCLUDE_DIR})

#csparse
FIND_PACKAGE(CSparse)
INCLUDE_DIRECTORIES(${CSPARSE_INCLUDE_DIR})

ADD_EXECUTABLE(generate_pointcloud generatePointCloud.cpp)
TARGET_LINK_LIBRARIES(generate_pointcloud 
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES})

ADD_LIBRARY(slam_base slamBase.cpp)
TARGET_LINK_LIBRARIES(slam_base 
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES})

ADD_EXECUTABLE(detectFeatures detectFeatures.cpp)
TARGET_LINK_LIBRARIES(detectFeatures
    slam_base
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES})

ADD_EXECUTABLE(joinPointCloud joinPointCloud.cpp)
TARGET_LINK_LIBRARIES(joinPointCloud
    slam_base
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES})

ADD_EXECUTABLE(visualOdometry visualOdometry.cpp)
TARGET_LINK_LIBRARIES(visualOdometry
    slam_base
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES})

ADD_EXECUTABLE(slamEnd slamEnd.cpp)
TARGET_LINK_LIBRARIES(slamEnd
    slam_base
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARCE_LIBRARY}
    )

ADD_EXECUTABLE(slam slam.cpp)
TARGET_LINK_LIBRARIES(slam
    slam_base
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARCE_LIBRARY}
    )
