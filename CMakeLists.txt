cmake_minimum_required(VERSION 2.8)

project(fvpackage)

find_package(Eigen3 REQUIRED)
find_package(OpenMP)

link_directories(${Eigen_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)

include_directories(header)

target_link_libraries(${PROJECT_NAME} ${Eigen_LIBRARIES})
if(OpenMP_CXX_FOUND)
    message("-- OpenMP found")
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()