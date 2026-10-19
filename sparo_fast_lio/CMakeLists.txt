cmake_minimum_required(VERSION 3.16)
project(sparo_fast_lio VERSION 0.1.0 LANGUAGES C CXX)

# ---------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------
option(SPARO_FAST_LIO_BUILD_ROS "Build the ROS 2 node (requires ament_cmake / rclcpp)" ON)
option(SPARO_FAST_LIO_BUILD_EXAMPLES "Build the ROS-free example (examples/run_core.cpp)" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Same optimization flags as upstream SPARK-FAST-LIO (keeps results numerically identical).
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -pthread")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -fexceptions")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Number of OpenMP threads used for the nearest-neighbor search (as upstream).
set(SPARO_MP_DEFINITIONS MP_PROC_NUM=1)
message(STATUS "Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  include(ProcessorCount)
  ProcessorCount(N)
  message(STATUS "Processer number:  ${N}")
  if(N GREATER 4)
    set(SPARO_MP_DEFINITIONS MP_EN MP_PROC_NUM=3)
    message(STATUS "core for MP: 3")
  elseif(N GREATER 3)
    set(SPARO_MP_DEFINITIONS MP_EN MP_PROC_NUM=2)
    message(STATUS "core for MP: 2")
  endif()
endif()

# ---------------------------------------------------------------------------------------------
# Dependencies of the ROS-independent core
# ---------------------------------------------------------------------------------------------
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

set(PCL_FIND_QUIETLY ON)
set(CLEAR_WARNING_SUPPRESSION OFF)
if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
  set(CLEAR_WARNING_SUPPRESSION ON)
  set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()
find_package(PCL REQUIRED COMPONENTS common filters io)
if(CLEAR_WARNING_SUPPRESSION)
  unset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
endif()

if(SPARO_FAST_LIO_BUILD_ROS)
  find_package(ament_cmake QUIET)
  if(NOT ament_cmake_FOUND)
    message(STATUS "ament_cmake not found: building the ROS-independent core library only")
    set(SPARO_FAST_LIO_BUILD_ROS OFF)
  endif()
endif()

# ---------------------------------------------------------------------------------------------
# sparo_fast_lio_core: ESIKF + ikd-Tree LiDAR-inertial odometry (no ROS)
# ---------------------------------------------------------------------------------------------
add_library(sparo_fast_lio_core SHARED
  src/core/fast_lio.cpp
  src/core/preprocessor.cpp
  third_party/ikd-Tree/ikd_Tree.cpp
)
add_library(sparo_fast_lio::sparo_fast_lio_core ALIAS sparo_fast_lio_core)

target_include_directories(sparo_fast_lio_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    src/core
    third_party/ikd-Tree
    third_party/IKFoM_toolkit
)
target_include_directories(sparo_fast_lio_core SYSTEM PUBLIC ${PCL_INCLUDE_DIRS})
target_link_libraries(sparo_fast_lio_core
  PUBLIC
    Eigen3::Eigen
    ${PCL_COMMON_LIBRARIES}
  PRIVATE
    ${PCL_FILTERS_LIBRARIES}
    OpenMP::OpenMP_CXX
    Threads::Threads
)
target_compile_definitions(sparo_fast_lio_core PRIVATE ${SPARO_MP_DEFINITIONS})
target_compile_features(sparo_fast_lio_core PUBLIC cxx_std_17)
# Only the public API is exported; internal FAST-LIO symbols (ikd-Tree, IKFoM, ...) stay hidden
# so that the library can coexist with other FAST-LIO copies in one process.
set_target_properties(sparo_fast_lio_core PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

# ---------------------------------------------------------------------------------------------
# Example (no ROS): drives the core from exported PCD + CSV files
# ---------------------------------------------------------------------------------------------
if(SPARO_FAST_LIO_BUILD_EXAMPLES)
  add_executable(sparo_fast_lio_run_core examples/run_core.cpp)
  target_link_libraries(sparo_fast_lio_run_core sparo_fast_lio_core ${PCL_IO_LIBRARIES})
endif()

# ---------------------------------------------------------------------------------------------
# Install / export of the core library
# ---------------------------------------------------------------------------------------------
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS sparo_fast_lio_core
  EXPORT sparo_fast_lioTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
if(SPARO_FAST_LIO_BUILD_EXAMPLES)
  install(TARGETS sparo_fast_lio_run_core RUNTIME DESTINATION lib/${PROJECT_NAME})
endif()

if(NOT SPARO_FAST_LIO_BUILD_ROS)
  # Plain CMake package: find_package(sparo_fast_lio) -> sparo_fast_lio::sparo_fast_lio_core
  include(CMakePackageConfigHelpers)
  install(EXPORT sparo_fast_lioTargets
    NAMESPACE sparo_fast_lio::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
  configure_package_config_file(cmake/sparo_fast_lioConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sparo_fast_lioConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/sparo_fast_lioConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
  )
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/sparo_fast_lioConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/sparo_fast_lioConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
  return()
endif()

# ---------------------------------------------------------------------------------------------
# ROS 2 node / component (thin wrapper around the core)
# ---------------------------------------------------------------------------------------------
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_conversions REQUIRED)

add_library(sparo_fast_lio_component SHARED src/ros/sparo_fast_lio_node.cpp)
target_link_libraries(sparo_fast_lio_component sparo_fast_lio_core ${PCL_IO_LIBRARIES})
target_compile_definitions(sparo_fast_lio_component PRIVATE
  ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")
ament_target_dependencies(sparo_fast_lio_component
  rclcpp
  rclcpp_components
  builtin_interfaces
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_ros
  pcl_conversions
)
rclcpp_components_register_node(sparo_fast_lio_component
  PLUGIN "sparo_fast_lio::SparoFastLioNode"
  EXECUTABLE sparo_lio_mapping
)

install(TARGETS sparo_fast_lio_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch config rviz DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_targets(sparo_fast_lioTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 PCL OpenMP Threads)
ament_package()
