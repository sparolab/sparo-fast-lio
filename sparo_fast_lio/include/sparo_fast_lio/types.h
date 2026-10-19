// SPARO-FAST-LIO: ROS-independent FAST-LIO2 core library.
// Based on SPARK-FAST-LIO (MIT-SPARK) and FAST-LIO2 (HKU MaRS Lab). License: GPLv2.
#pragma once

#include <functional>
#include <string>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#if defined(_WIN32)
#define SPARO_FAST_LIO_API
#else
#define SPARO_FAST_LIO_API __attribute__((visibility("default")))
#endif

namespace sparo_fast_lio {

/// Point type used everywhere inside FAST-LIO. `curvature` holds the per-point time offset
/// w.r.t. the scan timestamp in **milliseconds**; `intensity` is the reflectivity.
using PointType  = pcl::PointXYZINormal;
using PointCloud = pcl::PointCloud<PointType>;

/// Same numeric values as the `preprocess.lidar_type` ROS parameter.
enum class LidarType : int {
  AVIA     = 1,  ///< Livox (CustomMsg). Not supported by the PointCloud2-based front-end.
  VELO16   = 2,  ///< Velodyne-style clouds (x y z intensity time ring), any number of rings.
  OUST64   = 3,  ///< Ouster-style clouds (x y z intensity t reflectivity ring ambient range).
  KMOUST64 = 4,  ///< Ouster variant used in the Kimera-Multi dataset.
};

/// Unit of the per-point time field (`preprocess.timestamp_unit`).
enum class TimeUnit : int { SEC = 0, MS = 1, US = 2, NS = 3 };

/// One IMU measurement. `t` in seconds (same clock as the LiDAR scans), `acc` in m/s^2
/// (or in g: FAST-LIO rescales by the mean norm observed during initialization), `gyr` in rad/s.
struct ImuSample {
  double t = 0.0;
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyr = Eigen::Vector3d::Zero();
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// One preprocessed LiDAR scan (output of `Preprocessor`, or any cloud in the LiDAR frame whose
/// `curvature` field holds the per-point time offset from `t` in milliseconds).
/// `t` is the scan (header) timestamp in seconds, i.e. the time of the first point.
struct LidarScan {
  double t = 0.0;
  PointCloud::Ptr points;
};

enum class LogLevel { Debug, Info, Warn, Error, Fatal };

/// Optional log sink. If unset, warnings and errors are printed to stderr.
using LogFunction = std::function<void(LogLevel, const std::string &)>;

}  // namespace sparo_fast_lio
