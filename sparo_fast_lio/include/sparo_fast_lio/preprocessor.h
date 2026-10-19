// SPARO-FAST-LIO: ROS-independent FAST-LIO2 core library.
// Based on SPARK-FAST-LIO (MIT-SPARK) and FAST-LIO2 (HKU MaRS Lab). License: GPLv2.
#pragma once

#include <cstdint>
#include <memory>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>

#include "sparo_fast_lio/types.h"

namespace sparo_fast_lio {

/// Velodyne-style input point (also matches HeLiPR / many Ouster drivers re-published as
/// velodyne layout): x y z intensity time ring. `time` is the per-point offset from the scan
/// stamp in the unit given by `PreprocessConfig::timestamp_unit`.
struct EIGEN_ALIGN16 VelodynePoint {
  PCL_ADD_POINT4D;
  float intensity;
  float time;
  std::uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// Ouster-driver-style input point: x y z intensity t reflectivity ring ambient range.
struct EIGEN_ALIGN16 OusterPoint {
  PCL_ADD_POINT4D;
  float intensity;
  std::uint32_t t;
  std::uint16_t reflectivity;
  std::uint8_t ring;
  std::uint16_t ambient;
  std::uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// Front-end parameters (ROS parameters `preprocess.*` and `point_filter_num_for_preprocessing`).
struct PreprocessConfig {
  LidarType lidar_type          = LidarType::AVIA;  // ROS default (unsupported for PointCloud2)
  int scan_line                 = 16;
  int scan_rate                 = 10;  // [Hz], only used when the cloud has no per-point time
  TimeUnit timestamp_unit       = TimeUnit::US;
  double blind                  = 0.01;  // [m]
  double blind_for_human_pilots = 1.5;   // [m], Kimera-Multi specific pilot-zone rejection
  int point_filter_num          = 1;     // keep every N-th raw point
  bool feature_enabled          = false;
};

/// Converts raw sensor clouds into the FAST-LIO input cloud (`PointCloud` with per-point time
/// offset in milliseconds stored in `curvature`). PointCloud2-free: feed typed PCL clouds.
class SPARO_FAST_LIO_API Preprocessor {
 public:
  explicit Preprocessor(const PreprocessConfig &config = PreprocessConfig(),
                        LogFunction log                = LogFunction());
  ~Preprocessor();
  Preprocessor(const Preprocessor &)            = delete;
  Preprocessor &operator=(const Preprocessor &) = delete;

  /// For `LidarType::VELO16`.
  PointCloud::Ptr process(const pcl::PointCloud<VelodynePoint> &cloud);
  /// For `LidarType::OUST64` and `LidarType::KMOUST64`.
  PointCloud::Ptr process(const pcl::PointCloud<OusterPoint> &cloud);

  const PreprocessConfig &config() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sparo_fast_lio

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(sparo_fast_lio::VelodynePoint,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (float, time, time)
                                  (std::uint16_t, ring, ring))

POINT_CLOUD_REGISTER_POINT_STRUCT(sparo_fast_lio::OusterPoint,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (std::uint32_t, t, t)
                                  (std::uint16_t, reflectivity, reflectivity)
                                  (std::uint8_t, ring, ring)
                                  (std::uint16_t, ambient, ambient)
                                  (std::uint32_t, range, range))
// clang-format on
