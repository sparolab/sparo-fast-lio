// SPARO-FAST-LIO: ROS-independent FAST-LIO2 core library.
// Based on SPARK-FAST-LIO (MIT-SPARK) and FAST-LIO2 (HKU MaRS Lab). License: GPLv2.
#pragma once

#include <memory>
#include <optional>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "sparo_fast_lio/preprocessor.h"
#include "sparo_fast_lio/types.h"

namespace sparo_fast_lio {

/// All parameters of the estimator. Defaults are identical to the defaults of the ROS 2 node
/// parameters (shown in brackets).
struct Config {
  /// Front-end parameters, only used by `Preprocessor` (kept here so one struct holds everything).
  PreprocessConfig preprocess;

  // --- mapping.* ---
  double gyr_cov        = 0.1;     // [mapping.gyr_cov]
  double acc_cov        = 0.1;     // [mapping.acc_cov]
  double b_gyr_cov      = 0.0001;  // [mapping.b_gyr_cov]
  double b_acc_cov      = 0.0001;  // [mapping.b_acc_cov]
  double fov_degree     = 360.0;   // [mapping.fov_degree] (read but unused, as in FAST-LIO2)
  double det_range      = 300.0;   // [mapping.det_range]
  bool extrinsic_est_en = false;   // [mapping.extrinsic_est_en]
  /// LiDAR pose w.r.t. the IMU: p_imu = extrinsic_R * p_lidar + extrinsic_T.
  Eigen::Vector3d extrinsic_T = Eigen::Vector3d::Zero();      // [mapping.extrinsic_T]
  Eigen::Matrix3d extrinsic_R = Eigen::Matrix3d::Identity();  // [mapping.extrinsic_R] row-major

  int max_iteration       = 4;      // [max_iteration] IEKF iterations
  double filter_size_map  = 0.5;    // [filter_size_map] voxel size (scan & map) [m]
  double cube_side_length = 200.0;  // [cube_side_length] local map size [m]
  int point_filter_num    = 4;      // [point_filter_num] keep every N-th undistorted point

  // --- gravity_alignment.* (only active when `base_T_lidar` is set) ---
  struct GravityAlignment {
    bool enable                      = true;  // [gravity_alignment.enable_gravity_alignment]
    double acc_diff_thr              = 0.2;   // [gravity_alignment.acc_diff_thr]
    int num_moving_frames_thr        = 20;    // [gravity_alignment.num_moving_frames_thr]
    int num_gravity_measurements_thr = 20;    // [gravity_alignment.num_gravity_measurements_thr]
    Eigen::Vector3d g_base{0.0, 0.0, -1.0};   // [gravity_alignment.g_base]
  } gravity_alignment;

  /// LiDAR pose w.r.t. an optional robot base frame (the ROS node looks it up on TF when
  /// `common.base_frame` is set). Enables gravity alignment and base-frame outputs.
  std::optional<Eigen::Isometry3d> base_T_lidar;

  /// What `Result::scan_world` contains (not a ROS parameter; the node computes its own clouds).
  enum class WorldScan { None, Downsampled, Dense } world_scan = WorldScan::Downsampled;

  bool verbose = false;  // [verbose]
};

/// Full IESKF state. The world frame is the IMU frame at initialization (rotated by the gravity
/// alignment, if enabled). "body" == IMU frame.
struct SPARO_FAST_LIO_API State {
  Eigen::Vector3d pos            = Eigen::Vector3d::Zero();         // world_p_imu
  Eigen::Quaterniond rot         = Eigen::Quaterniond::Identity();  // world_R_imu
  Eigen::Quaterniond offset_R_L_I = Eigen::Quaterniond::Identity(); // imu_R_lidar (estimated)
  Eigen::Vector3d offset_T_L_I   = Eigen::Vector3d::Zero();         // imu_p_lidar (estimated)
  Eigen::Vector3d vel            = Eigen::Vector3d::Zero();         // world-frame velocity
  Eigen::Vector3d bg             = Eigen::Vector3d::Zero();         // gyro bias
  Eigen::Vector3d ba             = Eigen::Vector3d::Zero();         // accelerometer bias
  Eigen::Vector3d grav           = Eigen::Vector3d::Zero();         // gravity in world frame

  Eigen::Isometry3d imuPose() const;    ///< world_T_imu
  Eigen::Isometry3d lidarPose() const;  ///< world_T_lidar = world_T_imu * imu_T_lidar
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// Error-state covariance (23x23) in IKFoM tangent-space order:
/// pos(0-2) rot(3-5) offset_R_L_I(6-8) offset_T_L_I(9-11) vel(12-14) bg(15-17) ba(18-20) grav(21-22)
using StateCovariance = Eigen::Matrix<double, 23, 23>;

/// Output of one LiDAR update (`FastLio::step`).
struct Result {
  double t = 0.0;  ///< end time of the scan [s] (time the state refers to)
  State state;
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();  ///< world_T_imu
  Eigen::Vector3d vel    = Eigen::Vector3d::Zero();
  StateCovariance cov    = StateCovariance::Zero();  ///< covariance after the update
  PointCloud::Ptr scan_undistorted;  ///< dense de-skewed scan, LiDAR frame
  PointCloud::Ptr scan_down;         ///< voxel-downsampled scan used for registration, LiDAR frame
  PointCloud::Ptr scan_world;        ///< see `Config::world_scan` (nullptr if None)
  int num_effective_points = 0;  ///< point-to-plane correspondences in the last IEKF iteration
  double mean_residual     = 0.0;  ///< mean |point-to-plane distance| of those correspondences [m]
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// IMU-rate state prediction from the latest LiDAR-updated state (returned by `addImu`).
struct ImuPropagation {
  double t = 0.0;
  State state;
  StateCovariance cov = StateCovariance::Zero();  ///< covariance of the last LiDAR update
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// FAST-LIO2 (IESKF + ikd-Tree) LiDAR-inertial odometry, ROS-free.
///
/// Usage: push data with `addImu` / `addLidar` in timestamp order (per sensor), and call `step()`
/// repeatedly; each call processes at most one LiDAR scan once enough IMU data is buffered.
///
/// Threading model (same as the original node): `addImu`/`addLidar` lock an internal buffer
/// mutex, but `step()` and the IMU-rate propagation inside `addImu` share filter state, so calls
/// must be serialized (e.g. one thread, or a single-threaded ROS executor).
class SPARO_FAST_LIO_API FastLio {
 public:
  explicit FastLio(const Config &config, LogFunction log = LogFunction());
  ~FastLio();
  FastLio(const FastLio &)            = delete;
  FastLio &operator=(const FastLio &) = delete;

  /// Buffers an IMU sample. After the first successful LiDAR update, also propagates the latest
  /// updated state with this sample and returns the IMU-rate prediction.
  std::optional<ImuPropagation> addImu(const ImuSample &imu);

  /// Buffers a preprocessed scan.
  void addLidar(const LidarScan &scan);

  /// Processes the oldest buffered scan if all IMU samples up to its end time are available.
  /// Returns a result when the state was updated and the map extended (nullopt while
  /// initializing, when no data is ready, or when a scan is skipped).
  std::optional<Result> step();

  const Config &config() const;

  /// Latest (gravity-aligned) state and its covariance.
  State state() const;
  StateCovariance covariance() const;

  bool isGravityAligned() const;
  Eigen::Matrix3d gravityAlignment() const;  ///< rotation applied to all outputs

  /// Copy of the current ikd-Tree map (world frame).
  PointCloud::Ptr map() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Transforms a LiDAR-frame cloud to the world frame with `state` (pose + estimated extrinsic).
SPARO_FAST_LIO_API PointCloud::Ptr transformToWorld(const PointCloud &cloud_lidar,
                                                    const State &state);

}  // namespace sparo_fast_lio
