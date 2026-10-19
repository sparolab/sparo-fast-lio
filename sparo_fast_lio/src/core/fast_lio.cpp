// SPARO-FAST-LIO core: ROS-independent port of the SPARK-FAST-LIO mapping loop
// (originally `SPARKFastLIO2` in spark_fast_lio.cpp). The estimator code below is kept
// line-by-line equivalent to the original so that results are numerically identical.
#include "sparo_fast_lio/fast_lio.h"

#include <cmath>
#include <cstdio>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <omp.h>
#include <pcl/filters/voxel_grid.h>

#include "common/so3_math.h"
#include "ikd_Tree.h"
#include "imu_processing.hpp"

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MOV_THRESHOLD (1.5)

namespace sparo_fast_lio {

namespace {

State toState(const state_ikfom &s) {
  State out;
  out.pos          = s.pos;
  out.rot          = Eigen::Quaterniond(s.rot.coeffs());
  out.offset_R_L_I = Eigen::Quaterniond(s.offset_R_L_I.coeffs());
  out.offset_T_L_I = s.offset_T_L_I;
  out.vel          = s.vel;
  out.bg           = s.bg;
  out.ba           = s.ba;
  out.grav         = V3D(s.grav[0], s.grav[1], s.grav[2]);
  return out;
}

void defaultLog(LogLevel level, const std::string &msg) {
  if (level == LogLevel::Warn || level == LogLevel::Error || level == LogLevel::Fatal) {
    std::cerr << "[sparo_fast_lio] " << msg << std::endl;
  }
}

}  // namespace

Eigen::Isometry3d State::imuPose() const {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear()          = rot.toRotationMatrix();
  T.translation()     = pos;
  return T;
}

Eigen::Isometry3d State::lidarPose() const {
  Eigen::Isometry3d T_I_L = Eigen::Isometry3d::Identity();
  T_I_L.linear()          = offset_R_L_I.toRotationMatrix();
  T_I_L.translation()     = offset_T_L_I;
  return imuPose() * T_I_L;
}

PointCloud::Ptr transformToWorld(const PointCloud &cloud_lidar, const State &s) {
  const int size = cloud_lidar.points.size();
  PointCloud::Ptr out(new PointCloud(size, 1));
  for (int i = 0; i < size; i++) {
    const auto &pi = cloud_lidar.points[i];
    auto &po       = out->points[i];
    V3D p_body(pi.x, pi.y, pi.z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
    po.x         = p_global(0);
    po.y         = p_global(1);
    po.z         = p_global(2);
    po.intensity = pi.intensity;
  }
  return out;
}

struct FastLio::Impl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Impl(const Config &cfg, LogFunction log_fn);

  void log(LogLevel level, const std::string &msg) const { log_(level, msg); }

  M3D computeRelativeRotation(const Eigen::Vector3d &g_a, const Eigen::Vector3d &g_b);
  void pointBodyToWorld(PointType const *const pi, PointType *const po, const state_ikfom &s);
  template <typename T>
  void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi,
                        Eigen::Matrix<T, 3, 1> &po,
                        const state_ikfom &s) const {
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
  }
  void collectRemovedPoints();
  std::optional<ImuPropagation> addImu(const ImuSample &imu);
  void addLidar(const LidarScan &scan);
  std::optional<ImuPropagation> integrateIMU(esekfom::esekf<state_ikfom, 12, input_ikfom> &state,
                                             const ImuSample &msg);
  void calcHModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);
  void lasermapFovSegment();
  void mapIncremental();
  bool syncPackages(MeasureGroup &meas, bool verbose);
  bool isMotionStopped(const V3D &acc_ref, const V3D &acc_curr, const double acc_diff_thr);
  std::optional<Result> processLidarAndImu(MeasureGroup &Measure);

  Config config_;
  LogFunction log_;

  std::mutex buffer_mutex_;

  /*** Time Log Variables ***/
  double kdtree_incremental_time_ = 0.0;
  double kdtree_search_time_      = 0.0;
  double kdtree_delete_time_      = 0.0;
  double match_time_              = 0.0;
  double solve_time_              = 0.0;

  int kdtree_size_st_        = 0;
  int add_point_size_        = 0;
  int kdtree_delete_counter_ = 0;

  bool extrinsic_est_en_ = false;
  bool verbose_          = false;

  bool enable_gravity_alignment_ = false;
  bool is_gravity_aligned_       = false;
  bool has_base_frame_           = false;

  float res_last_[100000] = {0.0};
  float det_range_        = 300.0f;

  double last_lidar_timestamp_ = 0.0;
  double last_imu_timestamp_   = 0.0;

  double res_mean_last_           = 0.05;
  double total_residual_          = 0.0;
  double filter_size_map_min_     = 0.0;
  double cube_len_                = 0.0;
  double lidar_end_time_          = 0.0;
  double first_lidar_time_        = 0.0;

  int effect_feat_num_     = 0;
  int scan_count_          = 0;
  int feats_down_size_     = 0;
  int NUM_MAX_ITERATIONS_  = 0;
  int point_filter_num_    = 4;

  double lidar_mean_scantime_ = 0.0;
  int scan_num_               = 0;

  double acc_diff_thr_              = 0.2;
  int num_moving_frames_thr_        = 10;
  int num_gravity_measurements_thr_ = 10;
  bool point_selected_surf_[100000] = {0};
  bool lidar_pushed_                = false;
  bool flg_first_scan_              = true;
  bool flg_EKF_inited_              = false;

  // Former function-local statics of the original node (now per instance).
  bool localmap_initialized_          = false;
  int num_consecutive_moving_frames_  = 0;
  std::optional<M3D> offset_R_I_B_;  // gravity alignment: imu -> base rotation (fixed once)
  std::deque<ImuSample> imu_prop_queue_;
  size_t num_lidar_prev_ = 0;
  size_t num_imu_prev_   = 0;
  double last_imu_timestamp_prev_ = -1.0;

  std::deque<V3D> global_gravity_directions_;

  BoxPointType localmap_points_;
  std::vector<BoxPointType> cub_needrm_;

  std::vector<PointVector> nearest_points_;

  std::deque<double> time_buffer_;
  std::deque<PointCloudXYZI::Ptr> lidar_buffer_;
  std::deque<std::shared_ptr<const ImuSample>> imu_buffer_;

  PointCloudXYZI::Ptr cloud_undistort_;
  PointCloudXYZI::Ptr feats_undistort_;
  PointCloudXYZI::Ptr feats_down_body_;
  PointCloudXYZI::Ptr feats_down_world_;
  PointCloudXYZI::Ptr normvec_;
  PointCloudXYZI::Ptr laser_cloud_ori_;
  PointCloudXYZI::Ptr corr_normvec_;

  pcl::VoxelGrid<PointType> down_size_filter_;
  KD_TREE<PointType> ikd_tree_;

  V3F xaxis_point_body_;
  V3F xaxis_point_world_;
  V3D g_base_;
  V3D mean_acc_stopped_;
  M3D R_gravity_aligned_;

  V3D lidar_T_wrt_base_;
  M3D lidar_R_wrt_base_;

  /*** EKF inputs and output ***/
  MeasureGroup Measures_;
  esekfom::esekf<state_ikfom, 12, input_ikfom> kf_;
  std::optional<esekfom::esekf<state_ikfom, 12, input_ikfom>> kf_for_preintegration_;
  state_ikfom latest_state_;

  std::shared_ptr<ImuProcess> imu_processor_;
};

FastLio::Impl::Impl(const Config &cfg, LogFunction log_fn) : config_(cfg) {
  log_ = log_fn ? std::move(log_fn) : LogFunction(defaultLog);

  xaxis_point_body_ << LIDAR_SP_LEN, 0.0, 0.0;
  xaxis_point_world_ << LIDAR_SP_LEN, 0.0, 0.0;
  g_base_            = Zero3d;
  mean_acc_stopped_  = Zero3d;
  R_gravity_aligned_ = Eye3d;

  lidar_T_wrt_base_ = Zero3d;
  lidar_R_wrt_base_ = Eye3d;

  NUM_MAX_ITERATIONS_ = config_.max_iteration;

  filter_size_map_min_ = config_.filter_size_map;
  cube_len_            = config_.cube_side_length;
  det_range_           = config_.det_range;

  enable_gravity_alignment_     = config_.gravity_alignment.enable;
  acc_diff_thr_                 = config_.gravity_alignment.acc_diff_thr;
  num_moving_frames_thr_        = config_.gravity_alignment.num_moving_frames_thr;
  num_gravity_measurements_thr_ = config_.gravity_alignment.num_gravity_measurements_thr;
  g_base_                       = config_.gravity_alignment.g_base;

  verbose_          = config_.verbose;
  extrinsic_est_en_ = config_.extrinsic_est_en;
  point_filter_num_ = config_.point_filter_num;

  imu_processor_      = std::make_shared<ImuProcess>();
  imu_processor_->log = log_;
  imu_processor_->set_extrinsic(config_.extrinsic_T, config_.extrinsic_R);
  imu_processor_->set_gyr_cov(V3D(config_.gyr_cov, config_.gyr_cov, config_.gyr_cov));
  imu_processor_->set_acc_cov(V3D(config_.acc_cov, config_.acc_cov, config_.acc_cov));
  imu_processor_->set_gyr_bias_cov(V3D(config_.b_gyr_cov, config_.b_gyr_cov, config_.b_gyr_cov));
  imu_processor_->set_acc_bias_cov(V3D(config_.b_acc_cov, config_.b_acc_cov, config_.b_acc_cov));

  down_size_filter_.setLeafSize(filter_size_map_min_, filter_size_map_min_, filter_size_map_min_);

  double epsi[23];
  for (int i = 0; i < 23; ++i) epsi[i] = 0.001;

  kf_.init_dyn_share(
      get_f,
      df_dx,
      df_dw,
      [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
        calcHModel(s, ekfom_data);
      },
      NUM_MAX_ITERATIONS_,
      epsi);

  cloud_undistort_.reset(new PointCloudXYZI());
  feats_undistort_.reset(new PointCloudXYZI());
  feats_down_body_.reset(new PointCloudXYZI());
  feats_down_world_.reset(new PointCloudXYZI());
  normvec_.reset(new PointCloudXYZI(100000, 1));
  laser_cloud_ori_.reset(new PointCloudXYZI(100000, 1));
  corr_normvec_.reset(new PointCloudXYZI(100000, 1));

  if (config_.base_T_lidar.has_value()) {
    has_base_frame_   = true;
    lidar_T_wrt_base_ = config_.base_T_lidar->translation();
    lidar_R_wrt_base_ = config_.base_T_lidar->linear();
  }
}

// Outputs rotation matrix that aligns a to b, i.e., R such that R * g_a = g_b
M3D FastLio::Impl::computeRelativeRotation(const Eigen::Vector3d &g_a,
                                           const Eigen::Vector3d &g_b) {
  Eigen::Vector3d g_a_norm = g_a.normalized();
  Eigen::Vector3d g_b_norm = g_b.normalized();

  Eigen::Vector3d axis = g_a_norm.cross(g_b_norm);
  double cos_theta     = g_a_norm.dot(g_b_norm);

  if (std::fabs(1.0 - cos_theta) < 1e-3) {
    return Eigen::Matrix3d::Identity();
  }

  // Degenerate condition a = -b
  if (std::fabs(1.0 + cos_theta) < 1e-3) {
    Eigen::Vector3d perturbed = g_a_norm + Eigen::Vector3d(1, 2, 3);
    axis                      = g_a_norm.cross(perturbed);

    if (axis.norm() < 1e-6) {
      perturbed = g_a_norm + Eigen::Vector3d(3, 2, 1);
      axis      = g_a_norm.cross(perturbed);
    }

    axis.normalize();
    return Eigen::AngleAxisd(M_PI, axis).toRotationMatrix();
  } else {
    axis.normalize();
    double theta = std::acos(cos_theta);

    Eigen::Quaterniond q(Eigen::AngleAxisd(theta, axis));

    return q.toRotationMatrix();
  }
}

void FastLio::Impl::pointBodyToWorld(PointType const *const pi,
                                     PointType *const po,
                                     const state_ikfom &s) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

  po->x         = p_global(0);
  po->y         = p_global(1);
  po->z         = p_global(2);
  po->intensity = pi->intensity;
}

void FastLio::Impl::collectRemovedPoints() {
  PointVector points_history;
  ikd_tree_.acquire_removed_points(points_history);
}

void FastLio::Impl::addLidar(const LidarScan &scan) {
  std::lock_guard<std::mutex> lk(buffer_mutex_);
  scan_count_++;
  const double msg_time = scan.t;

  if (msg_time < last_lidar_timestamp_) {
    log(LogLevel::Error, "Lidar loopback detected, clearing buffers");
    lidar_buffer_.clear();
  }
  last_lidar_timestamp_ = msg_time;

  // The scan is only read (never modified) by the estimator, so no copy is needed.
  PointCloudXYZI::Ptr ptr = scan.points ? scan.points : PointCloudXYZI::Ptr(new PointCloudXYZI());

  lidar_buffer_.push_back(ptr);
  time_buffer_.push_back(msg_time);
}

std::optional<ImuPropagation> FastLio::Impl::addImu(const ImuSample &msg) {
  const double stamp = msg.t;
  std::lock_guard<std::mutex> lk(buffer_mutex_);

  auto imu_input = std::make_shared<ImuSample>(msg);

  if (stamp < last_imu_timestamp_) {
    std::stringstream ss;
    ss << std::fixed << "IMU loopback, clearing buffers (previous: " << last_imu_timestamp_
       << " vs. received: " << stamp << " [s]";
    log(LogLevel::Warn, ss.str());
    imu_buffer_.clear();
    kf_for_preintegration_.reset();
  }
  last_imu_timestamp_ = stamp;

  std::optional<ImuPropagation> out;
  if (kf_for_preintegration_.has_value()) {
    out = integrateIMU(*kf_for_preintegration_, *imu_input);
  }

  imu_buffer_.push_back(imu_input);
  return out;
}

std::optional<ImuPropagation> FastLio::Impl::integrateIMU(
    esekfom::esekf<state_ikfom, 12, input_ikfom> &state, const ImuSample &msg) {
  imu_prop_queue_.push_back(msg);

  if (imu_prop_queue_.size() < 2) {
    return std::nullopt;
  }

  // Assume that timestamps are sufficiently close and ascending order
  double dt = imu_prop_queue_[1].t - imu_prop_queue_[0].t;

  if (dt <= 0) {
    log(LogLevel::Error, "IMU timestamps must be in ascending order!");
    imu_prop_queue_.pop_front();
    return std::nullopt;
  }

  auto integrated_state = imu_processor_->IntegrateIMU(imu_prop_queue_, state);
  const double stamp    = imu_prop_queue_[1].t;
  imu_prop_queue_.pop_front();

  integrated_state.pos = R_gravity_aligned_ * integrated_state.pos;
  integrated_state.rot = R_gravity_aligned_ * integrated_state.rot;

  ImuPropagation out;
  out.t     = stamp;
  out.state = toState(integrated_state);
  out.cov   = kf_.get_P();
  return out;
}

void FastLio::Impl::calcHModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
  double match_start = omp_get_wtime();
  laser_cloud_ori_->clear();
  corr_normvec_->clear();
  total_residual_ = 0.0;

  /** closest surface search and residual computation **/
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int i = 0; i < feats_down_size_; i++) {
    PointType &point_body  = feats_down_body_->points[i];
    PointType &point_world = feats_down_world_->points[i];

    /* transform to world frame */
    V3D p_body(point_body.x, point_body.y, point_body.z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
    point_world.x         = p_global(0);
    point_world.y         = p_global(1);
    point_world.z         = p_global(2);
    point_world.intensity = point_body.intensity;

    std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

    auto &points_near = nearest_points_[i];

    if (ekfom_data.converge) {
      /** Find the closest surfaces in the map **/
      ikd_tree_.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
      point_selected_surf_[i] = points_near.size() < NUM_MATCH_POINTS        ? false
                                : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                             : true;
    }

    if (!point_selected_surf_[i]) continue;

    VF(4) pabcd;
    point_selected_surf_[i] = false;
    if (esti_plane(pabcd, points_near, 0.1f)) {
      float pd2 =
          pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
      float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

      if (s > 0.9) {
        point_selected_surf_[i]       = true;
        normvec_->points[i].x         = pabcd(0);
        normvec_->points[i].y         = pabcd(1);
        normvec_->points[i].z         = pabcd(2);
        normvec_->points[i].intensity = pd2;
        res_last_[i]                  = abs(pd2);
      }
    }
  }

  effect_feat_num_ = 0;

  for (int i = 0; i < feats_down_size_; i++) {
    if (point_selected_surf_[i]) {
      laser_cloud_ori_->points[effect_feat_num_] = feats_down_body_->points[i];
      corr_normvec_->points[effect_feat_num_]    = normvec_->points[i];
      total_residual_ += res_last_[i];
      effect_feat_num_++;
    }
  }

  if (effect_feat_num_ < 1) {
    ekfom_data.valid = false;
    log(LogLevel::Warn, "No Effective Points!");
    return;
  }

  res_mean_last_ = total_residual_ / effect_feat_num_;
  match_time_ += omp_get_wtime() - match_start;
  double solve_start = omp_get_wtime();

  /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
  ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_feat_num_, 12);  // 23
  ekfom_data.h.resize(effect_feat_num_);

  for (int i = 0; i < effect_feat_num_; i++) {
    const PointType &laser_p = laser_cloud_ori_->points[i];
    V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
    M3D point_be_crossmat;
    point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
    V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);

    /*** get the normal vector of closest surface/corner ***/
    const PointType &norm_p = corr_normvec_->points[i];
    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

    /*** calculate the Measuremnt Jacobian matrix H ***/
    V3D C(s.rot.conjugate() * norm_vec);
    V3D A(point_crossmat * C);
    if (extrinsic_est_en_) {
      V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);  // s.rot.conjugate()*norm_vec);
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A),
          VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
    } else {
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0;
    }

    /*** Measuremnt: distance to the closest surface/corner ***/
    ekfom_data.h(i) = -norm_p.intensity;
  }
  solve_time_ += omp_get_wtime() - solve_start;
}

void FastLio::Impl::lasermapFovSegment() {
  cub_needrm_.clear();
  kdtree_delete_counter_ = 0;
  kdtree_delete_time_    = 0.0;
  pointBodyToWorld(xaxis_point_body_, xaxis_point_world_, latest_state_);

  // `lidar_xyz`: LiDAR position w.r.t. world frame
  V3D lidar_xyz = kf_.get_lidar_position();
  if (!localmap_initialized_) {
    for (int i = 0; i < 3; i++) {
      localmap_points_.vertex_min[i] = lidar_xyz(i) - cube_len_ / 2.0;
      localmap_points_.vertex_max[i] = lidar_xyz(i) + cube_len_ / 2.0;
    }
    localmap_initialized_ = true;
    return;
  }

  float dist_to_map_edge[3][2];
  bool need_move = false;
  for (int i = 0; i < 3; i++) {
    dist_to_map_edge[i][0] = fabs(lidar_xyz(i) - localmap_points_.vertex_min[i]);
    dist_to_map_edge[i][1] = fabs(lidar_xyz(i) - localmap_points_.vertex_max[i]);
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * det_range_ ||
        dist_to_map_edge[i][1] <= MOV_THRESHOLD * det_range_)
      need_move = true;
  }
  if (!need_move) return;
  BoxPointType new_localmap_points, tmp_boxpoints;
  new_localmap_points = localmap_points_;
  float mov_dist      = std::max((cube_len_ - 2.0 * MOV_THRESHOLD * det_range_) * 0.5 * 0.9,
                            static_cast<double>(det_range_ * (MOV_THRESHOLD - 1)));
  for (int i = 0; i < 3; i++) {
    tmp_boxpoints = localmap_points_;
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * det_range_) {
      new_localmap_points.vertex_max[i] -= mov_dist;
      new_localmap_points.vertex_min[i] -= mov_dist;
      tmp_boxpoints.vertex_min[i] = localmap_points_.vertex_max[i] - mov_dist;
      cub_needrm_.push_back(tmp_boxpoints);
    } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * det_range_) {
      new_localmap_points.vertex_max[i] += mov_dist;
      new_localmap_points.vertex_min[i] += mov_dist;
      tmp_boxpoints.vertex_max[i] = localmap_points_.vertex_min[i] + mov_dist;
      cub_needrm_.push_back(tmp_boxpoints);
    }
  }
  localmap_points_ = new_localmap_points;

  collectRemovedPoints();
  double delete_begin = omp_get_wtime();
  if (cub_needrm_.size() > 0) {
    kdtree_delete_counter_ = ikd_tree_.Delete_Point_Boxes(cub_needrm_);
  }
  kdtree_delete_time_ = omp_get_wtime() - delete_begin;
}

void FastLio::Impl::mapIncremental() {
  PointVector PointToAdd;
  PointVector PointNoNeedDownsample;
  PointToAdd.reserve(feats_down_size_);
  PointNoNeedDownsample.reserve(feats_down_size_);

  for (int i = 0; i < feats_down_size_; i++) {
    // transform to world frame
    pointBodyToWorld(
        &(feats_down_body_->points[i]), &(feats_down_world_->points[i]), latest_state_);

    // decide if we need to add to map
    if (!nearest_points_[i].empty() && flg_EKF_inited_) {
      const PointVector &points_near = nearest_points_[i];
      bool need_add                  = true;

      PointType mid_point;
      mid_point.x =
          std::floor(feats_down_world_->points[i].x / filter_size_map_min_) * filter_size_map_min_ +
          0.5 * filter_size_map_min_;
      mid_point.y =
          std::floor(feats_down_world_->points[i].y / filter_size_map_min_) * filter_size_map_min_ +
          0.5 * filter_size_map_min_;
      mid_point.z =
          std::floor(feats_down_world_->points[i].z / filter_size_map_min_) * filter_size_map_min_ +
          0.5 * filter_size_map_min_;

      float dist = calc_dist(feats_down_world_->points[i], mid_point);
      if (std::fabs(points_near[0].x - mid_point.x) > 0.5f * filter_size_map_min_ &&
          std::fabs(points_near[0].y - mid_point.y) > 0.5f * filter_size_map_min_ &&
          std::fabs(points_near[0].z - mid_point.z) > 0.5f * filter_size_map_min_) {
        PointNoNeedDownsample.push_back(feats_down_world_->points[i]);
        continue;
      }

      for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
        if (points_near.size() < NUM_MATCH_POINTS) break;
        if (calc_dist(points_near[readd_i], mid_point) < dist) {
          need_add = false;
          break;
        }
      }
      if (need_add) {
        PointToAdd.push_back(feats_down_world_->points[i]);
      }
    } else {
      // no nearest points, or not in EKF inited
      PointToAdd.push_back(feats_down_world_->points[i]);
    }
  }

  double st_time  = omp_get_wtime();
  add_point_size_ = ikd_tree_.Add_Points(PointToAdd, true);
  ikd_tree_.Add_Points(PointNoNeedDownsample, false);

  add_point_size_          = PointToAdd.size() + PointNoNeedDownsample.size();
  kdtree_incremental_time_ = omp_get_wtime() - st_time;
}

bool FastLio::Impl::syncPackages(MeasureGroup &meas, bool verbose) {
  std::lock_guard<std::mutex> lk(buffer_mutex_);
  if (verbose) {
    size_t num_lidar_curr = lidar_buffer_.size();
    size_t num_imu_curr   = imu_buffer_.size();

    // To only print out when changes occur
    if ((num_lidar_prev_ != num_lidar_curr) || (num_imu_prev_ != num_imu_curr)) {
      log(LogLevel::Info, std::to_string(num_lidar_curr) + " vs. " + std::to_string(num_imu_curr));
      num_lidar_prev_ = num_lidar_curr;
      num_imu_prev_   = num_imu_curr;
    }
  }

  if (lidar_buffer_.empty() || imu_buffer_.empty()) return false;

  if (!lidar_pushed_) {
    meas.lidar                = lidar_buffer_.front();
    meas.lidar_beg_time       = time_buffer_.front();
    const double denominator  = 1000;

    if (meas.lidar->points.size() <= 1) {
      lidar_end_time_ = meas.lidar_beg_time + lidar_mean_scantime_;
    } else if (meas.lidar->points.back().curvature / denominator < 0.5 * lidar_mean_scantime_) {
      lidar_end_time_ = meas.lidar_beg_time + lidar_mean_scantime_;
    } else {
      scan_num_++;
      if (meas.lidar->points.back().curvature < 80 || meas.lidar->points.back().curvature > 120) {
        char buf[512];
        std::snprintf(
            buf,
            sizeof(buf),
            "meas.lidar->points.back().curvature (%.2f) should be close to 100. Please "
            "check the `timestamp_unit` "
            "or values of `time` (or `t`) field of the point cloud input from your sensor.",
            meas.lidar->points.back().curvature);
        log(LogLevel::Warn, buf);
      }

      double dt       = meas.lidar->points.back().curvature / 1000.0;
      lidar_end_time_ = meas.lidar_beg_time + dt;
      lidar_mean_scantime_ += (dt - lidar_mean_scantime_) / static_cast<double>(scan_num_);
    }
    meas.lidar_end_time = lidar_end_time_;
    lidar_pushed_       = true;
  }

  if (last_imu_timestamp_ < lidar_end_time_) {
    if (verbose) {
      // To only print out when changes occur
      if (last_imu_timestamp_prev_ != last_imu_timestamp_) {
        char buf[128];
        std::snprintf(buf,
                      sizeof(buf),
                      "Not enough IMU data (%.6f < %.6f)",
                      last_imu_timestamp_,
                      lidar_end_time_);
        log(LogLevel::Info, buf);
        last_imu_timestamp_prev_ = last_imu_timestamp_;
      }
    }
    return false;
  }

  /*** push imu data, and pop from imu buffer ***/
  double imu_time = imu_buffer_.front()->t;
  meas.imu.clear();
  while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_)) {
    imu_time = imu_buffer_.front()->t;
    if (imu_time > lidar_end_time_) break;
    meas.imu.push_back(imu_buffer_.front());
    imu_buffer_.pop_front();
  }

  lidar_buffer_.pop_front();
  time_buffer_.pop_front();
  lidar_pushed_ = false;

  return true;
}

bool FastLio::Impl::isMotionStopped(const V3D &acc_ref,
                                    const V3D &acc_curr,
                                    const double acc_diff_thr) {
  return (acc_ref - acc_curr).norm() <= acc_diff_thr;
}

std::optional<Result> FastLio::Impl::processLidarAndImu(MeasureGroup &Measures) {
  if (flg_first_scan_) {
    first_lidar_time_                = Measures.lidar_beg_time;
    imu_processor_->first_lidar_time = first_lidar_time_;
    flg_first_scan_                  = false;
    return std::nullopt;
  }

  // NOTE(hlim): Place resampling outside `Process` function to get full cloud point,
  // i.e., `cloud_undistort_`: raw undistorted cloud points
  // & `feats_undistort_`: Undistorted cloud points for pose estimation module
  // (fresh clouds every scan so that clouds handed out in `Result` are never overwritten)
  cloud_undistort_.reset(new PointCloudXYZI());
  feats_undistort_->clear();

  imu_processor_->Process(Measures, kf_, cloud_undistort_);
  feats_undistort_->reserve(cloud_undistort_->size() / point_filter_num_);

  for (size_t i = 0; i < cloud_undistort_->points.size(); i++) {
    if (i % point_filter_num_ == 0) {
      feats_undistort_->push_back(cloud_undistort_->points[i]);
    }
  }

  latest_state_ = kf_.get_x();

  if (feats_undistort_->empty() || (feats_undistort_ == NULL)) {
    log(LogLevel::Warn, "No point, skip this scan!\n");
    return std::nullopt;
  }

  if (enable_gravity_alignment_ && !is_gravity_aligned_ && has_base_frame_) {
    if (!flg_EKF_inited_) {
      // Assume that it is stationary at the beginning.
      mean_acc_stopped_ = Measures.getMeanAcc();
    } else {
      const auto &mean_acc = Measures.getMeanAcc();
      if (isMotionStopped(mean_acc_stopped_, mean_acc, acc_diff_thr_)) {
        log(LogLevel::Warn,
            "Waiting for motion to perform gravity alignment...now a robot has been stopped");
        num_consecutive_moving_frames_ = 0;
      } else {
        num_consecutive_moving_frames_ = std::min(num_consecutive_moving_frames_ + 1, 100000);
      }
    }
  }

  flg_EKF_inited_ = (Measures.lidar_beg_time - first_lidar_time_) < INIT_TIME ? false : true;
  lasermapFovSegment();

  feats_down_body_.reset(new PointCloudXYZI());
  down_size_filter_.setInputCloud(feats_undistort_);
  down_size_filter_.filter(*feats_down_body_);
  feats_down_size_ = feats_down_body_->points.size();

  if (ikd_tree_.Root_Node == nullptr) {
    if (feats_down_size_ > 5) {
      ikd_tree_.set_downsample_param(filter_size_map_min_);
      feats_down_world_->resize(feats_down_size_);
      for (int i = 0; i < feats_down_size_; i++) {
        pointBodyToWorld(
            &(feats_down_body_->points[i]), &(feats_down_world_->points[i]), latest_state_);
      }
      ikd_tree_.Build(feats_down_world_->points);
    }
    return std::nullopt;
  }
  kdtree_size_st_ = ikd_tree_.size();

  /*** ICP and iterated Kalman filter update ***/
  if (feats_down_size_ < 5) {
    log(LogLevel::Warn, "No point, skip this scan!");
    return std::nullopt;
  }

  normvec_->resize(feats_down_size_);
  feats_down_world_->resize(feats_down_size_);

  nearest_points_.resize(feats_down_size_);

  /*** iterated state estimation ***/
  double solve_H_time = 0;
  kf_.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);

  /***** Perform gravity alignment *****/
  // NOTE(hlim): Introduce a delay using `num_moving_frames_thr_` to make sure
  // the gravity vectors are sufficiently updated.
  if (enable_gravity_alignment_ && !is_gravity_aligned_ && has_base_frame_ &&
      (num_consecutive_moving_frames_ > num_moving_frames_thr_)) {
    if (!offset_R_I_B_.has_value()) {
      offset_R_I_B_ = M3D(lidar_R_wrt_base_ * latest_state_.offset_R_L_I.inverse());
    }
    const M3D &offset_R_I_B = *offset_R_I_B_;

    // NOTE(hlim): Here, we don't need to normalize the scale of vectors
    V3D gravity_direction = kf_.get_x().grav;
    if (global_gravity_directions_.size() < static_cast<size_t>(num_gravity_measurements_thr_)) {
      {
        std::stringstream ss;
        ss << "Waiting for motion: " << global_gravity_directions_.size() << " / "
           << num_gravity_measurements_thr_;
        log(LogLevel::Info, ss.str());
      }

      global_gravity_directions_.push_back(offset_R_I_B * gravity_direction);
    } else {
      V3D avg_global_gravity_vec = Eigen::Vector3d::Zero();
      for (const auto &gravity_vec : global_gravity_directions_) {
        avg_global_gravity_vec += gravity_vec;
      }
      avg_global_gravity_vec /= global_gravity_directions_.size();

      R_gravity_aligned_ = computeRelativeRotation(avg_global_gravity_vec, g_base_);

      {
        std::stringstream ss;
        ss << "Gravity alignment complete! `R_gravity_aligned`: " << R_gravity_aligned_;
        log(LogLevel::Info, ss.str());
      }

      is_gravity_aligned_ = true;
    }
  }

  latest_state_          = kf_.get_x();
  kf_for_preintegration_ = kf_;
  // Update corrected rotation here
  latest_state_.pos = R_gravity_aligned_ * latest_state_.pos;
  latest_state_.rot = R_gravity_aligned_ * latest_state_.rot;

  if (enable_gravity_alignment_ && !is_gravity_aligned_ && has_base_frame_) {
    log(LogLevel::Warn,
        "Gravity alignment is enabled but not yet completed. Waiting for alignment...");
    return std::nullopt;
  }

  Result result;
  result.t     = lidar_end_time_;
  result.state = toState(latest_state_);
  result.pose  = result.state.imuPose();
  result.vel   = result.state.vel;
  result.cov   = kf_.get_P();
  result.num_effective_points = effect_feat_num_;
  result.mean_residual        = res_mean_last_;

  mapIncremental();

  result.scan_undistorted = cloud_undistort_;
  result.scan_down        = feats_down_body_;
  switch (config_.world_scan) {
    case Config::WorldScan::None:
      break;
    case Config::WorldScan::Downsampled:
      result.scan_world = transformToWorld(*feats_down_body_, result.state);
      break;
    case Config::WorldScan::Dense:
      result.scan_world = transformToWorld(*cloud_undistort_, result.state);
      break;
  }
  return result;
}

// ---------------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------------
FastLio::FastLio(const Config &config, LogFunction log)
    : impl_(std::make_unique<Impl>(config, std::move(log))) {}

FastLio::~FastLio() = default;

std::optional<ImuPropagation> FastLio::addImu(const ImuSample &imu) { return impl_->addImu(imu); }

void FastLio::addLidar(const LidarScan &scan) { impl_->addLidar(scan); }

std::optional<Result> FastLio::step() {
  if (impl_->syncPackages(impl_->Measures_, impl_->verbose_)) {
    return impl_->processLidarAndImu(impl_->Measures_);
  }
  return std::nullopt;
}

const Config &FastLio::config() const { return impl_->config_; }

State FastLio::state() const { return toState(impl_->latest_state_); }

StateCovariance FastLio::covariance() const { return impl_->kf_.get_P(); }

bool FastLio::isGravityAligned() const { return impl_->is_gravity_aligned_; }

Eigen::Matrix3d FastLio::gravityAlignment() const { return impl_->R_gravity_aligned_; }

PointCloud::Ptr FastLio::map() const {
  PointCloud::Ptr out(new PointCloud());
  if (impl_->ikd_tree_.Root_Node == nullptr) return out;
  PointVector storage;
  impl_->ikd_tree_.flatten(impl_->ikd_tree_.Root_Node, storage, NOT_RECORD);
  out->points.assign(storage.begin(), storage.end());
  out->width  = out->points.size();
  out->height = 1;
  return out;
}

}  // namespace sparo_fast_lio
