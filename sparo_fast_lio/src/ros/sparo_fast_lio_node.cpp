#include "sparo_fast_lio_node.h"

#include <cmath>
#include <stdexcept>

#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp_components/register_node_macro.hpp>

#define PUBFRAME_PERIOD (20)

namespace sparo_fast_lio {

namespace {
// Same conversion as rclcpp::Time(stamp).seconds() in the original node.
inline double toSec(const builtin_interfaces::msg::Time &stamp) {
  return rclcpp::Time(stamp).seconds();
}
}  // namespace

SparoFastLioNode::SparoFastLioNode(const rclcpp::NodeOptions &options)
    : Node("sparo_fast_lio_node", options), clock_(get_clock()) {
  path_en_           = declare_parameter<bool>("publish.path_en", true);
  scan_pub_en_       = declare_parameter<bool>("publish.scan_publish_en", false);
  dense_pub_en_      = declare_parameter<bool>("publish.dense_publish_en", false);
  scan_lidar_pub_en_ = declare_parameter<bool>("publish.scan_lidarframe_pub_en", false);
  scan_body_pub_en_  = declare_parameter<bool>("publish.scan_bodyframe_pub_en", false);
  scan_base_pub_en_  = declare_parameter<bool>("publish.scan_baseframe_pub_en", false);

  config_.max_iteration = declare_parameter<int>("max_iteration", 4);

  map_file_path_ = declare_parameter<std::string>("map_file_path", "");
  save_dir_      = declare_parameter<std::string>("common.save_dir", "");
  sequence_name_ = declare_parameter<std::string>("common.sequence_name", "");
  map_frame_     = declare_parameter<std::string>("common.map_frame", "odom");
  lidar_frame_   = declare_parameter<std::string>("common.lidar_frame", "lidar");
  base_frame_    = declare_parameter<std::string>("common.base_frame", "");
  imu_frame_     = declare_parameter<std::string>("common.imu_frame", "imu");
  viz_frame_     = declare_parameter<std::string>("common.visualization_frame", "imu");
  // Only affects the (disabled) Livox front-end upstream; declared for parameter compatibility.
  declare_parameter<bool>("common.time_sync_en", false);

  config_.filter_size_map  = declare_parameter<double>("filter_size_map", 0.5);
  config_.cube_side_length = declare_parameter<double>("cube_side_length", 200.0);
  config_.det_range        = declare_parameter<double>("mapping.det_range", 300.0);
  config_.fov_degree       = declare_parameter<double>("mapping.fov_degree", 360.0);
  config_.gyr_cov          = declare_parameter<double>("mapping.gyr_cov", 0.1);
  config_.acc_cov          = declare_parameter<double>("mapping.acc_cov", 0.1);
  config_.b_gyr_cov        = declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
  config_.b_acc_cov        = declare_parameter<double>("mapping.b_acc_cov", 0.0001);

  auto &ga  = config_.gravity_alignment;
  ga.enable = declare_parameter<bool>("gravity_alignment.enable_gravity_alignment", true);
  ga.acc_diff_thr          = declare_parameter<double>("gravity_alignment.acc_diff_thr", 0.2);
  ga.num_moving_frames_thr = declare_parameter<int>("gravity_alignment.num_moving_frames_thr", 20);
  ga.num_gravity_measurements_thr =
      declare_parameter<int>("gravity_alignment.num_gravity_measurements_thr", 20);

  config_.verbose = declare_parameter<bool>("verbose", false);
  pcl_verbose_    = declare_parameter<bool>("pcl_verbose", true);
  if (!pcl_verbose_) {
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  }

  declare_parameter<bool>("runtime_pos_log_enable", false);  // kept for compatibility (unused)
  config_.extrinsic_est_en = declare_parameter<bool>("mapping.extrinsic_est_en", false);
  extrinsics_timeout_s_    = declare_parameter<double>("extrinsics_timeout_s", 10.0);
  pcd_save_en_             = declare_parameter<bool>("pcd_save.pcd_save_en", false);
  pcd_save_interval_       = declare_parameter<int>("pcd_save.interval", -1);

  config_.point_filter_num = declare_parameter<int>("point_filter_num", 4);

  // extrinsic_T and extrinsic_R are sized 3 and 9 respectively
  std::vector<double> extrinT{0.0, 0.0, 0.0};
  std::vector<double> extrinR{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  extrinT = declare_parameter<std::vector<double>>("mapping.extrinsic_T", extrinT);
  extrinR = declare_parameter<std::vector<double>>("mapping.extrinsic_R", extrinR);
  if (extrinT.size() == 3 && extrinR.size() == 9) {
    config_.extrinsic_T << extrinT[0], extrinT[1], extrinT[2];
    config_.extrinsic_R << extrinR[0], extrinR[1], extrinR[2], extrinR[3], extrinR[4], extrinR[5],
        extrinR[6], extrinR[7], extrinR[8];
  }

  auto g_vec = declare_parameter<std::vector<double>>("gravity_alignment.g_base", {0.0, 0.0, -1.0});
  ga.g_base << g_vec[0], g_vec[1], g_vec[2];

  rclcpp::QoS lidar_qos(rclcpp::KeepLast(10));
  lidar_qos.reliable();
  lidar_qos.durability_volatile();
  sub_lidar_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "lidar",
      lidar_qos,
      std::bind(&SparoFastLioNode::standardLiDARCallback, this, std::placeholders::_1));

  auto imu_qos = rclcpp::SensorDataQoS();
  sub_imu_     = create_subscription<sensor_msgs::msg::Imu>(
      "imu", imu_qos, std::bind(&SparoFastLioNode::imuCallback, this, std::placeholders::_1));

  rclcpp::QoS qos((rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile()));
  pub_cloud_full_  = create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered", qos);
  pub_cloud_lidar_ = create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered_lidar", qos);
  pub_cloud_body_  = create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered_body", qos);
  pub_cloud_base_  = create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered_base", qos);

  pub_odom_                 = create_publisher<nav_msgs::msg::Odometry>("odometry", qos);
  pub_path_                 = create_publisher<nav_msgs::msg::Path>("path", qos);
  path_msg_.header.frame_id = map_frame_;

  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
  tf_buffer_      = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_    = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  auto &pre                  = config_.preprocess;
  pre.blind                  = declare_parameter<double>("preprocess.blind", 0.01);
  pre.blind_for_human_pilots = declare_parameter<double>("preprocess.blind_for_human_pilots", 1.5);
  pre.lidar_type             = static_cast<LidarType>(
      declare_parameter<int>("preprocess.lidar_type", static_cast<int>(LidarType::AVIA)));
  pre.scan_line      = declare_parameter<int>("preprocess.scan_line", 16);
  pre.timestamp_unit = static_cast<TimeUnit>(
      declare_parameter<int>("preprocess.timestamp_unit", static_cast<int>(TimeUnit::US)));
  pre.scan_rate        = declare_parameter<int>("preprocess.scan_rate", 10);
  pre.point_filter_num = declare_parameter<int>("point_filter_num_for_preprocessing", 1);

  // The node computes its own world-frame clouds (it supports several visualization frames).
  config_.world_scan = Config::WorldScan::None;

  cloud_undistort_.reset(new PointCloud());
  feats_down_body_.reset(new PointCloud());
  cloud_to_be_saved_.reset(new PointCloud());

  bool base_lookup_ok = true;
  if (!base_frame_.empty()) {
    base_lookup_ok = lookupBaseExtrinsics(lidar_T_wrt_base_, lidar_R_wrt_base_);
    if (base_lookup_ok) {
      Eigen::Isometry3d base_T_lidar = Eigen::Isometry3d::Identity();
      base_T_lidar.linear()          = lidar_R_wrt_base_;
      base_T_lidar.translation()     = lidar_T_wrt_base_;
      config_.base_T_lidar           = base_T_lidar;
    }
  }

  auto logger = this->get_logger();
  auto log_fn = [logger](LogLevel level, const std::string &msg) {
    switch (level) {
      case LogLevel::Debug:
        RCLCPP_DEBUG(logger, "%s", msg.c_str());
        break;
      case LogLevel::Info:
        RCLCPP_INFO(logger, "%s", msg.c_str());
        break;
      case LogLevel::Warn:
        RCLCPP_WARN(logger, "%s", msg.c_str());
        break;
      case LogLevel::Error:
        RCLCPP_ERROR(logger, "%s", msg.c_str());
        break;
      case LogLevel::Fatal:
        RCLCPP_FATAL(logger, "%s", msg.c_str());
        break;
    }
  };
  preprocessor_ = std::make_unique<Preprocessor>(config_.preprocess, log_fn);
  lio_          = std::make_unique<FastLio>(config_, log_fn);

  if (!base_lookup_ok) {
    // Same as upstream: without the base extrinsics the main loop is never started.
    RCLCPP_ERROR(this->get_logger(), "Failed to lookup transform.");
    return;
  }

  main_loop_timer_ =
      create_wall_timer(std::chrono::milliseconds(1), std::bind(&SparoFastLioNode::main, this));

  if ((pre.point_filter_num != 1 && config_.point_filter_num > 1)) {
    RCLCPP_WARN(this->get_logger(),
                "Points may be too sparse. Set 'preprocessor_->point_filter_num = 1' and tune "
                "'point_filter_num_' instead.");
  }

  RCLCPP_INFO(this->get_logger(), "SparoFastLioNode constructed");
}

bool SparoFastLioNode::lookupBaseExtrinsics(Eigen::Vector3d &lidar_T_wrt_base,
                                            Eigen::Matrix3d &lidar_R_wrt_base) {
  RCLCPP_INFO(this->get_logger(),
              "Looking up transform from %s -> %s",
              base_frame_.c_str(),
              lidar_frame_.c_str());

  const auto lookup_time = rclcpp::Time(0);
  bool has_transform     = false;
  std::string err_str;
  auto start_time          = this->now();
  rclcpp::Duration timeout = rclcpp::Duration::from_seconds(extrinsics_timeout_s_);
  rclcpp::Rate rate(10.0);  // Just 10 Hz works

  while (rclcpp::ok()) {
    if (tf_buffer_->canTransform(
            base_frame_, lidar_frame_, lookup_time, tf2::durationFromSec(0.0), &err_str)) {
      RCLCPP_INFO_STREAM(this->get_logger(), "\033[1;32mExtrinsics detected.\033[1;0m");
      has_transform = true;
      break;
    }

    const auto time_since_start = now() - start_time;
    if (extrinsics_timeout_s_ > 0.0 && time_since_start > timeout) {
      RCLCPP_ERROR_STREAM(this->get_logger(),
                          "Timeout after "
                              << timeout.seconds() << " seconds waiting for transform from '"
                              << lidar_frame_ << "' to '" << base_frame_ << "': " << err_str);
      break;
    }

    RCLCPP_WARN_STREAM_SKIPFIRST_THROTTLE(get_logger(),
                                          *clock_,
                                          5000,
                                          "Waiting for transform from '" << lidar_frame_ << "' to '"
                                                                         << base_frame_
                                                                         << "': " << err_str);

    rate.sleep();
  }

  if (!has_transform) {
    return has_transform;
  }

  const auto &transform = tf_buffer_->lookupTransform(base_frame_, lidar_frame_, lookup_time);
  lidar_T_wrt_base(0)   = transform.transform.translation.x;
  lidar_T_wrt_base(1)   = transform.transform.translation.y;
  lidar_T_wrt_base(2)   = transform.transform.translation.z;

  Eigen::Quaterniond q(transform.transform.rotation.w,
                       transform.transform.rotation.x,
                       transform.transform.rotation.y,
                       transform.transform.rotation.z);

  lidar_R_wrt_base = q.toRotationMatrix();

  RCLCPP_INFO(this->get_logger(),
              "Translation: [%.3f, %.3f, %.3f]",
              lidar_T_wrt_base(0),
              lidar_T_wrt_base(1),
              lidar_T_wrt_base(2));

  RCLCPP_INFO(this->get_logger(),
              "Rotation (Quaternion): [%.3f, %.3f, %.3f, %.3f]",
              q.x(),
              q.y(),
              q.z(),
              q.w());

  return has_transform;
}

void SparoFastLioNode::pclPointBodyToWorld(PointType const *const pi, PointType *const po) const {
  const auto &s = latest_state_;
  Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
  Eigen::Vector3d p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

  po->x         = p_global(0);
  po->y         = p_global(1);
  po->z         = p_global(2);
  po->intensity = pi->intensity;
}

void SparoFastLioNode::pclPointBodyLidarToIMU(PointType const *const pi,
                                              PointType *const po) const {
  Eigen::Vector3d p_body_lidar(pi->x, pi->y, pi->z);
  Eigen::Vector3d p_body_imu(latest_state_.offset_R_L_I * p_body_lidar +
                             latest_state_.offset_T_L_I);

  po->x         = p_body_imu(0);
  po->y         = p_body_imu(1);
  po->z         = p_body_imu(2);
  po->intensity = pi->intensity;
}

void SparoFastLioNode::pclPointBodyLidarToBase(PointType const *const pi,
                                               PointType *const po) const {
  Eigen::Vector3d p_body_lidar(pi->x, pi->y, pi->z);
  Eigen::Vector3d p_body_base(lidar_R_wrt_base_ * p_body_lidar + lidar_T_wrt_base_);

  po->x         = p_body_base(0);
  po->y         = p_body_base(1);
  po->z         = p_body_base(2);
  po->intensity = pi->intensity;
}

void SparoFastLioNode::pclPointIMUToLiDAR(PointType const *const pi, PointType *const po) const {
  Eigen::Vector3d p_body_imu(pi->x, pi->y, pi->z);
  Eigen::Vector3d p_body_lidar(latest_state_.offset_R_L_I.inverse() *
                               (p_body_imu - latest_state_.offset_T_L_I));

  po->x         = p_body_lidar(0);
  po->y         = p_body_lidar(1);
  po->z         = p_body_lidar(2);
  po->intensity = pi->intensity;
}

// Upstream computes the imu->base offsets once (function-local statics) from the first state
// they are needed for; we keep that behavior but store the evaluated matrices.
void SparoFastLioNode::computeBaseOffsets(const State &state) {
  if (offset_R_B_I_.has_value()) return;
  offset_R_B_I_ = Eigen::Matrix3d(state.offset_R_L_I.toRotationMatrix() * lidar_R_wrt_base_.inverse());
  offset_T_B_I_ = -1 * (*offset_R_B_I_) * lidar_T_wrt_base_ + state.offset_T_L_I;
}

void SparoFastLioNode::pclPointIMUToBase(PointType const *const pi, PointType *const po) {
  computeBaseOffsets(latest_state_);
  const auto &offset_R_B_I = *offset_R_B_I_;
  const auto &offset_T_B_I = offset_T_B_I_;

  Eigen::Vector3d p_body_imu(pi->x, pi->y, pi->z);
  Eigen::Vector3d p_body_base(offset_R_B_I.inverse() * (p_body_imu - offset_T_B_I));

  po->x         = p_body_base(0);
  po->y         = p_body_base(1);
  po->z         = p_body_base(2);
  po->intensity = pi->intensity;
}

void SparoFastLioNode::standardLiDARCallback(const sensor_msgs::msg::PointCloud2 &msg) {
  const double msg_time = toSec(msg.header.stamp);

  PointCloud::Ptr ptr;
  switch (config_.preprocess.lidar_type) {
    case LidarType::VELO16: {
      pcl::PointCloud<VelodynePoint> pl_orig;
      pcl::fromROSMsg(msg, pl_orig);
      ptr = preprocessor_->process(pl_orig);
      break;
    }
    case LidarType::OUST64:
    case LidarType::KMOUST64: {
      pcl::PointCloud<OusterPoint> pl_orig;
      pcl::fromROSMsg(msg, pl_orig);
      ptr = preprocessor_->process(pl_orig);
      break;
    }
    default:
      RCLCPP_FATAL(rclcpp::get_logger("Preprocess"), "Error LiDAR Type");
      ptr.reset(new PointCloud());
      break;
  }

  lio_->addLidar(LidarScan{msg_time, ptr});
}

void SparoFastLioNode::imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg) {
  ++publish_count_;

  ImuSample imu;
  imu.t = toSec(msg->header.stamp);
  imu.acc << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
  imu.gyr << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;

  const auto propagated = lio_->addImu(imu);
  if (propagated) {
    publishOdometry(propagated->state, propagated->cov, msg->header.stamp);
  }
}

void SparoFastLioNode::publishOdometry(const State &state,
                                       const StateCovariance &P,
                                       const builtin_interfaces::msg::Time &stamp) {
  odomAftMapped_.header.frame_id = map_frame_;
  odomAftMapped_.header.stamp    = stamp;

  setPoseStamp(state, odomAftMapped_.pose, viz_frame_);  // our template function

  if (viz_frame_ == "lidar") {
    odomAftMapped_.child_frame_id = lidar_frame_;
  } else if (viz_frame_ == "base") {
    odomAftMapped_.child_frame_id = base_frame_;
  } else if (viz_frame_ == "imu") {
    odomAftMapped_.child_frame_id = imu_frame_;
  } else {
    throw std::invalid_argument("Invalid visualization frame has been given");
  }

  // fill the covariance
  for (int i = 0; i < 6; i++) {
    int k                                     = (i < 3) ? (i + 3) : (i - 3);
    odomAftMapped_.pose.covariance[i * 6 + 0] = P(k, 3);
    odomAftMapped_.pose.covariance[i * 6 + 1] = P(k, 4);
    odomAftMapped_.pose.covariance[i * 6 + 2] = P(k, 5);
    odomAftMapped_.pose.covariance[i * 6 + 3] = P(k, 0);
    odomAftMapped_.pose.covariance[i * 6 + 4] = P(k, 1);
    odomAftMapped_.pose.covariance[i * 6 + 5] = P(k, 2);
  }

  // publish
  pub_odom_->publish(odomAftMapped_);

  geometry_msgs::msg::TransformStamped transform_stamped;
  transform_stamped.header.stamp    = odomAftMapped_.header.stamp;
  transform_stamped.header.frame_id = map_frame_;
  transform_stamped.child_frame_id  = odomAftMapped_.child_frame_id;

  transform_stamped.transform.translation.x = odomAftMapped_.pose.pose.position.x;
  transform_stamped.transform.translation.y = odomAftMapped_.pose.pose.position.y;
  transform_stamped.transform.translation.z = odomAftMapped_.pose.pose.position.z;
  transform_stamped.transform.rotation      = odomAftMapped_.pose.pose.orientation;

  tf_broadcaster_->sendTransform(transform_stamped);
}

void SparoFastLioNode::publishPath(const State &state) {
  setPoseStamp(state, msg_body_pose_, viz_frame_);
  msg_body_pose_.header.stamp    = rclcpp::Time(lidar_end_time_ * 1e9);
  msg_body_pose_.header.frame_id = map_frame_;

  path_counter_++;
  if (path_counter_ % 10 == 0) {
    path_msg_.poses.push_back(msg_body_pose_);
    pub_path_->publish(path_msg_);
  }
}

void SparoFastLioNode::publishFrameWorld(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloud) {
  if (!scan_pub_en_) {
    return;
  }

  // choose which cloud to publish
  PointCloud::Ptr laserCloudFullRes(dense_pub_en_ ? cloud_undistort_ : feats_down_body_);

  int size = laserCloudFullRes->points.size();
  // allocate world frames
  PointCloud::Ptr laserCloudWorld(new PointCloud(size, 1));
  PointCloud::Ptr laserCloudTmp(new PointCloud(size, 1));

  for (int i = 0; i < size; i++) {
    if (viz_frame_ == "imu") {
      pclPointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
    } else if (viz_frame_ == "lidar") {
      pclPointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudTmp->points[i]);
      pclPointIMUToLiDAR(&laserCloudTmp->points[i], &laserCloudWorld->points[i]);
    } else if (viz_frame_ == "base") {
      pclPointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudTmp->points[i]);
      pclPointIMUToBase(&laserCloudTmp->points[i], &laserCloudWorld->points[i]);
    } else {
      throw std::invalid_argument("Invalid visualization frame has been given");
    }
  }

  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*laserCloudWorld, cloud_msg);
  // use lidar_end_time_ for the timestamp
  cloud_msg.header.stamp    = rclcpp::Time(lidar_end_time_ * 1e9);  // from seconds
  cloud_msg.header.frame_id = map_frame_;

  pubCloud->publish(cloud_msg);
  publish_count_ -= PUBFRAME_PERIOD;

  // Optionally do the pcd_save_en_ part
  if (pcd_save_en_) {
    int nsize = cloud_undistort_->points.size();
    PointCloud::Ptr laserCloudWorld2(new PointCloud(nsize, 1));

    for (int i = 0; i < nsize; i++) {
      pclPointBodyToWorld(&cloud_undistort_->points[i], &laserCloudWorld2->points[i]);
    }
    if (pcd_save_interval_ > 0) {
      *cloud_to_be_saved_ += *laserCloudWorld2;
    }

    scan_wait_num_++;
    if (cloud_to_be_saved_->size() > 0 && pcd_save_interval_ > 0 &&
        scan_wait_num_ >= pcd_save_interval_) {
      pcd_index_++;
      std::string all_points_dir(std::string(ROOT_DIR) + "PCD/scans_" + std::to_string(pcd_index_) +
                                 ".pcd");
      pcl::PCDWriter pcd_writer;
      std::cout << "Current scan saved to /PCD/ " << all_points_dir << std::endl;
      pcd_writer.writeBinary(all_points_dir, *cloud_to_be_saved_);
      cloud_to_be_saved_->clear();
      scan_wait_num_ = 0;
    }
  }
}

void SparoFastLioNode::publishFrame(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloud,
    const std::string &frame) {
  int size = cloud_undistort_->points.size();
  PointCloud::Ptr laserCloudTransformed(new PointCloud(size, 1));
  sensor_msgs::msg::PointCloud2 cloud_msg;

  if (frame == "lidar") {
    // direct copy
    for (int i = 0; i < size; i++) {
      laserCloudTransformed->points[i] = cloud_undistort_->points[i];
    }
    pcl::toROSMsg(*laserCloudTransformed, cloud_msg);
    cloud_msg.header.stamp    = rclcpp::Time(lidar_end_time_ * 1e9);
    cloud_msg.header.frame_id = lidar_frame_;
  } else if (frame == "imu") {
    for (int i = 0; i < size; i++) {
      pclPointBodyLidarToIMU(&cloud_undistort_->points[i], &laserCloudTransformed->points[i]);
    }
    pcl::toROSMsg(*laserCloudTransformed, cloud_msg);
    cloud_msg.header.stamp    = rclcpp::Time(lidar_end_time_ * 1e9);
    cloud_msg.header.frame_id = imu_frame_;
  } else if (frame == "base") {
    for (int i = 0; i < size; i++) {
      pclPointBodyLidarToBase(&cloud_undistort_->points[i], &laserCloudTransformed->points[i]);
    }
    pcl::toROSMsg(*laserCloudTransformed, cloud_msg);
    cloud_msg.header.stamp    = rclcpp::Time(lidar_end_time_ * 1e9);
    cloud_msg.header.frame_id = base_frame_;
  } else {
    throw std::invalid_argument("Invalid frame has been given");
  }

  pubCloud->publish(cloud_msg);
  publish_count_ -= PUBFRAME_PERIOD;
}

PoseStruct SparoFastLioNode::transformPoseWrtLidarFrame(const State &state) const {
  // offset_A_B: transformation matrix of A w.r.t. B
  Eigen::Vector3d lidar_position = state.offset_R_L_I.inverse() * (state.rot * state.offset_T_L_I +
                                                                   state.pos - state.offset_T_L_I);

  Eigen::Quaterniond lidar_orientation =
      state.offset_R_L_I.inverse() * state.rot * state.offset_R_L_I;

  PoseStruct output;
  output.position_    = lidar_position;
  output.orientation_ = lidar_orientation;
  return output;
}

PoseStruct SparoFastLioNode::transformPoseWrtBaseFrame(const State &state) {
  computeBaseOffsets(state);
  const Eigen::Matrix3d &offset_R_B_I = *offset_R_B_I_;
  const Eigen::Vector3d &offset_T_B_I = offset_T_B_I_;

  Eigen::Vector3d base_position =
      offset_R_B_I.inverse() * (state.rot * offset_T_B_I + state.pos - offset_T_B_I);

  Eigen::Quaterniond base_orientation =
      Eigen::Quaterniond(offset_R_B_I.inverse() * state.rot * offset_R_B_I);

  PoseStruct output;
  output.position_    = base_position;
  output.orientation_ = base_orientation;
  return output;
}

void SparoFastLioNode::main() {
  if (auto result = lio_->step()) {
    publishResult(*result);
  }
}

void SparoFastLioNode::publishResult(const Result &result) {
  lidar_end_time_  = result.t;
  latest_state_    = result.state;
  cloud_undistort_ = result.scan_undistorted;
  feats_down_body_ = result.scan_down;

  /******* Publish topics *******/
  const auto stamp = rclcpp::Time(lidar_end_time_ * 1e9);
  publishOdometry(latest_state_, result.cov, stamp);

  if (path_en_) {
    publishPath(latest_state_);
  }
  if (scan_pub_en_) {
    publishFrameWorld(pub_cloud_full_);
    if (scan_lidar_pub_en_) publishFrame(pub_cloud_lidar_, "lidar");
    if (scan_body_pub_en_) publishFrame(pub_cloud_body_, "imu");
    if (scan_base_pub_en_) publishFrame(pub_cloud_base_, "base");
  }
}

}  // namespace sparo_fast_lio

RCLCPP_COMPONENTS_REGISTER_NODE(sparo_fast_lio::SparoFastLioNode)
