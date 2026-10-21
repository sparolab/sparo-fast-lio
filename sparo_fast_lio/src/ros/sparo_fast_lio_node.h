// SPARO-FAST-LIO ROS 2 wrapper: reads parameters, converts messages, runs the
// `sparo_fast_lio::FastLio` core and publishes the same topics / TF as SPARK-FAST-LIO.
#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <optional>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include "sparo_fast_lio/fast_lio.h"
#include "sparo_fast_lio/preprocessor.h"

namespace sparo_fast_lio {

struct PoseStruct {
  Eigen::Vector3d position_;
  Eigen::Quaterniond orientation_;
};

class SparoFastLioNode : public rclcpp::Node {
 public:
  explicit SparoFastLioNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  ~SparoFastLioNode() override;

 private:
  bool lookupBaseExtrinsics(Eigen::Vector3d &lidar_T_wrt_base, Eigen::Matrix3d &lidar_R_wrt_base);

  void pclPointBodyToWorld(PointType const *const pi, PointType *const po) const;
  void pclPointBodyLidarToIMU(PointType const *const pi, PointType *const po) const;
  void pclPointBodyLidarToBase(PointType const *const pi, PointType *const po) const;
  void pclPointIMUToLiDAR(PointType const *const pi, PointType *const po) const;
  void pclPointIMUToBase(PointType const *const pi, PointType *const po);

  void standardLiDARCallback(const sensor_msgs::msg::PointCloud2 &msg);
  void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg);

  void publishOdometry(const State &state,
                       const StateCovariance &P,
                       const builtin_interfaces::msg::Time &stamp);
  void publishPath(const State &state);
  void publishFrameWorld(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloud);
  void publishFrame(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloud,
                    const std::string &frame);

  PoseStruct transformPoseWrtLidarFrame(const State &state) const;
  PoseStruct transformPoseWrtBaseFrame(const State &state);
  void computeBaseOffsets(const State &state);

  template <typename T>
  void setPoseStamp(const State &state, T &out, const std::string &frame) {
    if (frame == "imu") {
      out.pose.position.x    = state.pos(0);
      out.pose.position.y    = state.pos(1);
      out.pose.position.z    = state.pos(2);
      const auto quat        = state.rot.coeffs();
      out.pose.orientation.x = quat[0];
      out.pose.orientation.y = quat[1];
      out.pose.orientation.z = quat[2];
      out.pose.orientation.w = quat[3];
    } else if (frame == "lidar") {
      const auto &p          = transformPoseWrtLidarFrame(state);
      out.pose.position.x    = p.position_(0);
      out.pose.position.y    = p.position_(1);
      out.pose.position.z    = p.position_(2);
      out.pose.orientation.x = p.orientation_.x();
      out.pose.orientation.y = p.orientation_.y();
      out.pose.orientation.z = p.orientation_.z();
      out.pose.orientation.w = p.orientation_.w();
    } else if (frame == "base") {
      const auto &p          = transformPoseWrtBaseFrame(state);
      out.pose.position.x    = p.position_(0);
      out.pose.position.y    = p.position_(1);
      out.pose.position.z    = p.position_(2);
      out.pose.orientation.x = p.orientation_.x();
      out.pose.orientation.y = p.orientation_.y();
      out.pose.orientation.z = p.orientation_.z();
      out.pose.orientation.w = p.orientation_.w();
    } else {
      throw std::invalid_argument("Invalid visualization frame has been given");
    }
  }

  void main();
  void publishResult(const Result &result);

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_full_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_lidar_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_body_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_base_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Clock::SharedPtr clock_;
  // scans are processed on their own thread, so a slow scan never holds up the LiDAR and IMU callbacks
  std::thread worker_;
  std::atomic<bool> stop_{false};

  bool pcd_save_en_       = false;
  bool path_en_           = true;
  bool scan_pub_en_       = false;
  bool dense_pub_en_      = false;
  bool scan_lidar_pub_en_ = false;
  bool scan_body_pub_en_  = false;
  bool scan_base_pub_en_  = false;
  bool pcl_verbose_       = true;

  std::string map_file_path_;
  std::string save_dir_;
  std::string sequence_name_;
  std::string map_frame_;
  std::string lidar_frame_;
  std::string base_frame_;
  std::string imu_frame_;
  std::string viz_frame_;

  int publish_count_     = 0;
  int pcd_save_interval_ = -1;
  int pcd_index_         = 0;
  int path_counter_      = 0;
  int scan_wait_num_     = 0;

  double extrinsics_timeout_s_ = 10.0;
  double lidar_end_time_       = 0.0;

  Eigen::Vector3d lidar_T_wrt_base_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d lidar_R_wrt_base_ = Eigen::Matrix3d::Identity();
  // imu -> base offsets, fixed from the first state they are needed for (as upstream)
  std::optional<Eigen::Matrix3d> offset_R_B_I_;
  Eigen::Vector3d offset_T_B_I_ = Eigen::Vector3d::Zero();

  Config config_;
  std::unique_ptr<Preprocessor> preprocessor_;
  std::unique_ptr<FastLio> lio_;

  // Latest LiDAR-updated outputs (used by the publishers)
  State latest_state_;
  PointCloud::Ptr cloud_undistort_;
  PointCloud::Ptr feats_down_body_;
  PointCloud::Ptr cloud_to_be_saved_;

  nav_msgs::msg::Path path_msg_;
  nav_msgs::msg::Odometry odomAftMapped_;
  geometry_msgs::msg::PoseStamped msg_body_pose_;
};

}  // namespace sparo_fast_lio
