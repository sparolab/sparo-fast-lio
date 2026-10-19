#pragma once
// Internal FAST-LIO front-end (ROS-free port of SPARK-FAST-LIO's preprocess.h).
#include <string>
#include <vector>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "sparo_fast_lio/preprocessor.h"
#include "sparo_fast_lio/types.h"

#define IS_VALID(a) ((abs(a) > 1e8) ? true : false)

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

enum LID_TYPE { AVIA = 1, VELO16 = 2, OUST64 = 3, KMOUST64 = 4 };  // {1, 2, 3, 4}
enum TIME_UNIT { SEC = 0, MS = 1, US = 2, NS = 3 };
enum Feature { Nor, Poss_Plane, Real_Plane, Edge_Jump, Edge_Plane, Wire, ZeroPoint };
enum Surround { Prev, Next };
enum E_jump { Nr_nor, Nr_zero, Nr_180, Nr_inf, Nr_blind };

struct orgtype {
  double range;
  double dista;
  double angle[2];
  double intersect;
  E_jump edj[2];
  Feature ftype;
  orgtype() {
    range     = 0;
    edj[Prev] = Nr_nor;
    edj[Next] = Nr_nor;
    ftype     = Nor;
    intersect = 2;
  }
};

namespace velodyne_ros {
using Point = sparo_fast_lio::VelodynePoint;
}  // namespace velodyne_ros

namespace ouster_ros {
using Point = sparo_fast_lio::OusterPoint;
}  // namespace ouster_ros

class Preprocess {
 public:
  Preprocess();
  ~Preprocess();

  void process(const pcl::PointCloud<velodyne_ros::Point> &msg, PointCloudXYZI::Ptr &pcl_out);
  void process(const pcl::PointCloud<ouster_ros::Point> &msg, PointCloudXYZI::Ptr &pcl_out);
  void set(bool feat_en, int lid_type, double bld, int pfilt_num);

  // sensor_msgs::PointCloud2::ConstPtr pointcloud;
  PointCloudXYZI  pl_full, pl_corn, pl_surf, pl_from_pilots;
  PointCloudXYZI  pl_buff[128];   // maximum 128 line lidar
  std::vector<orgtype> typess[128];    // maximum 128 line lidar
  float           time_unit_scale;
  int             lidar_type, point_filter_num, N_SCANS, SCAN_RATE, time_unit;
  double          blind, blind_for_human_pilots;
  bool            feature_enabled, given_offset_time;
  sparo_fast_lio::LogFunction log;

 private:
  bool is_from_pilot_zone(const float &pt_x, const float &pt_y, const float &pt_z, const std::string mode = "velodyne");
  void set_time_unit_scale();
  void oust64_handler(const pcl::PointCloud<ouster_ros::Point> &pl_orig);
  void kmoust64_handler(pcl::PointCloud<ouster_ros::Point> pl_orig);
  void velodyne_handler(const pcl::PointCloud<velodyne_ros::Point> &pl_orig);
  void give_feature(PointCloudXYZI &pl, std::vector<orgtype> &types);
  int plane_judge(const PointCloudXYZI &pl, std::vector<orgtype> &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool small_plane(const PointCloudXYZI &pl,
                   std::vector<orgtype> &types,
                   uint i_cur,
                   uint &i_nex,
                   Eigen::Vector3d &curr_direct);
  bool edge_jump_judge(const PointCloudXYZI &pl, std::vector<orgtype> &types, uint i, Surround nor_dir);

  int    group_size;
  double disA, disB, inf_bound;
  double limit_maxmid, limit_midmin, limit_maxmin;
  double p2l_ratio;
  double jump_up_limit, jump_down_limit;
  double cos160;
  double edgea, edgeb;
  double smallp_intersect, smallp_ratio;
  double vx, vy, vz;
};
