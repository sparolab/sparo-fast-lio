// Minimal ROS-free driver for the sparo_fast_lio core library.
//
// Input directory layout (see scripts/export_bag.py to create it from a ROS 2 bag):
//   <dir>/imu.csv     header + rows: stamp_ns,ax,ay,az,gx,gy,gz
//   <dir>/scans.csv   header + rows: stamp_ns,relative/path/to/scan.pcd
//   scan PCDs         fields x y z intensity time ring (velodyne layout) or
//                     x y z intensity t reflectivity ring ambient range (ouster layout)
//
// Parameters are read from a plain text file with one `name value...` entry per line using
// the ROS parameter names, e.g.
//   preprocess.lidar_type 2
//   preprocess.scan_line 128
//   mapping.extrinsic_T -0.41 0.0 0.30
//
// Usage: sparo_fast_lio_run_core <dir> <params.txt> <trajectory_tum.txt> [map.pcd]
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <pcl/io/pcd_io.h>

#include "sparo_fast_lio/fast_lio.h"
#include "sparo_fast_lio/preprocessor.h"

using namespace sparo_fast_lio;

namespace {

std::map<std::string, std::vector<double>> readParams(const std::string &path) {
  std::map<std::string, std::vector<double>> params;
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::string line;
  while (std::getline(in, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::replace(line.begin(), line.end(), '[', ' ');
    std::replace(line.begin(), line.end(), ']', ' ');
    std::istringstream ss(line);
    std::string key;
    if (!(ss >> key) || key[0] == '#') continue;
    std::vector<double> values;
    double v;
    while (ss >> v) values.push_back(v);
    params[key] = values;
  }
  return params;
}

Config makeConfig(const std::map<std::string, std::vector<double>> &p) {
  Config c;
  auto get = [&](const std::string &k, auto &out) {
    auto it = p.find(k);
    if (it != p.end() && !it->second.empty()) {
      out = static_cast<std::decay_t<decltype(out)>>(it->second[0]);
    }
  };
  int lidar_type = static_cast<int>(c.preprocess.lidar_type);
  int time_unit  = static_cast<int>(c.preprocess.timestamp_unit);
  get("preprocess.lidar_type", lidar_type);
  get("preprocess.timestamp_unit", time_unit);
  c.preprocess.lidar_type     = static_cast<LidarType>(lidar_type);
  c.preprocess.timestamp_unit = static_cast<TimeUnit>(time_unit);
  get("preprocess.scan_line", c.preprocess.scan_line);
  get("preprocess.scan_rate", c.preprocess.scan_rate);
  get("preprocess.blind", c.preprocess.blind);
  get("preprocess.blind_for_human_pilots", c.preprocess.blind_for_human_pilots);
  get("point_filter_num_for_preprocessing", c.preprocess.point_filter_num);

  get("mapping.gyr_cov", c.gyr_cov);
  get("mapping.acc_cov", c.acc_cov);
  get("mapping.b_gyr_cov", c.b_gyr_cov);
  get("mapping.b_acc_cov", c.b_acc_cov);
  get("mapping.fov_degree", c.fov_degree);
  get("mapping.det_range", c.det_range);
  get("mapping.extrinsic_est_en", c.extrinsic_est_en);
  get("max_iteration", c.max_iteration);
  get("filter_size_map", c.filter_size_map);
  get("cube_side_length", c.cube_side_length);
  get("point_filter_num", c.point_filter_num);
  get("verbose", c.verbose);

  if (auto it = p.find("mapping.extrinsic_T"); it != p.end() && it->second.size() == 3) {
    c.extrinsic_T << it->second[0], it->second[1], it->second[2];
  }
  if (auto it = p.find("mapping.extrinsic_R"); it != p.end() && it->second.size() == 9) {
    const auto &r = it->second;
    c.extrinsic_R << r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8];
  }
  c.world_scan = Config::WorldScan::None;
  return c;
}

struct Event {
  int64_t stamp_ns;
  bool is_imu;
  ImuSample imu;
  std::string pcd;
};

// Same conversion as rclcpp::Time(stamp).seconds()
double toSec(int64_t ns) { return static_cast<double>(ns) / 1e9; }

}  // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <dir> <params.txt> <trajectory_tum.txt> [map.pcd]\n";
    return 1;
  }
  const std::string dir = argv[1];
  const Config config   = makeConfig(readParams(argv[2]));

  // ---- read the input sequence ----
  std::vector<Event> events;
  {
    std::ifstream in(dir + "/imu.csv");
    std::string line;
    std::getline(in, line);  // header
    while (std::getline(in, line)) {
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream ss(line);
      Event e;
      e.is_imu = true;
      ss >> e.stamp_ns >> e.imu.acc.x() >> e.imu.acc.y() >> e.imu.acc.z() >> e.imu.gyr.x() >>
          e.imu.gyr.y() >> e.imu.gyr.z();
      e.imu.t = toSec(e.stamp_ns);
      events.push_back(e);
    }
  }
  {
    std::ifstream in(dir + "/scans.csv");
    std::string line;
    std::getline(in, line);  // header
    while (std::getline(in, line)) {
      const auto comma = line.find(',');
      Event e;
      e.is_imu   = false;
      e.stamp_ns = std::stoll(line.substr(0, comma));
      e.pcd      = dir + "/" + line.substr(comma + 1);
      events.push_back(e);
    }
  }
  // Scans are delivered when they are complete (stamp = first point, ~one sweep later in
  // practice); feeding them by header stamp is equivalent because the filter waits for IMU data.
  std::stable_sort(events.begin(), events.end(), [](const Event &a, const Event &b) {
    return a.stamp_ns < b.stamp_ns;
  });
  std::cout << "Loaded " << events.size() << " messages from " << dir << std::endl;

  // ---- run ----
  auto log = [](LogLevel level, const std::string &msg) {
    if (level >= LogLevel::Info) std::cerr << msg << std::endl;
  };
  Preprocessor preprocessor(config.preprocess, log);
  FastLio lio(config, log);

  std::ofstream traj(argv[3]);
  traj << std::setprecision(9) << std::fixed;
  int num_results     = 0;
  double sum_residual = 0.0;
  double sum_points   = 0.0;
  auto drain      = [&]() {
    while (auto result = lio.step()) {
      const Eigen::Quaterniond q(result->pose.linear());
      const Eigen::Vector3d t = result->pose.translation();
      traj << result->t << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " "
           << q.y() << " " << q.z() << " " << q.w() << "\n";
      ++num_results;
      sum_residual += result->mean_residual;
      sum_points += result->num_effective_points;
    }
  };

  for (const auto &e : events) {
    if (e.is_imu) {
      lio.addImu(e.imu);  // returns the IMU-rate prediction once the filter is initialized
    } else {
      PointCloud::Ptr scan;
      if (config.preprocess.lidar_type == LidarType::VELO16) {
        pcl::PointCloud<VelodynePoint> raw;
        pcl::io::loadPCDFile(e.pcd, raw);
        scan = preprocessor.process(raw);
      } else {
        pcl::PointCloud<OusterPoint> raw;
        pcl::io::loadPCDFile(e.pcd, raw);
        scan = preprocessor.process(raw);
      }
      lio.addLidar(LidarScan{toSec(e.stamp_ns), scan});
    }
    drain();
  }
  for (int i = 0; i < 5; ++i) drain();  // flush scans that became ready with the last message

  std::cout << "Wrote " << num_results << " poses to " << argv[3] << std::endl;
  if (num_results > 0) {
    std::cout << "Mean point-to-plane residual: " << sum_residual / num_results
              << " m, mean #correspondences: " << sum_points / num_results << std::endl;
  }
  const State final_state = lio.state();
  std::cout << "Final extrinsic (LiDAR in IMU): T = " << final_state.offset_T_L_I.transpose()
            << "\nR =\n"
            << final_state.offset_R_L_I.toRotationMatrix()
            << "\nGravity (world): " << final_state.grav.transpose() << std::endl;
  if (argc > 4) {
    auto map = lio.map();
    pcl::io::savePCDFileBinary(argv[4], *map);
    std::cout << "Wrote map with " << map->size() << " points to " << argv[4] << std::endl;
  }
  return 0;
}
