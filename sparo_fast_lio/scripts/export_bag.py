#!/usr/bin/env python3
"""Export IMU + PointCloud2 topics of a ROS 2 bag into the plain-file format read by
examples/run_core.cpp (no ROS needed to *run* the core afterwards).

  <out>/imu.csv     stamp_ns,ax,ay,az,gx,gy,gz
  <out>/scans.csv   stamp_ns,scans/<stamp_ns>.pcd
  <out>/scans/*.pcd binary PCD with the PointCloud2 fields, in the original point order

Usage: python3 export_bag.py <bag_dir> <out_dir> [--imu /imu/data] [--lidar /ouster/points]
"""
import argparse
import os

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# PointField datatype -> (PCD TYPE, SIZE, numpy dtype)
PF = {1: ('I', 1, 'i1'), 2: ('U', 1, 'u1'), 3: ('I', 2, 'i2'), 4: ('U', 2, 'u2'),
      5: ('I', 4, 'i4'), 6: ('U', 4, 'u4'), 7: ('F', 4, 'f4'), 8: ('F', 8, 'f8')}


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def write_pcd(path, msg):
    fields = sorted(msg.fields, key=lambda f: f.offset)
    dtype = np.dtype({'names': [f.name for f in fields],
                      'formats': [('<' if not msg.is_bigendian else '>') + PF[f.datatype][2]
                                  for f in fields],
                      'offsets': [f.offset for f in fields],
                      'itemsize': msg.point_step})
    n = msg.width * msg.height
    raw = np.frombuffer(bytes(msg.data), dtype=dtype, count=n)
    packed = np.empty(n, dtype=[(f.name, '<' + PF[f.datatype][2]) for f in fields])
    for f in fields:
        packed[f.name] = raw[f.name]
    header = ('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n'
              f"FIELDS {' '.join(f.name for f in fields)}\n"
              f"SIZE {' '.join(str(PF[f.datatype][1]) for f in fields)}\n"
              f"TYPE {' '.join(PF[f.datatype][0] for f in fields)}\n"
              f"COUNT {' '.join('1' for _ in fields)}\n"
              f'WIDTH {msg.width}\nHEIGHT {msg.height}\nVIEWPOINT 0 0 0 1 0 0 0\n'
              f'POINTS {n}\nDATA binary\n')
    with open(path, 'wb') as fp:
        fp.write(header.encode('ascii'))
        fp.write(packed.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('out')
    ap.add_argument('--imu', default='/imu/data')
    ap.add_argument('--lidar', default='/ouster/points')
    ap.add_argument('--storage', default='sqlite3')
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, 'scans'), exist_ok=True)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id=args.storage),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    n_imu = n_scan = 0
    with open(os.path.join(args.out, 'imu.csv'), 'w') as fi, \
            open(os.path.join(args.out, 'scans.csv'), 'w') as fs:
        fi.write('stamp_ns,ax,ay,az,gx,gy,gz\n')
        fs.write('stamp_ns,file\n')
        while reader.has_next():
            topic, data, _ = reader.read_next()
            if topic == args.imu:
                m = deserialize_message(data, get_message(types[topic]))
                a, g = m.linear_acceleration, m.angular_velocity
                fi.write(f'{stamp_ns(m.header.stamp)},{a.x!r},{a.y!r},{a.z!r},'
                         f'{g.x!r},{g.y!r},{g.z!r}\n')
                n_imu += 1
            elif topic == args.lidar:
                m = deserialize_message(data, get_message(types[topic]))
                ns = stamp_ns(m.header.stamp)
                rel = f'scans/{ns}.pcd'
                write_pcd(os.path.join(args.out, rel), m)
                fs.write(f'{ns},{rel}\n')
                n_scan += 1
    print(f'exported {n_imu} IMU samples and {n_scan} scans to {args.out}')


if __name__ == '__main__':
    main()
