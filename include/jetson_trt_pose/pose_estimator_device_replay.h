#pragma once
#include "jetson_trt_pose/pose_estimator.h"

#include <rosbag/bag.h>

using std::string;

struct PoseEstimatorParamsReplay : PoseEstimatorParams{
  bool replay_sync = false;
  double replay_rate_factor = 1.0;
  double replay_t0 = 0.0;
  string bag_file;
  string bag_file_cam_topic = "/camera/color/image_raw";
};

class PoseEstimatorReplay : public PoseEstimator{
private:
  bool replay_sync;
  double replay_rate_factor;
  double replay_t0;
  string bag_file_cam_topic;

  //ros::Publisher publisher_image_orig;
  rosbag::Bag bag;
  cv::Mat orig_image;
  ros::Time orig_image_timestamp;
  bool orig_updated, first_frame;
  std::mutex orig_image_mutex;
  std::condition_variable inference_cv;
  std::thread replay_thread;

  void run_replay();

  void fb_skeleton_replaysync_cb(const person_msgs::Person2DOcclusionListConstPtr& humans){fb_skeletons = *humans;}

public:
  PoseEstimatorReplay(ros::NodeHandle& nh_, const PoseEstimatorParamsReplay& params = PoseEstimatorParamsReplay());
  ~PoseEstimatorReplay();

  void start_replay();
  void run_inference_replay();
};
