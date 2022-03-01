#pragma once
#include "jetson_trt_pose/pose_estimator.h"

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
//#include <opencv2/imgproc.hpp>

using std::string;

class PoseEstimatorRosimg : public PoseEstimator{
private:
  ros::Subscriber sub_thermal, sub_img, sub_depth;
  //message_filters::Subscriber<sensor_msgs::Image> sub_img, sub_depth;
  //message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync;
  ros::Publisher publisher_info_orig, publisher_raw_hm_info; //, pub_thermal_debug;
  //cv::Ptr<cv::CLAHE> clahe;

  void depth_cb(const sensor_msgs::ImageConstPtr& depth_msg);
  void img_cb(const sensor_msgs::ImageConstPtr& img_msg);
  void thermal_img_cb(const sensor_msgs::ImageConstPtr& img_msg); // TODO: run thermal callback in a different thread ?!

public:
  PoseEstimatorRosimg(ros::NodeHandle& nh_, const PoseEstimatorParams& params = PoseEstimatorParams());
  ~PoseEstimatorRosimg();
};
