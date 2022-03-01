#include <ros/ros.h>
#include <sys/stat.h>
#include "jetson_trt_pose/pose_estimator_rosimg.h"

bool check_file(const char *file) {
  struct stat buf;
  if (stat(file, &buf) != 0) {
    //std::cerr << file << " does not exist" << std::endl;
    return false;
  }
  return true;
}

using std::string;

int main(int argc, char** argv) {
  // Initialize ROS
  ros::init (argc, argv, "edgetpu_cpp_pose_rosimg");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  PoseEstimatorParams params;

  nh_private.param<bool>("debug_heatmap", params.debug_heatmap, false);
  nh_private.param<bool>("pub_raw", params.pub_raw, false);
  nh_private.param<float>("det_thresh", params.det_thresh, 0.5f);
  nh_private.param<int>("top_k", params.top_k, 10);
  nh_private.param<double>("delta_t_det", params.max_delta_t_det, 1.0);
  nh_private.param<string>("camera", params.camera, "camera");
  nh_private.param<string>("feedback_type", params.feedback_type, "");
  nh_private.param<double>("fb_min_iou", params.fb_min_iou, 0.40);
  //nh_private.param<bool>("swap_hm_ch", params.do_swap_hm_channels, false);
  nh_private.param<bool>("flip", params.flip, false);

  string precision;
  nh_private.param<string>("precision", precision, "fp16");
  std::transform(precision.begin(), precision.end(), precision.begin(), ::tolower);
  params.precision = 0;
  if(precision.find("fp16") != string::npos)
    params.precision |= 0x1;
  else if (precision.find("int8") != string::npos)
    params.precision |= 0x2;
  if (precision.find("dla") != string::npos)
    params.precision |= 0x4;

  nh_private.param<string>("calib_data_dir", params.calib_data_dir, "/home/jetson/datasets/coco/val2017");
  nh_private.param<string>("model_segm", params.model_segm, "");
  nh_private.param<string>("model_thermal", params.model_det_thermal, "");
  nh_private.param<string>("calib_data_dir_thermal", params.calib_data_dir_thermal, "/home/jetson/datasets/flir_lepton/val");

  if(!nh_private.getParam("model_det", params.model_det)){
    ROS_ERROR("Parameter \"model_det\" giving path to detection model is required!");
    return -1;
  }

  if(!nh_private.getParam("model_pose", params.model_pose)){
    ROS_ERROR("Parameter \"model_pose\" giving path to pose estimation model is required!");
    return -1;
  }

  if(!check_file(params.model_det.c_str())){
    ROS_ERROR("Detection model file \"%s\" does not exist! exiting!", params.model_det.c_str());
    return -2;
  }
  if(!check_file(params.model_pose.c_str())){
    ROS_ERROR("Pose model file \"%s\" does not exist! exiting!", params.model_pose.c_str());
    return -2;
  }
  if(!check_file(params.model_segm.c_str())){
    ROS_WARN("Segmentation model file \"%s\" does not exist. Not using segmentation.", params.model_segm.c_str());
    params.model_segm = "";
  }

  if(!check_file(params.model_det_thermal.c_str())){
    ROS_WARN("Thermal dectection model file \"%s\" does not exist. Not using thermal detection.", params.model_det_thermal.c_str());
    params.model_det_thermal = "";
  }

  PoseEstimatorRosimg estimator(nh, params);

  ros::spin();

  std::cout << std::endl << "Average runtime per frame: " << estimator.runtimes[0] / estimator.runtimes_cnt[0] << "ms." << std::endl;
  for(int i = 1; i < PoseEstimator::max_num_timings; ++i){
    if(estimator.runtimes_cnt[i] > 0){
      std::cout << "Average runtime per frame (" << i << " detections): " << estimator.runtimes[i] / estimator.runtimes_cnt[i] << "ms." << std::endl;
    }
  }

  if(estimator.fb_delays_cnt[0] > 0)
    std::cout << std::endl << "Average feedback-delay: " << estimator.feedback_delays[0] / estimator.fb_delays_cnt[0] << "s." << std::endl;
  for(int i = 1; i < PoseEstimator::max_num_timings; ++i){
    if(estimator.fb_delays_cnt[i] > 0){
      std::cout << "Average feedback-delay (" << i << " detections): " << estimator.feedback_delays[i] / estimator.fb_delays_cnt[i] << "s." << std::endl;
    }
  }

  if(estimator.runtime_cnt_det > 0){
    std::cout << std::endl << "Average detector inference time: " << estimator.runtime_det / estimator.runtime_cnt_det << "ms." << std::endl;
  }

  if(estimator.runtime_cnt_det_thermal > 0){
    std::cout << std::endl << "Average thermal detector inference time: " << estimator.runtime_det_thermal / estimator.runtime_cnt_det_thermal << "ms." << std::endl;
  }
}
