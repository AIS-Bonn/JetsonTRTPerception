#include "jetson_trt_pose/pose_estimator_rosimg.h"

#include <memory>
#include <fstream>
#include <chrono>
#include <iostream>
#include <numeric>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <edgetpu_segmentation_msgs/DetectionList.h>

using std::cout;
using std::endl;

PoseEstimatorRosimg::PoseEstimatorRosimg(ros::NodeHandle& nh_, const PoseEstimatorParams& params)
  : PoseEstimator (nh_, params) //, sync(sub_img, sub_depth, 3)
{
  sub_img = nh.subscribe("/" + camera + "/color/image_raw", 1, &PoseEstimatorRosimg::img_cb, this);
  sub_depth = nh.subscribe("/" + camera + "/depth/image_rect_raw", 1, &PoseEstimatorRosimg::depth_cb, this);
//  sub_img.subscribe(nh, "/" + camera + "/color/image_raw", 1);
//  sub_depth.subscribe(nh, "/" + camera + "/depth/image_rect_raw", 1);
//  sync.registerCallback(&PoseEstimatorRosimg::img_cb, this);

  const auto& output_shapes_det = engine.det_output_tensor_shapes();
  const auto& input_shape_det = engine.det_input_tensor_shape();
  cout << "Detection model:" << endl << "  Input shape: (";
  for (const auto& dim : input_shape_det)
    cout << dim << ", ";
  cout << ")" << endl << "  Output shapes: " << endl;
  for (int i = 0 ; i < output_shapes_det.size(); ++i) {
    cout << "    (" << i << "): (";
    for(const auto& dim : output_shapes_det[i])
      cout << dim << ", ";
    cout << ")" << endl;
  }

  if(thermal_engine_ptr){
    sub_thermal = nh.subscribe("/" + camera + "/lepton/image", 1, &PoseEstimatorRosimg::thermal_img_cb, this);
    //pub_thermal_debug = nh.advertise<sensor_msgs::Image>("thermal/image_rescaled", 1);
    //clahe = cv::createCLAHE();

    const auto& output_shapes_det = thermal_engine_ptr->det_output_tensor_shapes();
    const auto& input_shape_det = thermal_engine_ptr->det_input_tensor_shape();
    cout << "Detection model:" << endl << "  Input shape: (";
    for (const auto& dim : input_shape_det)
      cout << dim << ", ";
    cout << ")" << endl << "  Output shapes: " << endl;
    for (int i = 0 ; i < output_shapes_det.size(); ++i) {
      cout << "    (" << i << "): (";
      for(const auto& dim : output_shapes_det[i])
        cout << dim << ", ";
      cout << ")" << endl;
    }
  }

  if(pub_raw_heatmap)
    publisher_raw_hm_info = nh.advertise<sensor_msgs::CameraInfo>("/" + camera + "/raw_output_info", 1);
}

PoseEstimatorRosimg::~PoseEstimatorRosimg(){
  sub_img.shutdown();
  sub_depth.shutdown();

  if(thermal_engine_ptr){
    sub_thermal.shutdown();
  }
}

void PoseEstimatorRosimg::depth_cb(const sensor_msgs::ImageConstPtr &depth_msg){
  auto img_depth = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
  depth_input_ts = depth_msg->header.stamp;
  if(m_align_depth){
    {
      std::lock_guard<std::mutex> lck(depth_mutex);
      img_depth->image.copyTo(depth_input_img);
      depth_input_updated = true;
    }
    depth_cv.notify_one();
  }
  else{
    img_depth->image.copyTo(depth_aligned);
  }
}

void PoseEstimatorRosimg::img_cb(const sensor_msgs::ImageConstPtr &img_msg){
  static std::chrono::time_point<std::chrono::high_resolution_clock> t_prev;
  static ros::Time t_det, t_param_upd, t_raw_pub, t_print;
  static const cv::Size input_size_det(engine.det_input_tensor_shape()[2], engine.det_input_tensor_shape()[1]);
  static const cv::Rect roi_det(cv::Point2i(0,0), input_size_det);
  static const bool segm_engine_loaded = (engine.get_segm_engine().get() != nullptr);
  static std::vector<Detection> dets_person, dets_obj;
  static std::vector<Human> prev_humans;
  static std::vector<int> assignment(top_k, -1); // Allocate for max number of detections

  auto t0 = std::chrono::high_resolution_clock::now();
  auto delta_t = std::chrono::duration_cast<std::chrono::microseconds>( t0 - t_prev ).count();
  t_prev = t0;

  auto img_cv = cv_bridge::toCvShare(img_msg, "rgb8"); // rgb image for detector.

  image_size = cv::Size(img_msg->width, img_msg->height);
  cv::Mat orig_image;
  if(flip)
    cv::flip(img_cv->image, orig_image, -1);
  else
    orig_image = img_cv->image;

  ros::Duration delta_t_last_det = img_msg->header.stamp - t_det;
  bool detector_run = false;
  cv::Mat input_image_det_segm;
  if (delta_t_last_det.toSec() > max_delta_t_det || delta_t_last_det.toSec() < 0.0 || dets_person.empty()){
    int x0 = 0, y0 = 0;
    double scale = 1.0;

    if(image_size != input_size_det){
      //### Fixed aspect ratio resizing of the whole image to the input resolution ###
      scale = std::min((double)input_size_det.width / image_size.width, (double)input_size_det.height / image_size.height);
      cv::Mat image_resized;
      if(scale != 1.0)
        cv::resize(orig_image, image_resized, cv::Size(0,0), scale, scale, cv::INTER_LINEAR);
      else
        image_resized = orig_image;
      cv::Size new_size(image_resized.cols, image_resized.rows);
      x0 = (input_size_det.width - new_size.width) / 2;
      y0 = (input_size_det.height - new_size.height) / 2;
      cv::Rect RoI(x0, y0, new_size.width, new_size.height);
      input_image_det_segm = cv::Mat(input_size_det, CV_8UC3, cv::Scalar(0,0,0));
      image_resized.copyTo(input_image_det_segm(RoI));
    }
    else {
      input_image_det_segm = orig_image;
    }

    const auto t0_det = std::chrono::high_resolution_clock::now();

    engine.process_det_input(input_image_det_segm);
    engine.invoke_det();
    engine.get_det_result(dets_person, dets_obj, det_thresh, top_k, input_size_det, x0, y0, (float)scale);

    const auto t1_det = std::chrono::high_resolution_clock::now();
    const auto duration_inf_det = std::chrono::duration_cast<std::chrono::microseconds>( t1_det - t0_det ).count();
    runtime_det += duration_inf_det / 1000.;
    ++runtime_cnt_det;

    t_det = img_msg->header.stamp;
    detector_run = true;
  }

  if(thermal_dets_updated){
    thermal_dets_updated = false;
    for (const auto& det_thermal : thermal_dets_transformed_to_color){
      bool add_to_dets = true;
      for(const auto& det : dets_person){
        if(bbox_iou(det_thermal.bbox, det.bbox) > 0.1f){ // only use thermal detection, if there is no rgb detection with significant overlap..
          add_to_dets = false;
          break;
        }
      }
      if(add_to_dets){
        dets_person.push_back(det_thermal);
        cout << "Added thermal det to person detections at: (" << det_thermal.bbox.xmin << ", " << det_thermal.bbox.ymin << "), (" << det_thermal.bbox.xmax << ", " << det_thermal.bbox.ymax << ")" << endl;
      }
    }
  }

  bool rcv_skel_fb = false, skel_fb_associated = false;
  const int n_det_person = dets_person.size();
  if(feedback_type == "skeleton" && (w_feedback > 0.0 || w_mult > 0.0)){ //&& !dets_person.empty()
    rcv_skel_fb = true;
    if((img_msg->header.stamp - fb_skeletons.header.stamp).toSec() < max_fb_delay && (img_msg->header.stamp - fb_skeletons.header.stamp).toSec() >= 0.0){ // associate only, when feedback is valid (recent enough..)
      associate_feedback(assignment.data(), dets_person);
      skel_fb_associated = true;
    }
  }
  else if(feedback_type == "skeleton"){ // enable switching feedback on again after setting w_feedback to 0; TODO: let an async. background process handle the parameter update ?!
    ros::Duration delta_t_last_upd = img_msg->header.stamp - t_param_upd;
    if(delta_t_last_upd.toSec() > 5.0){
      t_param_upd = img_msg->header.stamp;
      double wfb_add, wfb_mul;
      if(nh.getParam("/wfb_add", wfb_add)){
        if(std::abs(wfb_add - w_feedback) > 1e-3){
          cout << "updating wfb_add to: " << wfb_add << ".     " << endl;
          {
            std::lock_guard<std::mutex> lck_hm (fb_hm_mutex);
            w_feedback = wfb_add;
          }
        }
      }
      if(nh.getParam("/wfb_mul", wfb_mul)){
        if(std::abs(wfb_mul - w_mult) > 1e-3){
          cout << "updating wfb_mul to: " << wfb_mul << ".     " << endl;
          {
            std::lock_guard<std::mutex> lck_hm (fb_hm_mutex);
            w_mult = wfb_mul;
          }
        }
      }
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();

  std::vector<Human> humans, humans_det;
  humans.reserve(n_det_person);
  humans_det.reserve(n_det_person);
  std::vector<int> human2det_idx; // not all detections / heatmaps may result in a valid human with a minimum number of keypoints. -> Save the mapping from validated humans to original detections.
  human2det_idx.reserve(n_det_person);
  std::vector<Detection::BBox> crop_infos(n_det_person);
  std::vector<cv::Mat> heatmaps_vec, heatmaps_fused_vec;
  heatmaps_vec.reserve(n_det_person);
  if(pub_debug_heatmap || pub_raw_heatmap){
    heatmaps_fused_vec.reserve(n_det_person);
  }

  double fb_delta_t = 0.0;
  float fb_delta_t_pred = 0.0f;

  for(int i = 0; i < n_det_person; ++i)
    crop_infos[i] = calc_crop_info(dets_person[i]);

  bool use_skel_fb = false;
  if(skel_fb_associated){
    use_skel_fb = true;
    {
      std::lock_guard<std::mutex> lck(humans_fb_mutex);
      humans_fb_cropinfos = crop_infos;
      humans_fb_img_size = image_size;
      humans_fb_assignment = assignment; // feedback skeleton association.
      humans_fb = fb_skeletons.persons;
      humans_fb_updated = true;
    }
    humans_fb_cv.notify_one();
  }

  std::vector<cv::Mat> img_crops(n_det_person);
  for(int i = 0; i < n_det_person; ++i)
    crop_bbox(img_crops[i], orig_image, crop_infos[i]);

  auto start_inf = std::chrono::high_resolution_clock::now();
  engine.infer_batch_pose(heatmaps_vec, img_crops);

  auto start_wait_fb_hm = std::chrono::high_resolution_clock::now();

  bool segm_run = false;
  if(segm_engine_loaded && detector_run){
    std::unique_lock<std::mutex> lck(segm_mutex, std::try_to_lock);
    if(lck){
      segm_run = true;
      input_image_det_segm.copyTo(segm_input_img(roi_det));
      segm_input_header = img_msg->header;
      segm_input_updated = true;
      lck.unlock();
      segm_cv.notify_one();
    }
  }

  bool fb_hms_rendered = false;
  if(use_skel_fb && ros::ok()){
    fb_delta_t = (img_msg->header.stamp - fb_skeletons.header.stamp).toSec();
    fb_delta_t_pred = fb_skeletons.fb_delay;
    if(fb_delta_t > 0.17)
      cout << "Feedback skeletons have unexpectedly large delay of " << fb_delta_t << "s. (predicted: " << fb_delta_t_pred << "s)" << endl;

    std::unique_lock<std::mutex> lck(fb_hm_mutex);
    fb_hm_cv.wait(lck, [this]{return fb_hm_updated;});
    fb_hm_updated = false;
    fb_hms_rendered = true;
  }

  auto start_pp = std::chrono::high_resolution_clock::now();
  bool fb_hm_used_once = false;
  std::vector<bool> fb_hm_used(n_det_person, false);
  for(int i = 0; i < n_det_person; ++i){
    cv::Mat heatmaps_fused;
    if(fb_hms_rendered && assignment[i] >= 0){
      {
        std::lock_guard<std::mutex> lck(fb_hm_mutex);
//        if (do_swap_hm_channels)
//          swap_hm_channels(heatmaps_vec[i], hm_fb_kps_centers[i], engine.get_kps_symmetry());

        heatmaps_fused = heatmaps_vec[i] + (w_feedback / (1 - w_feedback - w_mult)) * heatmaps_feedback[i] + (w_mult / (1 - w_feedback - w_mult)) * heatmaps_vec[i].mul(heatmaps_feedback[i]);
      }

      fb_hm_used[i] = true;
      fb_hm_used_once = true;
    }
    else {
      if(rcv_skel_fb && !skel_fb_associated){
        ros::Duration delta_t_last_print = img_msg->header.stamp - t_print;
        if(delta_t_last_print.toSec() > 1.0 || delta_t_last_print.toSec() < 0.0){ // Throttle to once per second.
          t_print = img_msg->header.stamp;
          cout << "No feedback of type \"" << feedback_type << "\" received! delta_t: " << (img_msg->header.stamp - fb_skeletons.header.stamp).toSec() << "s.     " << endl;
        }
      }

      heatmaps_fused = heatmaps_vec[i];
    }

    Human human = parse_skeleton(heatmaps_fused, crop_infos[i]);

    if(human.num_valid_kps > 0 && human.conf > det_thresh){
      humans_det.push_back(human);
      human2det_idx.push_back(i);

      if(pub_debug_heatmap || pub_raw_heatmap){
        heatmaps_fused_vec.push_back(heatmaps_fused); //.clone() ?
      }
    }

    if(use_skel_fb && assignment[i] >= 0 && fb_skeletons.persons[assignment[i]].n_occluded > 0){
      human_check_occlusion(human, fb_skeletons.persons[assignment[i]]); //check for occluded joints with associated feedback skeleton.
    }
    if(human.num_valid_kps > 0 && (human.conf > det_thresh || human.num_occluded_kps > 0)){
      humans.push_back(human);
    }
  }

  nms_humans(humans, nms_threshold);

  if(ros::ok())
  {
    //auto start_wait_depth_align = std::chrono::high_resolution_clock::now();
    std::lock_guard<std::mutex> lck(depth_mutex);
    if(segm_run)
      depth_aligned.copyTo(segm_depth);
    const double depth_delta_t = (depth_input_ts - img_msg->header.stamp).toSec();
    if(depth_delta_t <= 0.17)
      humans_kps_depth_est(humans);
    //long wait_depth_align = std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() - start_wait_depth_align ).count();
    //cout << "waited " << wait_depth_align << "us for depth alignment: " << depth_aligned.size << endl;
  }

  auto end_pp = std::chrono::high_resolution_clock::now();

  person_msgs::Person2DOcclusionList persons_msg;
  humans_to_msg(persons_msg, humans, img_msg->header.stamp, skel_fb_associated);

  if(fb_hm_used_once){
    persons_msg.fb_delay = static_cast<float>(fb_delta_t);
  }
  else{
    persons_msg.fb_delay = -1.0f;
  }
  publisher_skeleton.publish(persons_msg);

  if(detector_run){
    const int n_det_obj = dets_obj.size();
    edgetpu_segmentation_msgs::DetectionList dets_obj_msg;
    dets_obj_msg.header = persons_msg.header;
    dets_obj_msg.detections.reserve(n_det_obj + n_det_person);
    for (int i = 0; i < n_det_obj; ++i) {
      edgetpu_segmentation_msgs::Detection det_msg;
      det_msg.score = dets_obj[i].score;
      det_msg.label = static_cast<uint8_t>(dets_obj[i].label);
      det_msg.bbox.xmin = static_cast<double>(flip ? image_size.width - dets_obj[i].bbox.xmax : dets_obj[i].bbox.xmin);
      det_msg.bbox.xmax = static_cast<double>(flip ? image_size.width - dets_obj[i].bbox.xmin : dets_obj[i].bbox.xmax);
      det_msg.bbox.ymin = static_cast<double>(flip ? image_size.height - dets_obj[i].bbox.ymax : dets_obj[i].bbox.ymin);
      det_msg.bbox.ymax = static_cast<double>(flip ? image_size.height - dets_obj[i].bbox.ymin : dets_obj[i].bbox.ymax);
      dets_obj_msg.detections.push_back(det_msg);
    }

    for (const auto& human : humans) {
      if(human.num_valid_kps - human.num_occluded_kps > 0 && human.conf > det_thresh){
        edgetpu_segmentation_msgs::Detection det_msg;
        det_msg.score = std::min(0.99f, human.conf); //TODO: scale conf correctly
        det_msg.label = static_cast<uint8_t>(1); // 1 = person class idx
        det_msg.bbox.xmin = static_cast<double>(flip ? image_size.width - human.bbox.xmax : human.bbox.xmin);
        det_msg.bbox.xmax = static_cast<double>(flip ? image_size.width - human.bbox.xmin : human.bbox.xmax);
        det_msg.bbox.ymin = static_cast<double>(flip ? image_size.height - human.bbox.ymax : human.bbox.ymin);
        det_msg.bbox.ymax = static_cast<double>(flip ? image_size.height - human.bbox.ymin : human.bbox.ymax);
        dets_obj_msg.detections.push_back(det_msg);
      }
    }

//    for(const auto& thermal_det: thermal_dets_transformed_to_color) {
//      edgetpu_segmentation_msgs::Detection det_msg;
//      det_msg.score = thermal_det.score;
//      det_msg.label = static_cast<uint8_t>(2); // class 2 (bicycle) for visualizatzion // thermal_det.label
//      det_msg.bbox.xmin = static_cast<double>(flip ? image_size.width - thermal_det.bbox.xmax : thermal_det.bbox.xmin);
//      det_msg.bbox.xmax = static_cast<double>(flip ? image_size.width - thermal_det.bbox.xmin : thermal_det.bbox.xmax);
//      det_msg.bbox.ymin = static_cast<double>(flip ? image_size.height - thermal_det.bbox.ymax : thermal_det.bbox.ymin);
//      det_msg.bbox.ymax = static_cast<double>(flip ? image_size.height - thermal_det.bbox.ymin : thermal_det.bbox.ymax);
//      dets_obj_msg.detections.push_back(det_msg);
//    }

    publisher_obj.publish(dets_obj_msg);
  }

  if(!humans_det.empty() && ((pub_raw_heatmap && publisher_raw_hm.getNumSubscribers() > 0) || (pub_debug_heatmap && publisher_debug_img.getNumSubscribers() > 0))){
    std::vector<int> human_assignment(prev_humans.size());
    int idx_to_pub = -1;
    if(!prev_humans.empty()){
      associate_humans(human_assignment.data(), prev_humans, humans_det);

      for(int prev_idx = 0; idx_to_pub < 0; ++prev_idx)
        idx_to_pub = human_assignment[prev_idx];

      update_prev_humans(prev_humans, humans_det, human_assignment);
    }
    else{
      idx_to_pub = 0;
      prev_humans = humans_det;
    }

    static ros::Time last_stamp_printed;
    if(last_stamp_printed != pred_heatmaps.header.stamp){
    	cout << "Curr pred: delta_t: " << (img_msg->header.stamp - pred_heatmaps.header.stamp).toSec() << "s.     " << endl;
    	last_stamp_printed = pred_heatmaps.header.stamp;
    }
//    cout << "Debug publishing human idx: " << idx_to_pub << " (assignment: [";
//    for(const auto& assign_idx: human_assignment)
//      cout << assign_idx << ", ";
//    cout << "] )" << endl;

    if(pub_raw_heatmap && publisher_raw_hm.getNumSubscribers() > 0){
      ros::Duration delta_t_last_raw_pub = img_msg->header.stamp - t_raw_pub;
      if(delta_t_last_raw_pub.toSec() > 0.090){ // publish raw heatmap at max. 10Hz.
        t_raw_pub = img_msg->header.stamp;
        sensor_msgs::ImagePtr msg_img = cv_bridge::CvImage(persons_msg.header, "32FC" + std::to_string(heatmaps_fused_vec[idx_to_pub].channels()), heatmaps_fused_vec[idx_to_pub]).toImageMsg();
        publisher_raw_hm.publish(msg_img);

        sensor_msgs::CameraInfo msg_info;
        msg_info.header = persons_msg.header;
        msg_info.width = heatmaps_fused_vec[idx_to_pub].cols;
        msg_info.height = heatmaps_fused_vec[idx_to_pub].rows;
        msg_info.roi.x_offset = std::lround(crop_infos[human2det_idx[idx_to_pub]].xmin);
        msg_info.roi.y_offset = std::lround(crop_infos[human2det_idx[idx_to_pub]].ymin);
        msg_info.roi.width = std::lround(crop_infos[human2det_idx[idx_to_pub]].width());
        msg_info.roi.height = std::lround(crop_infos[human2det_idx[idx_to_pub]].height());
        publisher_raw_hm_info.publish(msg_info);
      }
    }

    if(pub_debug_heatmap && publisher_debug_img.getNumSubscribers() > 0){ // draw debug heatmap for the first detection
      std::unique_lock<std::mutex> lck(debug_hm_mutex, std::try_to_lock);
      if(lck){
        heatmaps_vec[human2det_idx[idx_to_pub]].copyTo(debug_heatmaps);
        if(fb_hm_used[human2det_idx[idx_to_pub]])
          heatmaps_feedback[human2det_idx[idx_to_pub]].copyTo(debug_heatmaps_feedback);
        else
          debug_heatmaps_feedback = cv::Mat();
//      heatmaps_fused_vec[id_to_pub].copyTo(debug_heatmaps_feedback);
        debug_human = humans_det[idx_to_pub];
        debug_crop_info = crop_infos[human2det_idx[idx_to_pub]];
        debug_hm_timestamp = img_msg->header.stamp;
        debug_hm_updated = true;
        lck.unlock();
        debug_hm_cv.notify_one();
      }
    }
  }

  humans_to_dets(dets_person, humans);

  auto t5 = std::chrono::high_resolution_clock::now();
  auto duration_det = std::chrono::duration_cast<std::chrono::microseconds>( t1 - t0 ).count();
  long duration_crop = std::chrono::duration_cast<std::chrono::microseconds>( start_inf - t1 ).count();
  long duration_inf = std::chrono::duration_cast<std::chrono::microseconds>( start_wait_fb_hm - start_inf).count();
  long duration_wait_fb = std::chrono::duration_cast<std::chrono::microseconds>( start_pp - start_wait_fb_hm).count();
  long duration_pp = std::chrono::duration_cast<std::chrono::microseconds>( end_pp - start_pp ).count();
  auto duration_all = std::chrono::duration_cast<std::chrono::microseconds>( t5 - t0 ).count();
  runtimes[0] += duration_all / 1000.;
  ++runtimes_cnt[0];
  if(n_det_person > 0 && n_det_person < max_num_timings){
    runtimes[n_det_person] += duration_all / 1000.;
    ++runtimes_cnt[n_det_person];
  }
  if(fb_hm_used_once){
    feedback_delays[0] += fb_delta_t;
    ++fb_delays_cnt[0];
    if(n_det_person > 0 && n_det_person < max_num_timings){
      feedback_delays[n_det_person] += fb_delta_t;
      ++fb_delays_cnt[n_det_person];
    }
  }

  cout << "Inference: delta_t: " << delta_t / 1000.0 << "ms, fb_delta_t: " << fb_delta_t * 1000.0 << "ms (pred: " << fb_delta_t_pred * 1000.0f << "ms), det time: " << duration_det / 1000.0 << "ms (" << n_det_person << " dets), crop time: " << duration_crop / 1000.0 << "ms, inf time: " << duration_inf / 1000.0 << "ms, wait fb: " << duration_wait_fb / 1000.0 << "ms, pp time: " << duration_pp / 1000.0 << "ms, total duration: " << duration_all / 1000.0 <<"ms.\r";
  cout.flush();
}

void PoseEstimatorRosimg::thermal_img_cb(const sensor_msgs::ImageConstPtr &img_msg){
  static std::chrono::time_point<std::chrono::high_resolution_clock> t_prev;
  static const cv::Size input_size_det(thermal_engine_ptr->det_input_tensor_shape()[2], thermal_engine_ptr->det_input_tensor_shape()[1]);
  static std::vector<Detection> dets_person;
  static const Eigen::Matrix3f K_thermal_inv = K_thermal.inverse().cast<float>();

  static constexpr int depth_env_delta = 5;
  static constexpr int depth_env_size = 2 * depth_env_delta + 1;

  auto t0 = std::chrono::high_resolution_clock::now();
  auto delta_t = std::chrono::duration_cast<std::chrono::microseconds>( t0 - t_prev ).count();
  t_prev = t0;

  cv_bridge::CvImageConstPtr img_cv;
  bool thermal_16bit = false;
  if(img_msg->encoding.find("8") != string::npos)
    img_cv = cv_bridge::toCvShare(img_msg, "mono8"); // grayscale image for thermal detector.
  else if(img_msg->encoding.find("16") != string::npos){
    img_cv = cv_bridge::toCvShare(img_msg, "mono16");
    thermal_16bit = true;
  }
  else {
    ROS_ERROR("Unknown encoding %s. Aborting!", img_msg->encoding.c_str());
    return;
  }

  cv::Mat orig_image;
  if(flip)
    cv::flip(img_cv->image, orig_image, -1);
  else
    orig_image = img_cv->image;

  cv::Mat orig_image_8bit;
  if(thermal_16bit){
    const int bottom = std::lround((0. + 273.15)*100); // 0°C
    const int top    = std::lround((60. + 273.15)*100); // 60°C
//    cv::Mat orig_image_normalized;
//    clahe->apply(orig_image, orig_image_normalized);
//    double top, bottom;
//    cv::minMaxLoc(orig_image_normalized, &bottom, &top);
    const double alpha = 255. / (top - bottom);
    const double beta = -1. * bottom * alpha;
//    orig_image_normalized.convertTo(orig_image_8bit, CV_8UC1, alpha, beta);
    orig_image.convertTo(orig_image_8bit, CV_8UC1, alpha, beta);
//    cv_bridge::CvImage img_debug(img_msg->header, "mono8", orig_image_8bit);
//    pub_thermal_debug.publish(img_debug.toImageMsg());
  }
  else
    orig_image_8bit = orig_image;

  cv::Mat input_image_det;
  cv::Size thermal_img_size(img_msg->width, img_msg->height);
  int x0 = 0, y0 = 0;
  double scale = 1.0;

  if(thermal_img_size != input_size_det){
    //### Fixed aspect ratio resizing of the whole image to the input resolution ###
    scale = std::min((double)input_size_det.width / thermal_img_size.width, (double)input_size_det.height / thermal_img_size.height);
    cv::Mat image_resized;
    if(scale != 1.0)
      cv::resize(orig_image_8bit, image_resized, cv::Size(0,0), scale, scale, cv::INTER_LINEAR);
    else
      image_resized = orig_image_8bit;
    cv::Size new_size(image_resized.cols, image_resized.rows);
    x0 = (input_size_det.width - new_size.width) / 2;
    y0 = (input_size_det.height - new_size.height) / 2;
    cv::Rect RoI(x0, y0, new_size.width, new_size.height);
    input_image_det = cv::Mat(input_size_det, CV_8UC1, cv::Scalar(0));
    image_resized.copyTo(input_image_det(RoI));
  }
  else {
    input_image_det = orig_image_8bit;
  }

  auto t1 = std::chrono::high_resolution_clock::now();

  thermal_engine_ptr->process_det_input(input_image_det);
  thermal_engine_ptr->invoke_det();
  thermal_engine_ptr->get_det_result(dets_person, det_thresh, top_k, input_size_det, x0, y0, (float)scale);

  auto t2 = std::chrono::high_resolution_clock::now();

  const int n_det = dets_person.size();
  edgetpu_segmentation_msgs::DetectionList dets_thermal_msg;
  dets_thermal_msg.header.stamp = img_msg->header.stamp;
  dets_thermal_msg.header.frame_id = img_msg->header.frame_id;
  dets_thermal_msg.detections.reserve(n_det);
  for (int i = 0; i < n_det; ++i) {
    edgetpu_segmentation_msgs::Detection det_msg;
    det_msg.score = dets_person[i].score;
    det_msg.label = static_cast<uint8_t>(dets_person[i].label);
    det_msg.bbox.xmin = static_cast<double>(flip ? thermal_img_size.width - dets_person[i].bbox.xmax : dets_person[i].bbox.xmin);
    det_msg.bbox.xmax = static_cast<double>(flip ? thermal_img_size.width - dets_person[i].bbox.xmin : dets_person[i].bbox.xmax);
    det_msg.bbox.ymin = static_cast<double>(flip ? thermal_img_size.height - dets_person[i].bbox.ymax : dets_person[i].bbox.ymin);
    det_msg.bbox.ymax = static_cast<double>(flip ? thermal_img_size.height - dets_person[i].bbox.ymin : dets_person[i].bbox.ymax);
    dets_thermal_msg.detections.push_back(det_msg);
  }

  publisher_det_thermal.publish(dets_thermal_msg);

  thermal_dets_transformed_to_color.clear();
  thermal_dets_transformed_to_color.reserve(n_det);
  {
    std::lock_guard<std::mutex> lck(depth_thermal_mutex);
    for(const auto& det: dets_person){
      std::vector<uint16_t> depth_candidates; // use the median depth value of a small environment around the detected joint.
      depth_candidates.reserve(depth_env_size * depth_env_size);

      int u = std::lround(flip ? thermal_img_size.width - det.bbox.cx() : det.bbox.cx());
      int v = std::lround(flip ? thermal_img_size.height - det.bbox.cy() : det.bbox.cy());
      for (int uu = u - depth_env_delta; uu <= u + depth_env_delta; ++uu) {
        for (int vv = v - depth_env_delta; vv <= v + depth_env_delta; ++vv) {
          if(uu < 0 || vv < 0 || uu >= depth_aligned_thermal.cols || vv >= depth_aligned_thermal.rows)
            continue;
          if(depth_aligned_thermal.at<uint16_t>(vv,uu) > m_min_depth && depth_aligned_thermal.at<uint16_t>(vv,uu) < m_max_depth)
            depth_candidates.push_back(depth_aligned_thermal.at<uint16_t>(vv,uu));
        }
      }

      if(depth_candidates.empty()){
  //          ROS_WARN("No Valid Depth Measurements for thermal bbox %d!", kp_idx);
        continue; // no valid depth info found for joint.
      }

      std::sort(depth_candidates.begin(), depth_candidates.end());
      const float z_bbox = (float)(depth_candidates[depth_candidates.size() / 2]) / m_depth_scale; // median depth

      Eigen::Matrix<float,3,2> bbox_pts_homog;
      bbox_pts_homog << (flip ? thermal_img_size.width - det.bbox.xmax : det.bbox.xmin), (flip ? thermal_img_size.width - det.bbox.xmin : det.bbox.xmax), (flip ? thermal_img_size.height - det.bbox.ymax : det.bbox.ymin), (flip ? thermal_img_size.height - det.bbox.ymin : det.bbox.ymax), 1.f, 1.f;
      Eigen::Matrix<float,3,2> bbox_rays = K_thermal_inv * bbox_pts_homog;
      Eigen::Matrix<float,3,2> bbox_pts_3D = bbox_rays.array() * z_bbox;
      Eigen::Matrix<float,3,2> bbox_pts_transformed_homog = P_thermal_to_color * bbox_pts_3D.colwise().homogeneous();
      Eigen::Matrix2f bbox_pts_transformed = bbox_pts_transformed_homog.colwise().hnormalized(); //tl and br bbox points

      //flip bbox, so that it will be coherent with rgb detections...
      thermal_dets_transformed_to_color.push_back(Detection{0, det.label, det.score, Detection::BBox{flip ? image_size.height - bbox_pts_transformed.col(1).y(): bbox_pts_transformed.col(0).y(),
                                                                                                     flip ? image_size.width - bbox_pts_transformed.col(1).x(): bbox_pts_transformed.col(0).x(),
                                                                                                     flip ? image_size.height - bbox_pts_transformed.col(0).y(): bbox_pts_transformed.col(1).y(),
                                                                                                     flip ? image_size.width - bbox_pts_transformed.col(0).x() : bbox_pts_transformed.col(1).x()},
                                                            std::vector<Detection::Keypoint>()});
    }
  }
  thermal_dets_updated = true;

  auto t3 = std::chrono::high_resolution_clock::now();
  auto duration_pre = std::chrono::duration_cast<std::chrono::microseconds>( t1 - t0 ).count();
  long duration_inf = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1).count();
  long duration_post = std::chrono::duration_cast<std::chrono::microseconds>( t3 - t2).count();
  auto duration_all = std::chrono::duration_cast<std::chrono::microseconds>( t3 - t0 ).count();

  runtime_det_thermal += duration_inf / 1000.;
  ++runtime_cnt_det_thermal;

  cout << "  Thermal Inference: delta_t: " << delta_t / 1000.0 << "ms, preproc time: " << duration_pre / 1000.0 << "ms, inf time: " << duration_inf / 1000.0 << "ms (" << n_det << " dets), pp time: " << duration_post / 1000.0 << "ms, total duration: " << duration_all / 1000.0 <<"ms.                                        \r";
  cout.flush();
}

