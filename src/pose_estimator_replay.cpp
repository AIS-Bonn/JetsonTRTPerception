#include "jetson_trt_pose/pose_estimator_device_replay.h"

#include <memory>
#include <fstream>
#include <chrono>
#include <iostream>
#include <numeric>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <edgetpu_segmentation_msgs/DetectionList.h>
#include <std_msgs/Bool.h>
#include <rosbag/view.h>

using std::cout;
using std::endl;

PoseEstimatorReplay::PoseEstimatorReplay(ros::NodeHandle& nh_, const PoseEstimatorParamsReplay& params)
  : PoseEstimator(nh_, params), replay_sync(params.replay_sync), replay_rate_factor(params.replay_rate_factor), replay_t0(params.replay_t0), bag_file_cam_topic(params.bag_file_cam_topic),
    orig_updated(false)
{
  if(feedback_type == "" && replay_sync) // activate feedback callback just for syncing.
    fb_skel_sub = nh_.subscribe("/" + camera + "/skel_pred", 1, &PoseEstimatorReplay::fb_skeleton_replaysync_cb, this, ros::TransportHints().tcpNoDelay());

  bag.open(params.bag_file);
  //publisher_image_orig = nh.advertise<sensor_msgs::Image>("/" + camera + "/color/image_orig", 1);

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
}

PoseEstimatorReplay::~PoseEstimatorReplay(){
  bag.close();

  if(replay_thread.joinable()){
    replay_thread.join();
  }
}

void PoseEstimatorReplay::start_replay(){
  replay_thread = std::thread(&PoseEstimatorReplay::run_replay, this);
}

void PoseEstimatorReplay::run_replay(){
  rosbag::View view(bag, rosbag::TopicQuery(bag_file_cam_topic));
  ROS_INFO("Reading image topic from bag: %s, got %u messages", bag_file_cam_topic.c_str(), view.size());
  const ros::Duration replay_sync_delay(0.005);

  // Update parameters
  double wfb_add, wfb_mul;
  if(nh.getParam("/wfb_add", wfb_add)){
    if(std::abs(wfb_add - w_feedback) > 1e-6){
      cout << "updating wfb_add to: " << wfb_add << ".     " << endl;
      w_feedback = wfb_add;
    }
  }
  if(nh.getParam("/wfb_mul", wfb_mul)){
    if(std::abs(wfb_mul - w_mult) > 1e-6){
      cout << "updating wfb_mul to: " << wfb_mul << ".     " << endl;
      w_mult = wfb_mul;
    }
  }

  rosbag::View::const_iterator msg_itr = view.begin();
  if(replay_t0 <= 0.0) {
    if(!replay_sync)
      ROS_WARN("Using time of first message as bagfile t0. Might not be synchronized between cameras.");
    sensor_msgs::ImageConstPtr img_msg = msg_itr->instantiate<sensor_msgs::Image>();
    if(img_msg != nullptr){
      replay_t0 = img_msg->header.stamp.toSec();
    }
  }

  ros::Time stamp_t0(replay_t0);
  int skip_cnt = 0;
  for(; msg_itr != view.end(); ++msg_itr) {
    sensor_msgs::ImageConstPtr img_msg = msg_itr->instantiate<sensor_msgs::Image>();
    if(img_msg == nullptr){
      ROS_ERROR("invalid message type: %s!", msg_itr->getDataType().c_str());
      continue;
    }

    if(img_msg->header.stamp.toSec() >= stamp_t0.toSec())
      break;
    ++skip_cnt;
  }
  ROS_INFO("skipped %d messages before bagfile t0", skip_cnt);

  string pipeline_start_topic = "/start_camera_pipeline";
  ROS_INFO("Waiting for message on topic \"%s\" to start replay..", pipeline_start_topic.c_str());

  const auto& msg = ros::topic::waitForMessage<std_msgs::Bool>(pipeline_start_topic, nh, ros::Duration(300.0));
  if(msg == nullptr){
    ROS_WARN("starting replay after timeout!");
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> real_t0 = std::chrono::high_resolution_clock::now();
  cout << "Starting replay, t = " << ros::Time::now() << "s, bagfile t0 = " << stamp_t0 << "s." << endl;

  for(; msg_itr != view.end(); ++msg_itr) {
    sensor_msgs::ImageConstPtr img_msg = msg_itr->instantiate<sensor_msgs::Image>();
    if(img_msg == nullptr){
      ROS_ERROR("invalid message type: %s!", msg_itr->getDataType().c_str());
      continue;
    }

    auto img_cv = cv_bridge::toCvShare(img_msg, "rgb8"); // rgb image for detector.

    cv::Mat input_image;
    if(flip)
      cv::flip(img_cv->image, input_image, -1);
    else
      input_image = img_cv->image;

    if(!replay_sync){
      ros::Duration stamp_delta_t = img_msg->header.stamp - stamp_t0; // Time that should have passed since replay start
      auto real_delta_t = std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() - real_t0).count(); // Time that has already passed since replay start
      double sleep_delta_t = std::max(0.0, stamp_delta_t.toSec() / replay_rate_factor - real_delta_t / (double)1e6 - 0.001); // Sleep for remaining time (minus 1ms offset to account for additional delays)
      //cout << "sleeping for " << sleep_delta_t * 1000.0 << "ms." << endl;
      ros::Duration(sleep_delta_t).sleep(); // Sleep for remaining time: time between messages, scaled by replay rate factor minus the time already passed.
    }

    {
      std::lock_guard<std::mutex> lck (orig_image_mutex);
      input_image.copyTo(orig_image);
      orig_image_timestamp = img_msg->header.stamp;
      orig_updated = true;
    }

    inference_cv.notify_one();

    if(!ros::ok())
      break;

    if(replay_sync){
      replay_sync_delay.sleep();
      std::unique_lock<std::mutex> lck(orig_image_mutex);
      inference_cv.wait(lck, [this]{return !orig_updated;});
      lck.unlock();
    }
  }

  ROS_INFO("Done reading from bag.");
  ros::shutdown();

  {
    std::lock_guard<std::mutex> lck (orig_image_mutex);
    orig_updated = true;
  }
  inference_cv.notify_one();
}

void PoseEstimatorReplay::run_inference_replay(){
  std::chrono::time_point<std::chrono::high_resolution_clock> t_prev;
  static ros::Time t_det, t_print; //, t_msg;
  std::vector<Detection> dets_person, dets_obj;
  std::vector<int> assignment(top_k, -1); // Allocate for max number of detections

  ros::Rate fb_sync_rate(100);
  int fb_sync_max_its = static_cast<int>(std::ceil(max_fb_delay / fb_sync_rate.expectedCycleTime().toSec()));
  bool fb_first_frame = true;

  while(ros::ok()){
    cv::Mat input_image;
    std::unique_lock<std::mutex> lck(orig_image_mutex);
    inference_cv.wait(lck, [this]{return orig_updated;});

    orig_image.copyTo(input_image);
    ros::Time curr_timestamp = orig_image_timestamp;
    orig_updated = false;

    lck.unlock();

    image_size = cv::Size(input_image.cols, input_image.rows);

    if(replay_sync)
      inference_cv.notify_one();

    if(!ros::ok() || input_image.cols == 0 || input_image.rows == 0){
      if(input_image.cols == 0 || input_image.rows == 0)
        ROS_ERROR("Got empty image!");
      break;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto delta_t = std::chrono::duration_cast<std::chrono::microseconds>( t0 - t_prev ).count();
    t_prev = t0;

//    if((curr_timestamp - t_msg).toSec() > 0.15){
//      ROS_WARN("Missed a frame: delta_t = %fs (should be 0.10s)", (curr_timestamp - t_msg).toSec());
//    }
//    t_msg = curr_timestamp;

    ros::Duration delta_t_last_det = curr_timestamp - t_det;
    bool detector_run = false;
    if (delta_t_last_det.toSec() > max_delta_t_det || delta_t_last_det.toSec() < 0.0 || dets_person.empty()){
      t_det = curr_timestamp;
      detector_run = true;

      cv::Size input_size(engine.det_input_tensor_shape()[2], engine.det_input_tensor_shape()[1]);
      if(image_size != input_size){
        //### Fixed aspect ratio resizing of the whole image to the input resolution ###
        double scale = std::min((double)input_size.width / image_size.width, (double)input_size.height / image_size.height);
        cv::Mat image_resized;
        if(scale != 1.0)
          cv::resize(input_image, image_resized, cv::Size(0,0), scale, scale, cv::INTER_LINEAR);
        else
          image_resized = input_image;
        cv::Size new_size(image_resized.cols, image_resized.rows);
        int x0 = (input_size.width - new_size.width) / 2;
        int y0 = (input_size.height - new_size.height) / 2;
        cv::Rect RoI(x0, y0, new_size.width, new_size.height);
        cv::Mat input_image_resized(input_size.height, input_size.width, CV_8UC3, cv::Scalar(0,0,0));
        image_resized.copyTo(input_image_resized(RoI));

        const auto t0_det = std::chrono::high_resolution_clock::now();

        engine.process_det_input(input_image_resized);
        engine.invoke_det();
        engine.get_det_result(dets_person, dets_obj, det_thresh, top_k, input_size, x0, y0, (float)scale);

        const auto t1_det = std::chrono::high_resolution_clock::now();
        const auto duration_inf_det = std::chrono::duration_cast<std::chrono::microseconds>( t1_det - t0_det ).count();
        runtime_det += duration_inf_det / 1000.;
        ++runtime_cnt_det;
      }
      else{
        const auto t0_det = std::chrono::high_resolution_clock::now();

        engine.process_det_input(input_image);
        engine.invoke_det();
        engine.get_det_result(dets_person, dets_obj, det_thresh, top_k, image_size);

        const auto t1_det = std::chrono::high_resolution_clock::now();
        const auto duration_inf_det = std::chrono::duration_cast<std::chrono::microseconds>( t1_det - t0_det ).count();
        runtime_det += duration_inf_det / 1000.;
        ++runtime_cnt_det;
      }
    }

    bool rcv_skel_fb = false, skel_fb_associated = false;
    int n_det_person = dets_person.size();
    long fb_sync_wait = 0;
    if((feedback_type == "skeleton" && (w_feedback > 0.0 || w_mult > 0.0)) || replay_sync){ // && !dets_person.empty() )

      ros::spinOnce(); // receive feedback skeleton ((?) maybe do this asyncronously)

      if(replay_sync && !fb_first_frame){ // wait for feedback with one frame delay to arrive
        auto wait_t0 = std::chrono::high_resolution_clock::now();
        int fb_wait_its = 0;
        while(((curr_timestamp - fb_skeletons.header.stamp).toSec() > 0.11 || (curr_timestamp - fb_skeletons.header.stamp).toSec() < 0.0) && ros::ok()){
          ros::spinOnce();
          fb_sync_rate.sleep();
          if(++fb_wait_its > fb_sync_max_its){ // timeout
            ROS_WARN("Timeout in feedback sync!");
            break;
          }
        }
        auto wait_t1 = std::chrono::high_resolution_clock::now();
        fb_sync_wait = std::chrono::duration_cast<std::chrono::microseconds>( wait_t1 - wait_t0 ).count();
      }

      if(feedback_type == "skeleton" && (w_feedback > 0.0 || w_mult > 0.0)){ // && !dets_person.empty()
        rcv_skel_fb = true;
        if((curr_timestamp - fb_skeletons.header.stamp).toSec() < max_fb_delay && (curr_timestamp - fb_skeletons.header.stamp).toSec() >= 0.0){ // associate only, when feedback is valid (recent enough..)
          associate_feedback(assignment.data(), dets_person);
          skel_fb_associated = true;
        }
      }
    }
    fb_first_frame = false;

    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<Human> humans, humans_det;
    humans.reserve(n_det_person);
    humans_det.reserve(n_det_person);
    std::vector<Detection::BBox> crop_infos(n_det_person), crop_infos_debug;
    std::vector<cv::Mat> heatmaps_vec, heatmaps_fused_vec;
    heatmaps_vec.reserve(n_det_person);
    if(pub_debug_heatmap){
      heatmaps_fused_vec.reserve(n_det_person);
      crop_infos_debug.reserve(n_det_person);
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
        humans_fb_assignment = assignment; // feedback skeleton association.
        humans_fb = fb_skeletons.persons;
        humans_fb_updated = true;
      }
      humans_fb_cv.notify_one();
    }

    std::vector<cv::Mat> img_crops(n_det_person);
    for(int i = 0; i < n_det_person; ++i)
      crop_bbox(img_crops[i], input_image, crop_infos[i]);

    auto start_inf = std::chrono::high_resolution_clock::now();
    engine.infer_batch_pose(heatmaps_vec, img_crops);

    auto start_wait_fb_hm = std::chrono::high_resolution_clock::now();
    bool fb_hms_rendered = false;
    if(use_skel_fb && ros::ok()){
      fb_delta_t = (curr_timestamp - fb_skeletons.header.stamp).toSec();
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
          ros::Duration delta_t_last_print = curr_timestamp - t_print;
          if(delta_t_last_print.toSec() > 1.0 || delta_t_last_print.toSec() < 0.0){ // Throttle to once per second.
            t_print = curr_timestamp;
            cout << "No feedback of type \"" << feedback_type << "\" received! delta_t: " << (curr_timestamp - fb_skeletons.header.stamp).toSec() << "s.     " << endl;
          }
        }

        heatmaps_fused = heatmaps_vec[i];
      }

      Human human = parse_skeleton(heatmaps_fused, crop_infos[i]);

      if(human.num_valid_kps > 0 && human.conf > det_thresh){
        humans_det.push_back(human);
        //human2det_idx.push_back(i);

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

    auto end_pp = std::chrono::high_resolution_clock::now();

    person_msgs::Person2DOcclusionList persons_msg;
    humans_to_msg(persons_msg, humans, curr_timestamp, skel_fb_associated);
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

      publisher_obj.publish(dets_obj_msg);
    }

    if(pub_debug_heatmap && publisher_debug_img.getNumSubscribers() > 0 && !dets_person.empty() && !humans_det.empty()){ // draw debug heatmap for the first detection
      std::unique_lock<std::mutex> lck(debug_hm_mutex, std::try_to_lock);
      if(lck){
        heatmaps_vec[0].copyTo(debug_heatmaps);
        if(fb_hm_used[0])
          heatmaps_feedback[0].copyTo(debug_heatmaps_feedback);
        else
          debug_heatmaps_feedback = cv::Mat();
  //      heatmaps_fused_vec[0].copyTo(debug_heatmaps_feedback);
        debug_human = humans_det[0];
        debug_crop_info = crop_infos_debug[0];
        debug_hm_timestamp = curr_timestamp;
        debug_hm_updated = true;
        lck.unlock();
        debug_hm_cv.notify_one();
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

  if(replay_sync){
    {
      std::lock_guard<std::mutex> lck (orig_image_mutex);
      orig_updated = false;
    }
    inference_cv.notify_one();
  }
}
