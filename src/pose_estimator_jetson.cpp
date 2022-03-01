#include "jetson_trt_pose/pose_estimator.h"
#include "jetson_trt_pose/Hungarian.h"

#include <memory>
#include <chrono>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <eigen3/Eigen/Eigenvalues>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <edgetpu_segmentation_msgs/DetectionList.h>

using std::cout;
using std::endl;

namespace  {
int sign(float val){
  return (0.0f < val) - (val < 0.0f);
}
}

constexpr float PoseEstimator::oks_kappas[17];
constexpr float PoseEstimator::m_depth_scale;
constexpr int PoseEstimator::m_depth_env_delta;
constexpr int PoseEstimator::m_depth_env_size;
constexpr uint16_t PoseEstimator::m_min_depth;
constexpr uint16_t PoseEstimator::m_max_depth;
typedef Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, ADE20K_INDOOR::NUM_CLASSES, Eigen::RowMajor>> MapTypeLogitsConst;
typedef Eigen::Map<Eigen::Matrix<uint8_t, Eigen::Dynamic, 3, Eigen::RowMajor>> MapTypeRGB;
typedef Eigen::Matrix<float, 1, ADE20K_INDOOR::NUM_CLASSES, Eigen::RowMajor> VecLogits;

PoseEstimator::PoseEstimator(ros::NodeHandle& nh_, const PoseEstimatorParams& params)
    : pub_debug_heatmap(params.debug_heatmap), pub_raw_heatmap(params.pub_raw), camera(params.camera), det_thresh(params.det_thresh), top_k(params.top_k),
      max_delta_t_det(params.max_delta_t_det), prev_crop_factor(params.prev_crop_factor), nms_threshold(params.nms_threshold), feedback_type(params.feedback_type), max_fb_delay(params.max_fb_delay),
      hm_fb_sigma(params.sigma), sigma_max(params.sigma_max), fb_min_iou(params.fb_min_iou), do_swap_hm_channels(params.do_swap_hm_channels), min_dist_swap{params.min_dist_swap[0], params.min_dist_swap[1]}, max_env_swap(params.max_env_swap), flip(params.flip), hm_cov_thresh(params.hm_cov_thresh),
      nh(nh_), engine(params.model_pose, params.model_det, params.model_segm, params.h36m, params.precision, params.calib_data_dir),
      fb_hm_updated(false), humans_fb_updated(false), debug_hm_updated(false), runtimes(max_num_timings, 0.0), feedback_delays(max_num_timings, 0.0), runtimes_cnt(max_num_timings, 0), fb_delays_cnt(max_num_timings, 0), runtime_det(0.), runtime_det_thermal(0.), runtime_cnt_det(0), runtime_cnt_det_thermal(0)
{
    publisher_skeleton = nh.advertise<person_msgs::Person2DOcclusionList>("/" + camera + "/human_joints", 1);
    publisher_obj = nh.advertise<edgetpu_segmentation_msgs::DetectionList>("/" + camera + "/dets_obj", 1);

    if(params.model_segm != ""){
      publisher_segm = nh.advertise<sensor_msgs::Image>("/" + camera + "/logits", 1);
      publisher_segm_overlay = nh.advertise<sensor_msgs::Image>("/" + camera + "/segmentation_overlay", 1);
      segm_input_img = cv::Mat(engine.get_segm_input_dims()[1], engine.get_segm_input_dims()[2], CV_8UC3, cv::Scalar(0, 0, 0));
      ROS_INFO_STREAM("Segmentation input image size: " << segm_input_img.cols << " x " << segm_input_img.rows);
      segm_input_updated = false;
      segmentation_inference_thread = std::thread(&PoseEstimator::segmentation_inference_callback, this);
    }

    if(params.model_det_thermal != ""){
      publisher_det_thermal = nh.advertise<edgetpu_segmentation_msgs::DetectionList>("/" + camera + "/dets_thermal", 1);
      ROS_INFO("Creating thermal engine...");
      string calib_data_dir;
      if(params.model_det_thermal.find("_heq") != string::npos)
        calib_data_dir = params.calib_data_dir_thermal + "/normalized_heq";
      else
        calib_data_dir = params.calib_data_dir_thermal + "/normalized";

      thermal_engine_ptr = std::unique_ptr<ThermalDetEngine>(new ThermalDetEngine(params.model_det_thermal, params.precision, calib_data_dir));
    }

    if(pub_debug_heatmap)
      publisher_debug_img = nh.advertise<sensor_msgs::Image>("/" + camera + "/heatmap", 1);
    if(pub_raw_heatmap)
      publisher_raw_hm = nh.advertise<sensor_msgs::Image>("/" + camera + "/raw_output", 1);

    pred_hm_sub = nh.subscribe("/" + camera + "/pred_result", 1, &PoseEstimator::pred_hm_cb, this);

    if(feedback_type.find("skel") != string::npos){
      feedback_type = "skeleton";
      ROS_INFO("Using feedback type \"skeleton\"");
      if(!nh.getParam("/wfb_add", w_feedback)){
        w_feedback = 0.056256; //0.014 //0.1813; //0.680625;
        ROS_WARN("Parameter \"/wfb_add\" not found, defaulting to w_feedback = %.4f.", w_feedback);
      }
      else {
        ROS_INFO("wfb_add = %.4f", w_feedback);
      }

      if(!nh.getParam("/wfb_mul", w_mult)){
        w_mult = 0.876241; //0.639 //0.7062;
        ROS_WARN("Parameter \"/wfb_mul\" not found, defaulting to wfb_mul = %.4f.", w_mult);
      }
      else {
        ROS_INFO("wfb_mul = %.4f", w_mult);
      }

      fb_skel_sub = nh.subscribe("/" + camera + "/skel_pred", 1, &PoseEstimator::fb_skeleton_cb, this, ros::TransportHints().tcpNoDelay());
    }
    else {
      feedback_type = "";
      w_feedback = 0.0;
      w_mult = 0.0;
      ROS_INFO("Not using any feedback.");
    }

    const auto& output_shapes_pose = engine.pose_output_tensor_shapes();
    const auto& input_shape_pose = engine.pose_input_tensor_shape();
    cout << "Pose model:" << endl << "  Input shape: (";
    for (const auto& dim : input_shape_pose)
      cout << dim << ", ";
    cout << ")" << endl << "  Output shapes: " << endl;
    for (int i = 0 ; i < output_shapes_pose.size(); ++i) {
      cout << "    (" << i << "): (";
      for(const auto& dim : output_shapes_pose[i])
        cout << dim << ", ";
      cout << ")" << endl;
    }

    set_cov_kernels();
    noise_offset = Eigen::Matrix2f::Identity() * hm_fb_sigma * hm_fb_sigma;

    heatmap_size = cv::Size(output_shapes_pose[0][2], output_shapes_pose[0][1]);
    heatmap_channels = output_shapes_pose[0][3];
    cout << "heatmap size: " << heatmap_size << ", channels: " << heatmap_channels << endl;
    assert(heatmap_channels == Human::num_kps);

    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer);

    // get intrinsics
    string depth_info_topic = "/" + camera + "/depth/camera_info";
    string image_info_topic = "/" + camera + "/color/camera_info";
    string thermal_info_topic = "/" + camera + "/lepton/camera_info";
    ROS_INFO("Waiting for camera info message on topic \"%s\"...", depth_info_topic.c_str());
    const auto& caminfo_depth_msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(depth_info_topic, nh);
    const auto& caminfo_color_msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(image_info_topic, nh);
    const auto& caminfo_thermal_msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(thermal_info_topic, nh);
    if(caminfo_depth_msg == nullptr || caminfo_color_msg == nullptr || caminfo_thermal_msg == nullptr){
      ROS_ERROR("no camera info message received!");
      return;
    }

    string depth_frame = caminfo_depth_msg->header.frame_id;
    string color_frame = caminfo_color_msg->header.frame_id;
    string thermal_frame = caminfo_thermal_msg->header.frame_id;
    ROS_INFO("depth frame: %s, color frame: %s.", depth_frame.c_str(), color_frame.c_str());
    if(depth_frame != color_frame){
      ROS_INFO("aligning depth to color");
      m_align_depth = true;
    }
    else {
      ROS_INFO("depth and color are in same frame. No alignment needed.");
      m_align_depth = false;
    }

    depth_size = cv::Size(caminfo_depth_msg->width, caminfo_depth_msg->height);
    cv::Size color_size = cv::Size(caminfo_color_msg->width, caminfo_color_msg->height);
    depth_color_scale = (float) caminfo_depth_msg->width / caminfo_color_msg->width;
    thermal_size = cv::Size(caminfo_thermal_msg->width, caminfo_thermal_msg->height);
    thermal_dets_updated = false;
    cout << "depth size: " << depth_size << ", color size: " << color_size << " --> scale: " << depth_color_scale  << ", thermal size: " << thermal_size << endl;
    K_depth << caminfo_depth_msg->K[0], 0.0, caminfo_depth_msg->K[2], 0.0, caminfo_depth_msg->K[4], caminfo_depth_msg->K[5], 0.0, 0.0, 1.0;
    K_color_scaled << caminfo_color_msg->K[0] * depth_color_scale, 0.0, caminfo_color_msg->K[2] * depth_color_scale, 0.0, caminfo_color_msg->K[4] * depth_color_scale, caminfo_color_msg->K[5] * depth_color_scale, 0.0, 0.0, 1.0;
    K_thermal << caminfo_thermal_msg->K[0], 0.0, caminfo_thermal_msg->K[2], 0.0, caminfo_thermal_msg->K[4], caminfo_thermal_msg->K[5], 0.0, 0.0, 1.0;
    Eigen::Matrix3d K_color;
    K_color << caminfo_color_msg->K[0], 0.0, caminfo_color_msg->K[2], 0.0, caminfo_color_msg->K[4], caminfo_color_msg->K[5], 0.0, 0.0, 1.0;

    cout << "Received depth camera intrinsics: size: (" << depth_size.width << ", " << depth_size.height << ")" << endl << K_depth << endl;
    cout << "Received color camera intrinsics (scale: " << depth_color_scale << "): size: (" << color_size.width << ", " << color_size.height << ")" << endl << K_color_scaled << endl;
    cout << "Received thermal camera intrinsics: size: (" << thermal_size.width << ", " << thermal_size.height << ")" << endl << K_thermal << endl;

    m_pixels_depth = Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor>(2, depth_size.width * depth_size.height); // shape = 2 x HW
//    g_pixels_depth_br = Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor>(2, g_depth_size.width * g_depth_size.height); // shape = 2 x HW
    for (int u = 0; u < depth_size.width; ++u) {
      for (int v = 0; v < depth_size.height; ++v) {
        m_pixels_depth.col(v * depth_size.width + u) = Eigen::Vector2f(u, v); // - 0.5f
//        g_pixels_depth_br.col(v * g_depth_size.width + u) = Eigen::Vector2f(u + 0.5f, v + 0.5f);
      }
    }

    m_depth_pixel_rays = K_depth.cast<float>().inverse() * m_pixels_depth.colwise().homogeneous(); // shape 3 x HW // .homogenous(): (u,v) -> (u, v, 1)
//    g_depth_pixel_rays_br = K_depth.cast<float>().inverse() * g_pixels_depth_br.colwise().homogeneous(); // shape 3 x HW // .homogenous(): (u,v) -> (u, v, 1)

    m_depth_to_color = Eigen::Affine3d::Identity();
    if(m_align_depth){
      geometry_msgs::TransformStamped depth_to_color, depth_to_thermal, thermal_to_color;
      try {
          depth_to_color = tfBuffer.lookupTransform(color_frame, depth_frame, ros::Time(0), ros::Duration(1.0));
          depth_to_thermal = tfBuffer.lookupTransform(thermal_frame, depth_frame, ros::Time(0), ros::Duration(1.0));
          thermal_to_color = tfBuffer.lookupTransform(color_frame, thermal_frame, ros::Time(0), ros::Duration(1.0));
      }
      catch(tf2::TransformException &ex) {
          ROS_ERROR("%s",ex.what());
          return;
      }
      m_depth_to_color = tf2::transformToEigen(depth_to_color);
      m_depth_to_thermal = tf2::transformToEigen(depth_to_thermal);
      P_thermal_to_color = (K_color * tf2::transformToEigen(thermal_to_color).affine()).cast<float>();
      cout << "Depth to color extrinsics: translation: " << m_depth_to_color.translation().transpose() << endl << "rotation: " << endl << m_depth_to_color.linear() << endl;
      cout << "Depth to thermal extrinsics: translation: " << m_depth_to_thermal.translation().transpose() << endl << "rotation: " << endl << m_depth_to_thermal.linear() << endl;
      cout << "thermal to color Projection: " << endl << P_thermal_to_color << endl;
      aligned_depth_debug_pub = nh.advertise<sensor_msgs::Image>("/" + camera + "/debug_aligned_depth", 1);
    }

    if(m_align_depth){
      //depth_aligned = cv::Mat(depth_size, CV_16UC1, cv::Scalar(0));
      depth_alignment_thread = std::thread(&PoseEstimator::depth_alignment_cb, this);
    }

    if(feedback_type == "skeleton")
      feedback_hm_thread = std::thread(&PoseEstimator::feedback_hm_cb, this);

    if(pub_debug_heatmap)
      debug_hm_thread = std::thread(&PoseEstimator::debug_heatmap_cb, this);
}

PoseEstimator::~PoseEstimator()
{
    publisher_skeleton.shutdown();
    publisher_obj.shutdown();

    if(thermal_engine_ptr){
      publisher_det_thermal.shutdown();
    }

    if(pub_debug_heatmap)
      publisher_debug_img.shutdown();

    if(feedback_type == "skeleton")
      fb_skel_sub.shutdown();
    pred_hm_sub.shutdown();

    if(feedback_hm_thread.joinable()){
      humans_fb_cv.notify_one();
      feedback_hm_thread.join();
    }

    if(segmentation_inference_thread.joinable()){
      segm_cv.notify_one();
      segmentation_inference_thread.join();

      publisher_segm.shutdown();
      publisher_segm_overlay.shutdown();
    }

    if(depth_alignment_thread.joinable()){
      depth_cv.notify_one();
      depth_alignment_thread.join();
      aligned_depth_debug_pub.shutdown();
    }

    if(debug_hm_thread.joinable()){
      debug_hm_cv.notify_one();
      debug_hm_thread.join();
    }
}

void PoseEstimator::fb_skeleton_cb(const person_msgs::Person2DOcclusionListConstPtr &humans){
  fb_skeletons = *humans;
}

void PoseEstimator::pred_hm_cb(const sensor_msgs::ImageConstPtr &pred_result){
  pred_heatmaps = *pred_result;
}

void PoseEstimator::depth_alignment_cb(){
  std_msgs::Header header;
  header.frame_id = camera + "_depth_optical_frame";
  cv::Mat depth_thermal(thermal_size, CV_16UC1);
  while(ros::ok()){
    std::unique_lock<std::mutex> lck(depth_mutex);
    depth_cv.wait(lck, [this]{return depth_input_updated || !ros::ok();});
    if(!ros::ok())
      break;

    align_depth(depth_thermal);
    header.stamp = depth_input_ts;

    depth_input_updated = false;
    lck.unlock();

    {
      std::lock_guard<std::mutex> lck_thermal(depth_thermal_mutex);
      depth_thermal.copyTo(depth_aligned_thermal);
    }

    if(aligned_depth_debug_pub.getNumSubscribers() > 0){
      sensor_msgs::ImagePtr msg_depth = cv_bridge::CvImage(header, "mono16", depth_aligned).toImageMsg();
      aligned_depth_debug_pub.publish(msg_depth);
    }
  }
}

void PoseEstimator::feedback_hm_cb(){
  double wfb_add = 0.0, wfb_mul = 0.0;
  bool update_w_fb = false;
  while(ros::ok()){
    std::unique_lock<std::mutex> lck(humans_fb_mutex);
    humans_fb_cv.wait(lck, [this]{return humans_fb_updated || !ros::ok();});
    if(!ros::ok())
      break;

    humans_fb_updated = false;
    int n_det = humans_fb_cropinfos.size();
    {
      std::lock_guard<std::mutex> lck_hm (fb_hm_mutex);
      heatmaps_feedback.resize(n_det);
      for (int i = 0; i < n_det; ++i) {
        if(humans_fb_assignment[i] >= 0){
          if(do_swap_hm_channels)
            render_heatmaps(heatmaps_feedback[i], humans_fb[humans_fb_assignment[i]], humans_fb_cropinfos[i], &hm_fb_kps_centers[i]);
          else
            render_heatmaps(heatmaps_feedback[i], humans_fb[humans_fb_assignment[i]], humans_fb_cropinfos[i]);
        }
      }

      if(update_w_fb){
        w_feedback = wfb_add;
        w_mult = wfb_mul;
        update_w_fb = false;
      }
      fb_hm_updated = true;
    }
    fb_hm_cv.notify_one();
    lck.unlock();

    if(nh.getParam("/wfb_add", wfb_add))
      if(std::abs(wfb_add - w_feedback) > 1e-3){
        cout << "updating wfb_add to: " << wfb_add << ".     " << endl;
        update_w_fb = true;
    }
    if(nh.getParam("/wfb_mul", wfb_mul))
      if(std::abs(wfb_mul - w_mult) > 1e-3){
        cout << "updating wfb_mul to: " << wfb_mul << ".     " << endl;
        update_w_fb = true;
    }
  }
}

void PoseEstimator::segmentation_inference_callback(){
  auto engine_segm = engine.get_segm_engine();
  samplesCommon::BufferManager buffers(engine_segm);

  auto context = PoseDetEngine::SampleUniquePtr<nvinfer1::IExecutionContext>(engine_segm->createExecutionContext());
  if (!context)
  {
      ROS_ERROR("Could not create segmentation inference context!");
      return;
  }

  ROS_INFO_STREAM("Segmentation context created");
  auto input_dims = context->getBindingDimensions(0);
  ROS_INFO_STREAM("Context input dimensions: [" << input_dims.d[0] << ", " << input_dims.d[1] << ", " << input_dims.d[2] << ", " << input_dims.d[3] << "]");
  auto output_dims = context->getBindingDimensions(1);
  ROS_INFO_STREAM("Context output dimensions: [" << output_dims.d[0] << ", " << output_dims.d[1] << ", " << output_dims.d[2] << ", " << output_dims.d[3] << "]");

  cv::Mat logits;
  PoseDetEngine::def_colormap();

  double segm_inf_time_avg = 0.0;
  long segm_inf_cnt = -10;

  while(ros::ok()){
    std::unique_lock<std::mutex> lck(segm_mutex);
    segm_cv.wait(lck, [this]{return segm_input_updated || !ros::ok();});
    if(!ros::ok())
      break;

    segm_input_updated = false;

    auto t11 = std::chrono::high_resolution_clock::now();

    engine.process_segm_input(segm_input_img, buffers);
    std_msgs::Header header = segm_input_header;
    //lck.unlock();
    engine.invoke_segm(buffers, context);
    engine.get_segmentation(logits, buffers);

    auto t22 = std::chrono::high_resolution_clock::now();
    auto duration_inf = std::chrono::duration_cast<std::chrono::microseconds>( t22 - t11 ).count();
    if(segm_inf_cnt >= 0)
      segm_inf_time_avg = ((duration_inf / 1000.) + segm_inf_cnt * segm_inf_time_avg) / (segm_inf_cnt + 1);
    ++segm_inf_cnt;

    //Publish log-probabilities of segmentation classes
    if(publisher_segm.getNumSubscribers() > 0){
      sensor_msgs::ImagePtr msg_logits = cv_bridge::CvImage(header, "32FC" + std::to_string(ADE20K_INDOOR::NUM_CLASSES), logits).toImageMsg();
      publisher_segm.publish(msg_logits);
    }

    //Publish segmentation overlay image
    if(publisher_segm_overlay.getNumSubscribers() > 0){
      cv::Mat logits_inputsize;
      if(segm_input_img.cols != logits.cols || segm_input_img.rows != logits.rows){
        float transform_data[6] = {(float)(segm_input_img.cols - 1) / (float)(logits.cols - 1), 0.f, 0.f , 0.f, (float)(segm_input_img.rows - 1) / (float)(logits.rows - 1), 0.f};
        cv::Mat warp_mat = cv::Mat(2, 3, CV_32FC1, (void*)transform_data);
        cv::warpAffine(logits, logits_inputsize, warp_mat, cv::Size(segm_input_img.cols, segm_input_img.rows));
      }
      else{
        logits_inputsize = logits;
      }

      const int num_px = logits_inputsize.cols * logits_inputsize.rows;
      cv::Mat segm_color(logits_inputsize.rows, logits_inputsize.cols, CV_8UC3);
      MapTypeRGB segm_color_eigen(segm_color.ptr<uint8_t>(0), num_px, 3);
      MapTypeLogitsConst logits_eigen(logits_inputsize.ptr<float>(0), num_px, ADE20K_INDOOR::NUM_CLASSES);
      for(int idx = 0; idx < num_px; ++idx){ // Loops through the pixels
        MapTypeLogitsConst::Index maxIndex;
        logits_eigen.row(idx).maxCoeff(&maxIndex);
        segm_color_eigen.row(idx) = PoseDetEngine::colormap_ade20k.row(maxIndex);
      }

      cv::Mat img_overlay;
      cv::addWeighted(segm_color, PoseDetEngine::alpha_overlay, segm_input_img, 1.0 - PoseDetEngine::alpha_overlay, 0, img_overlay);
      sensor_msgs::ImagePtr msg_overlay = cv_bridge::CvImage(header, "rgb8", img_overlay).toImageMsg();
      publisher_segm_overlay.publish(msg_overlay);
    }

    lck.unlock();

    {
      std::lock_guard<std::mutex> lck_logits(segm_logits_depth_mutex);
      logits.copyTo(segm_logits);
    }
  }

  std::cout << std::endl << "Average segmentation inference time: " << segm_inf_time_avg << "ms." << std::endl;
}

void PoseEstimator::debug_heatmap_cb(){
  cv::Mat colors_inv(17, 1, CV_32FC3);
  if(heatmap_channels != 17){
    ROS_WARN("Defined 17 colors but have %d channels.", heatmap_channels);
  }

  colors_inv.at<cv::Vec3f>(0) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(255, 0, 0) / 255.0f;
  colors_inv.at<cv::Vec3f>(1) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(255, 85, 0) / 255.0f;
  colors_inv.at<cv::Vec3f>(2) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(255, 170, 0) / 255.0f;
  colors_inv.at<cv::Vec3f>(3) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(255, 255, 0) / 255.0f;
  colors_inv.at<cv::Vec3f>(4) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(170, 255, 0) / 255.0f;
  colors_inv.at<cv::Vec3f>(5) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(85, 255, 0) / 255.0f;
  colors_inv.at<cv::Vec3f>(6) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(0, 255, 0) / 255.0f;
  colors_inv.at<cv::Vec3f>(7) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(0, 255, 85) / 255.0f;
  colors_inv.at<cv::Vec3f>(8) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(0, 255, 170) / 255.0f;
  colors_inv.at<cv::Vec3f>(9) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(0, 255, 255) / 255.0f;
  colors_inv.at<cv::Vec3f>(10) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(0, 170, 255) / 255.0f;
  colors_inv.at<cv::Vec3f>(11) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(0, 85, 255) / 255.0f;
  colors_inv.at<cv::Vec3f>(12) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(0, 0, 255) / 255.0f;
  colors_inv.at<cv::Vec3f>(13) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(50, 0, 255) / 255.0f;
  colors_inv.at<cv::Vec3f>(14) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(100, 0, 255) / 255.0f;
  colors_inv.at<cv::Vec3f>(15) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(170, 0, 255) / 255.0f;
  colors_inv.at<cv::Vec3f>(16) = cv::Vec3f(1.0, 1.0, 1.0) - cv::Vec3f(255, 0, 255) / 255.0f;

  while(ros::ok()){
    cv::Mat heatmaps_color(heatmap_size, CV_32FC3, cv::Scalar(0,0,0));
    cv::Mat hm_color_reshaped = heatmaps_color.reshape(1, heatmaps_color.rows * heatmaps_color.cols);

    std::unique_lock<std::mutex> lck(debug_hm_mutex);
    debug_hm_cv.wait(lck, [this]{return debug_hm_updated || !ros::ok();});
    if(!ros::ok())
      break;

    ros::Time curr_timestamp = debug_hm_timestamp;

    cv::Mat hm_reshaped = debug_heatmaps.reshape(1, debug_heatmaps.rows * debug_heatmaps.cols);
    cv::Mat hm_fb_reshaped = debug_heatmaps_feedback.reshape(1, debug_heatmaps_feedback.rows * debug_heatmaps_feedback.cols);

    for(int ch_idx = 0; ch_idx < heatmap_channels; ++ch_idx){
      if(hm_reshaped.rows == hm_color_reshaped.rows){
        hm_color_reshaped.col(0) = cv::max(hm_color_reshaped.col(0), hm_reshaped.col(ch_idx));
        hm_color_reshaped.col(1) = cv::max(hm_color_reshaped.col(1), hm_reshaped.col(ch_idx));
        hm_color_reshaped.col(2) = cv::max(hm_color_reshaped.col(2), hm_reshaped.col(ch_idx));
      }

      if(hm_fb_reshaped.rows == hm_color_reshaped.rows){
        hm_color_reshaped.col(0) = cv::max(hm_color_reshaped.col(0), (double)colors_inv.at<cv::Vec3f>(ch_idx)[0] * hm_fb_reshaped.col(ch_idx));
        hm_color_reshaped.col(1) = cv::max(hm_color_reshaped.col(1), (double)colors_inv.at<cv::Vec3f>(ch_idx)[1] * hm_fb_reshaped.col(ch_idx));
        hm_color_reshaped.col(2) = cv::max(hm_color_reshaped.col(2), (double)colors_inv.at<cv::Vec3f>(ch_idx)[2] * hm_fb_reshaped.col(ch_idx));
      }
    }

    debug_hm_updated = false;

    lck.unlock();

    double min, max;
    cv::minMaxLoc(heatmaps_color, &min, &max);
    if(max > min){
      cv::Mat heatmaps_color_img;
      heatmaps_color.convertTo(heatmaps_color_img, CV_8UC3, 255.0 / (max - min), -255.0 * min / (max - min));

      //if(rosimg_crop)
      //plot_covariance(heatmaps_color_img, debug_human, debug_crop_info, colors_inv);

      std_msgs::Header header;
      header.stamp = curr_timestamp;
      header.frame_id = camera + "_color_optical_frame";
      sensor_msgs::ImagePtr msg_img = cv_bridge::CvImage(header, "rgb8", cv::Scalar(255, 255, 255) - heatmaps_color_img).toImageMsg();
      publisher_debug_img.publish(msg_img);
    }
  }
}

float PoseEstimator::bbox_iou(const Detection::BBox& bb1, const Detection::BBox& bb2){
  float x0 = std::max(bb1.xmin, bb2.xmin);
  float y0 = std::max(bb1.ymin, bb2.ymin);
  float x1 = std::min(bb1.xmax, bb2.xmax);
  float y1 = std::min(bb1.ymax, bb2.ymax);

  float interArea = std::max(0.0f, x1 - x0) * std::max(0.0f, y1 - y0);

  return interArea / (bb1.area() + bb2.area() - interArea + std::numeric_limits<float>::epsilon());
}

float PoseEstimator::oks_dist(const Detection &person_prev, const person_msgs::Person2DOcclusion &person_fb){
  float area = person_prev.bbox.area();
  if(area <= 0.0f)
    return 1.0f;

//  if(person_prev.kps.size() != person_fb.keypoints.size()){
//    ROS_ERROR("Unequal number of keypoints: %zu vs %zu", person_prev.kps.size(), person_fb.keypoints.size());
//  }
  assert(person_prev.kps.size() == person_fb.keypoints.size());

  float oks = 0.0f;
  int num_kps_used = 0;

  for(int i = 0; i < person_prev.kps.size(); ++i){
    if(person_prev.kps[i].conf > 0.0f && person_fb.keypoints[i].score > 0.0f){
      float dx = person_prev.kps[i].x - person_fb.keypoints[i].x;
      float dy = person_prev.kps[i].y - person_fb.keypoints[i].y;
      oks += std::exp(-(dx*dx + dy*dy) / (2 * area * PoseEstimator::oks_kappas[i]));
      ++num_kps_used;
    }
  }

  if(num_kps_used > 0)
    return oks / num_kps_used;
  else
    return 0.0f;
}

void PoseEstimator::associate_feedback(int* assignment, const std::vector<Detection> &dets){
  const int n_det = dets.size();
  const int n_fb = fb_skeletons.persons.size();
  double* C = new double[n_det * n_fb]; // ColumnMajor Order, n_det x n_fb
  for (int j = 0; j < n_fb; ++j) {
    Detection::BBox bbox_fb{fb_skeletons.persons[j].bbox[1], fb_skeletons.persons[j].bbox[0], fb_skeletons.persons[j].bbox[3], fb_skeletons.persons[j].bbox[2]};
    if(flip){ // undo flip of color image
      float xmin_tmp = image_size.width - bbox_fb.xmax;
      float xmax_tmp = image_size.width - bbox_fb.xmin;
      float ymin_tmp = image_size.height - bbox_fb.ymax;
      float ymax_tmp = image_size.height - bbox_fb.ymin;
      bbox_fb.xmin = xmin_tmp;
      bbox_fb.xmax = xmax_tmp;
      bbox_fb.ymin = ymin_tmp;
      bbox_fb.ymax = ymax_tmp;
    }

    for (int i = 0; i < n_det; ++i) {
      double iou = static_cast<double>(bbox_iou(dets[i].bbox, bbox_fb));
      C[i + n_det * j] = 1.0 - iou;
    }
  }
  double cost;
  HungarianAlgorithm::assignmentoptimal(assignment, &cost, C, n_det, n_fb);

  std::vector<bool> fb_used(n_fb, false);
  for(int i = 0; i < n_det; ++i){
    if(assignment[i] >= 0){
      if(C[i + n_det * assignment[i]] >= 1.0 - fb_min_iou){ // Costs are larger than upper bound (1.0 - min_iou) -> don't associate // veto[i + n_det * assignment[i]]
        assignment[i] = -1;
      }
      else{
        fb_used[assignment[i]] = true;
      }
    }
  }

  delete[] C;

  fb_skeletons_occluded.clear();
  fb_skeletons_occluded.reserve(n_fb);
  for (int fb_idx = 0; fb_idx < n_fb; ++fb_idx) {
    if(!fb_used[fb_idx] && (float)fb_skeletons.persons[fb_idx].n_occluded / fb_skeletons.persons[fb_idx].n_valid > 0.5f){ // this feedback skeleton has not been assigned and is significantly occluded -> add it to seperate list.
      fb_skeletons_occluded.push_back(fb_skeletons.persons[fb_idx]);
    }
  }
}

void PoseEstimator::associate_humans(int *assignment, const std::vector<Human> &humans_prev, const std::vector<Human> &humans){
  const int n_prev = humans_prev.size();
  const int n_curr = humans.size();
  double* C = new double[n_prev * n_curr]; // ColumnMajor Order, n_prev x n_curr
  for (int j = 0; j < n_curr; ++j) {
    for (int i = 0; i < n_prev; ++i) {
      double iou = static_cast<double>(bbox_iou(humans_prev[i].bbox, humans[j].bbox));
      C[i + n_prev * j] = 1.0 - iou;
    }
  }
  double cost;
  HungarianAlgorithm::assignmentoptimal(assignment, &cost, C, n_prev, n_curr);

  delete[] C;
}

void PoseEstimator::update_prev_humans(std::vector<Human> &humans_prev, const std::vector<Human> &humans, const std::vector<int> &assignment){
  const int n_prev = humans_prev.size();
  const int n_curr = humans.size();
  if(n_prev == n_curr){
    for (int i = 0; i < n_prev; ++i) {
      humans_prev[i] = humans[assignment[i]];
    }
  }
  else if (n_prev < n_curr) {
    std::vector<bool> obs_used(n_curr, false);
    for (int i = 0; i < n_prev; ++i) {
      humans_prev[i] = humans[assignment[i]];
      obs_used[assignment[i]] = true;
    }
    humans_prev.reserve(n_curr);
    for(int j = 0; j < n_curr; ++j){
      if(!obs_used[j]){
        humans_prev.push_back(humans[j]);
      }
    }
  }
  else if (n_prev > n_curr){
    std::vector<Human> humans_temp;
    humans_temp.reserve(n_curr);
    for (int i = 0; i < n_prev; ++i) {
      if(assignment[i] >= 0)
        humans_temp.push_back(humans[assignment[i]]);
    }
    humans_prev = humans_temp;
  }
}

void PoseEstimator::nms_bbox(std::vector<Detection> &dets, const float &nms_threshold_){
  if(dets.empty())
    return;

  std::sort(dets.begin(), dets.end(), [](const Detection& d1, const Detection& d2){return d1.score > d2.score;});

  for (int i = 0; i < dets.size(); ++i) {
    if(dets[i].score > 0.0f){
      for(int j = i+1; j < dets.size(); ++j){
        if(dets[j].score > 0.0f){
          float iou = bbox_iou(dets[i].bbox, dets[j].bbox);
          if(iou > nms_threshold_){
            dets[j].score = 0.0f; // Mark to be deleted.
          }
        }
      }
    }
  }

  dets.erase(std::remove_if(dets.begin(), dets.end(), [](const Detection& d){return d.score == 0.0f;}), dets.end());
}

void PoseEstimator::nms_humans(std::vector<Human> &humans, const float &nms_threshold_){
  if(humans.empty())
    return;

  std::sort(humans.begin(), humans.end(), [](const Human& h1, const Human& h2){return h1.conf > h2.conf;});

  for (int i = 0; i < humans.size(); ++i) {
    if(humans[i].conf > 0.0f){
      for(int j = i+1; j < humans.size(); ++j){
        if(humans[j].conf > 0.0f){
          float iou = bbox_iou(humans[i].bbox, humans[j].bbox);
          if(iou > nms_threshold_){
            humans[j].conf = -1.0f; // Mark to be deleted.
            //humans[j].estimate_depth = false; // irrelevant, as this detection will be deleted anyways..
          }
          else if(iou > 0.03f){ // if there is any overlap, don't use RGB-D depth for smaller bounding box which might be (partially) oclcuded
            if(humans[i].bbox.area() < humans[j].bbox.area())
              humans[i].estimate_depth = false;
            else
              humans[j].estimate_depth = false;
          }
        }
      }
    }
  }

  humans.erase(std::remove_if(humans.begin(), humans.end(), [](const Human& h){return h.conf <= -1.0f;}), humans.end());
}

void PoseEstimator::enlarge_bbox(std::vector<Detection> &dets){
  for (auto& det: dets) {
    float width = det.bbox.width();
    float height = det.bbox.height();
    float tmp_factor = (prev_crop_factor - 1.f) / 2.f;
    det.bbox = Detection::BBox{det.bbox.ymin - tmp_factor * height, det.bbox.xmin - tmp_factor * width,
                               det.bbox.ymax + tmp_factor * height, det.bbox.xmax + tmp_factor * width};
  }
}

void PoseEstimator::Human::set_bbox(){
  std::vector<Keypoint> kps_copy = kps;
  auto kps_copy_valid_end = std::remove_if(kps_copy.begin(), kps_copy.end(), [](const Keypoint& kp){return kp.conf == 0.0f;});
  std::vector<Keypoint>::iterator ymin, xmin, ymax, xmax;
  std::tie(ymin, ymax) = std::minmax_element(kps_copy.begin(), kps_copy_valid_end, [](const Keypoint& kp1, const Keypoint& kp2){return kp1.y < kp2.y;});
  std::tie(xmin, xmax) = std::minmax_element(kps_copy.begin(), kps_copy_valid_end, [](const Keypoint& kp1, const Keypoint& kp2){return kp1.x < kp2.x;});

  bbox = Detection::BBox{ymin->y, xmin->x, ymax->y, xmax->x};
}

float PoseEstimator::Human::oks_dist(const Human &other){
  float area = bbox.area();
  if(area <= 0.0f)
    return 1.0f;

  float oks = 0.0f;
  int num_kps_used = 0;

  for(int i = 0; i < Human::num_kps; ++i){
    if(this->kps[i].conf > 0.0f && other.kps[i].conf > 0.0f){
      float dx = this->kps[i].x - other.kps[i].x;
      float dy = this->kps[i].y - other.kps[i].y;
      oks += std::exp(-(dx*dx + dy*dy) / (2 * area * PoseEstimator::oks_kappas[i]));
      ++num_kps_used;
    }
  }

  if(num_kps_used > 0)
    return oks / num_kps_used;
  else
    return 0.0f;
}

void PoseEstimator::humans_to_dets(std::vector<Detection> &dets_out, const std::vector<Human> &humans){
  dets_out.clear();
  dets_out.reserve(humans.size());
  for (const auto& human : humans) {
    if(human.num_valid_kps - human.num_occluded_kps > 0 && human.conf > det_thresh){
      std::vector<Detection::Keypoint> kps_det(human.kps.size());
      for(int kp_idx = 0; kp_idx < human.kps.size(); ++kp_idx){
          kps_det[kp_idx].x = human.kps[kp_idx].x;
          kps_det[kp_idx].y = human.kps[kp_idx].y;
          kps_det[kp_idx].conf = human.kps[kp_idx].conf;
      }
      dets_out.push_back(Detection{0, 1, human.conf, human.bbox, kps_det}); // 1 = person class idx, human.bbox is w.r.t detected keypoints
    }
  }

  //nms_bbox(dets_out, nms_threshold);
  enlarge_bbox(dets_out);
}

void PoseEstimator::humans_kps_depth_est(std::vector<Human> &humans){
  std::lock_guard<std::mutex> lck_logits(segm_logits_depth_mutex);
  const bool check_kps_segm_class_and_depth = (!segm_logits.empty()) && segm_depth.cols == depth_aligned.cols && segm_depth.rows == depth_aligned.rows;
  const float scale_x = (float) (segm_logits.cols - 1) /  (float) (image_size.width); // -1 to imitate tensorflow "align-corners" behaviour:
  const float scale_y = (float) (segm_logits.rows - 1) / (float) (image_size.height);
  MapTypeLogitsConst logits_eigen(nullptr, 0, 0);
  if(check_kps_segm_class_and_depth)
    new (&logits_eigen) MapTypeLogitsConst(segm_logits.ptr<float>(0), segm_logits.cols * segm_logits.rows, ADE20K_INDOOR::NUM_CLASSES);

  for (auto& human : humans) {
    if(human.num_valid_kps - human.num_occluded_kps > 0 && human.conf > det_thresh && human.estimate_depth){
      const int n_kps = human.kps.size();
      human.depth_est.resize(n_kps, 0.0f);
      human.depth_sigma.resize(n_kps, 0.0f);
      for(int kp_idx = 0; kp_idx < n_kps; ++kp_idx) {
        const auto& kp = human.kps[kp_idx];
        if(kp.conf <= 0.f || (human.num_occluded_kps > 0 && human.kp_occluded[kp_idx]))
          continue;

        //interpolate semantic segmentation class of keypoint location
        bool kp_segm_is_occluding_class = false;
        if(check_kps_segm_class_and_depth){
          const Eigen::Vector2f kp_image_logits(scale_x * kp.x, scale_y * kp.y);
          //const Eigen::Vector2f kp_image_logits = flip ? Eigen::Vector2f(segm_logits.cols, segm_logits.rows) - kp_image : kp_image;

          typename VecLogits::Index maxIndex = -1;
          if ( kp_image_logits.x() >= 0 && kp_image_logits.x() <= segm_logits.cols - 1 && kp_image_logits.y() >= 0 && kp_image_logits.y() <= segm_logits.rows - 1)
          {
            //Bilinear Interpolation!
            const int v_up = static_cast<int>(std::floor(kp_image_logits.y()));
            const int v_down = static_cast<int>(std::ceil(kp_image_logits.y()));
            const int u_left = static_cast<int>(std::floor(kp_image_logits.x()));
            const int u_right = static_cast<int>(std::ceil(kp_image_logits.x()));
            const int idx_ul = v_up * segm_logits.cols + u_left;
            const int idx_ur = v_up * segm_logits.cols + u_right;
            const int idx_bl = v_down * segm_logits.cols + u_left;
            const int idx_br = v_down * segm_logits.cols + u_right;

            const float alpha_u = u_right - kp_image_logits.x();
            const float alpha_v = v_down - kp_image_logits.y();

            if(alpha_u < 0 || alpha_u > 1 || alpha_v < 0 || alpha_v > 1){
              ROS_ERROR("Interpolation factors out of bounds! u: %f, v: %f, alpha_u: %f, alpha_v: %f", kp_image_logits.x(), kp_image_logits.y(), alpha_u, alpha_v);
            }

            const VecLogits logit_up = alpha_u * logits_eigen.row(idx_ul) + (1.f - alpha_u) * logits_eigen.row(idx_ur);
            const VecLogits logit_down = alpha_u * logits_eigen.row(idx_bl) + (1.f - alpha_u) * logits_eigen.row(idx_br);
            VecLogits logit = alpha_v * logit_up + (1.f - alpha_v) * logit_down;
            logit.maxCoeff(&maxIndex);
          }

          kp_segm_is_occluding_class = (maxIndex != ADE20K_INDOOR::person) && (maxIndex >= ADE20K_INDOOR::table); //&& (maxIndex != ADE20K_INDOOR::poster), column ?!
        }

        std::vector<uint16_t> depth_candidates, depth_candidates_segm_bg; // use the median depth value of a small environment around the detected joint.
        depth_candidates.reserve(m_depth_env_size * m_depth_env_size);
        if(kp_segm_is_occluding_class)
          depth_candidates_segm_bg.reserve(m_depth_env_size * m_depth_env_size);

        int u = std::lround((flip ? image_size.width - kp.x : kp.x) * depth_color_scale);
        int v = std::lround((flip ? image_size.height - kp.y : kp.y) * depth_color_scale);
        for (int uu = u - m_depth_env_delta; uu <= u + m_depth_env_delta; ++uu) {
          for (int vv = v - m_depth_env_delta; vv <= v + m_depth_env_delta; ++vv) {
            if(uu < 0 || vv < 0 || uu >= depth_aligned.cols || vv >= depth_aligned.rows)
              continue;
            if(depth_aligned.at<uint16_t>(vv,uu) > m_min_depth && depth_aligned.at<uint16_t>(vv,uu) < m_max_depth)
              depth_candidates.push_back(depth_aligned.at<uint16_t>(vv,uu));
            if(kp_segm_is_occluding_class && segm_depth.at<uint16_t>(vv,uu) > m_min_depth && segm_depth.at<uint16_t>(vv,uu) < m_max_depth)
              depth_candidates_segm_bg.push_back(segm_depth.at<uint16_t>(vv,uu));
          }
        }

        if(depth_candidates.empty()){
//          ROS_WARN("No Valid Depth Measurements for joint %d!", kp_idx);
          continue; // no valid depth info found for joint.
        }
        std::sort(depth_candidates.begin(), depth_candidates.end());

        int depth_idx = depth_candidates.size() / 3; // 33 % quantile depth
        if(kp_idx == 15 || kp_idx == 16)
            depth_idx = std::min((size_t)(0.8*depth_candidates.size()), depth_candidates.size() - 1); // HACK for ankles, use farther depth (to avoid floor)
        const float depth_kp = (float)(depth_candidates[depth_idx]) / m_depth_scale;

        if(kp_segm_is_occluding_class){
          float depth_segm_bg = 0.f;
          if(!depth_candidates_segm_bg.empty()){
            std::sort(depth_candidates_segm_bg.begin(), depth_candidates_segm_bg.end());
            depth_segm_bg = (float)(depth_candidates_segm_bg[std::min(depth_idx, (int)depth_candidates_segm_bg.size() - 1)]) / m_depth_scale;
          }

          if(depth_segm_bg <= 0 || depth_segm_bg - depth_kp > 0.2f){ //background depth indetermined or keypoint significantly in front of background segmentation (-> keypoint moved in front of segmented surface)
            kp_segm_is_occluding_class = false; // --> keypoint is visible, use depth est
          }
        }

        if(!kp_segm_is_occluding_class){
          human.depth_est[kp_idx] = depth_kp;
          human.depth_sigma[kp_idx] = (PoseEstimatorParams::sigma_meas_depth_max - PoseEstimatorParams::sigma_meas_depth_min) / (1.f + std::exp(-PoseEstimatorParams::sigma_meas_depth_steepness * (PoseEstimatorParams::sigma_meas_depth_steep_point - std::min(kp.conf, kp.conf / human.conf)))) + PoseEstimatorParams::sigma_meas_depth_min;
        }
        else { // mark keypoint as occluded
          if(human.kp_occluded.empty()){
            human.kp_occluded.resize(n_kps, false);
            human.debug_occ_kps_orig.reserve(n_kps);
          }
          ++human.num_occluded_kps;
          human.kp_occluded[kp_idx] = static_cast<uint8_t>(human.num_occluded_kps); // save the index of occluded joint instead of just true / false, to be able to reference original keypoint in debug_occ_kps_orig..
          human.debug_occ_kps_orig.push_back(human.kps[kp_idx]);
          human.kps[kp_idx].conf = 0.f;
          // occluded keypoints must be flipped correctly for compatibility reasons with occlusion feedback..
          if(flip){
              human.kps[kp_idx].x = image_size.width - kp.x;
              human.kps[kp_idx].y = image_size.height - kp.y;
          }
        }
      }

      // Heuristic to detect partial occlusions.
      float mean_depth = 0.f;
      float mean_depth_all = 0.f, mean_depth_left = 0.f, mean_depth_right = 0.f, mean_depth_top = 0.f, mean_depth_bottom = 0.f;
      int mean_cnt_all = 0, mean_cnt_left = 0, mean_cnt_right = 0, mean_cnt_top = 0, mean_cnt_bottom = 0;
      if(human.depth_est[5] > 0.f){mean_depth_all += human.depth_est[5]; mean_depth_left += human.depth_est[5]; mean_depth_top += human.depth_est[5]; ++mean_cnt_all; ++mean_cnt_left; ++mean_cnt_top;} //LShoulder
      if(human.depth_est[6] > 0.f){mean_depth_all += human.depth_est[6]; mean_depth_right += human.depth_est[6]; mean_depth_top += human.depth_est[6]; ++mean_cnt_all; ++mean_cnt_right; ++mean_cnt_top;} //RShoulder
      if(human.depth_est[11] > 0.f){mean_depth_all += human.depth_est[11]; mean_depth_left += human.depth_est[11]; mean_depth_bottom += human.depth_est[11]; ++mean_cnt_all; ++mean_cnt_left; ++mean_cnt_bottom;} //LHip
      if(human.depth_est[12] > 0.f){mean_depth_all += human.depth_est[12]; mean_depth_right += human.depth_est[12]; mean_depth_bottom += human.depth_est[12]; ++mean_cnt_all; ++mean_cnt_right; ++mean_cnt_bottom;} //RHip

      if(mean_cnt_left > 0 && mean_cnt_right > 0){
        mean_depth_left /= mean_cnt_left;
        mean_depth_right /= mean_cnt_right;
        if(std::abs(mean_depth_left - mean_depth_right) > 0.5f)
          mean_depth = std::max(mean_depth_left, mean_depth_right); // left or right side are occluded, choose larger depth;
        else
          mean_depth = (mean_depth_left + mean_depth_right) / 2.f;
      }
      else if(mean_cnt_top > 0 && mean_cnt_bottom > 0){
        mean_depth_top /= mean_cnt_top;
        mean_depth_bottom /= mean_cnt_bottom;
        if(std::abs(mean_depth_top - mean_depth_bottom) > 0.7f)
          mean_depth = std::max(mean_depth_top, mean_depth_bottom); // top or bottom part are occluded, choose larger depth;
        else
          mean_depth = (mean_depth_top + mean_depth_bottom) / 2.f;
      }
      else if (mean_cnt_all > 0)
        mean_depth = mean_depth_all / mean_cnt_all;

      if(mean_depth > 0.f){
        for(int kp_idx = 0; kp_idx < n_kps; ++kp_idx) {
          if(human.depth_est[kp_idx] > 0){
            const float delta_to_mean = std::abs(human.depth_est[kp_idx] - mean_depth); //TODO: expected depths varies with keypoint vertical pos.. (see pose_analyzer.py)
            if(delta_to_mean > 0.75f){
              human.depth_est[kp_idx] = mean_depth;
              human.depth_sigma[kp_idx] = PoseEstimatorParams::sigma_meas_depth_max;
            }
          }
        }
      }
    }
  }
}

void PoseEstimator::humans_to_msg(person_msgs::Person2DOcclusionList &msg_out, const std::vector<Human> &humans, const ros::Time &curr_time, const bool add_occluded_fb){
  msg_out.header.stamp = curr_time;
  msg_out.header.frame_id = camera + "_color_optical_frame";

  msg_out.persons.reserve(humans.size());
  for (const auto& human : humans) {
    person_msgs::Person2DOcclusion person;
    person.score = human.conf; // / (1.0f + (float) w_feedback);
    person.n_valid = static_cast<uint8_t>(human.num_valid_kps);
    person.n_occluded = static_cast<uint8_t>(human.num_occluded_kps);
    person.occluded = human.kp_occluded;
    person.depth_est = human.depth_est;
    person.depth_sigma =  human.depth_sigma;
    //TODO: person.id

    if(flip)
      person.bbox = {image_size.width - human.bbox.xmax, image_size.height - human.bbox.ymax, image_size.width - human.bbox.xmin, image_size.height - human.bbox.ymin};
    else
      person.bbox = {human.bbox.xmin, human.bbox.ymin, human.bbox.xmax, human.bbox.ymax};

    const int n_kps = human.kps.size();
    person.keypoints.reserve(n_kps);
    for(int kp_idx = 0; kp_idx < n_kps; ++kp_idx) {
      const auto& kp = human.kps[kp_idx];
      bool kp_occluded = false; // occluded keypoints are replaced by feedback and don't need to be flipped!
      if(person.n_occluded > 0 && person.occluded[kp_idx]){ //update bbox in case of occluded joints!
        kp_occluded = true;
        if(kp.x < person.bbox[0]){person.bbox[0] = kp.x;}
        if(kp.y < person.bbox[1]){person.bbox[1] = kp.y;}
        if(kp.x > person.bbox[2]){person.bbox[2] = kp.x;}
        if(kp.y > person.bbox[3]){person.bbox[3] = kp.y;}
      }

      person_msgs::Keypoint2D kp_msg;
      kp_msg.x = flip && !kp_occluded ? image_size.width - kp.x : kp.x;
      kp_msg.y = flip && !kp_occluded ? image_size.height - kp.y : kp.y;
      kp_msg.score = kp.conf;
      kp_msg.cov[0] = kp.cov(0, 0);
      kp_msg.cov[1] = kp.cov(0, 1);
      kp_msg.cov[2] = kp.cov(1, 1);
      person.keypoints.push_back(kp_msg);
    }

    person.debug_occ_kps_orig.reserve(human.debug_occ_kps_orig.size());
    for (const auto& kp : human.debug_occ_kps_orig) {
      person_msgs::Keypoint2D kp_msg;
      kp_msg.x = flip ? image_size.width - kp.x : kp.x;
      kp_msg.y = flip ? image_size.height - kp.y : kp.y;
      kp_msg.score = kp.conf;
      kp_msg.cov[0] = kp.cov(0, 0);
      kp_msg.cov[1] = kp.cov(0, 1);
      kp_msg.cov[2] = kp.cov(1, 1);
      person.debug_occ_kps_orig.push_back(kp_msg);
    }

    msg_out.persons.push_back(person);
  }

  if(add_occluded_fb){
    for(auto& fb_skel_occ : fb_skeletons_occluded){
      fb_skel_occ.score = 0.f;
      fb_skel_occ.n_occluded = fb_skel_occ.n_valid;
      const int n_kps_occ = fb_skel_occ.keypoints.size();
      for (int kp_idx = 0; kp_idx < n_kps_occ; ++kp_idx) { // have to iterate through the keypoints, as not all of them might initially have been marked occlued.
        auto& kp_occ = fb_skel_occ.keypoints[kp_idx];
        if(kp_occ.score > 0.f){ // valid keypoint
          fb_skel_occ.occluded[kp_idx] = true;
          kp_occ.score = 0.f; // set score to 0, so that occluded keypoint will not be used in triangulation..
        }
      }
    }
    msg_out.persons.insert( msg_out.persons.end(), fb_skeletons_occluded.begin(), fb_skeletons_occluded.end() );
  }
}

void PoseEstimator::human_check_occlusion(Human &human, const person_msgs::Person2DOcclusion& human_feedback){
  if(human.kps.size() != human_feedback.keypoints.size() || human.kps.size() != human_feedback.occluded.size()){
    ROS_ERROR("Unequal number of keypoints in occlusion check: %zu vs %zu (occluded: %zu). Aborting!", human.kps.size(), human_feedback.keypoints.size(), human_feedback.occluded.size());
  }
  const int num_keypoints = human.kps.size();
  human.kp_occluded.resize(num_keypoints, false);
  human.debug_occ_kps_orig.reserve(human_feedback.n_occluded);

  for (int kp_idx = 0; kp_idx < num_keypoints; ++ kp_idx) {
    if(human_feedback.occluded[kp_idx]){
      ++human.num_occluded_kps;
      if(human.kps[kp_idx].conf == 0.0f)
        ++human.num_valid_kps;
      human.kp_occluded[kp_idx] = static_cast<uint8_t>(human.num_occluded_kps); // save the index of occluded joint instead of just true / false, to be able to reference original keypoint in debug_occ_kps_orig..
      human.debug_occ_kps_orig.push_back(human.kps[kp_idx]);

      auto& kp_occluded = human.kps[kp_idx];
      const auto& kp_feedback = human_feedback.keypoints[kp_idx];
      kp_occluded.x = kp_feedback.x;
      kp_occluded.y = kp_feedback.y;
      kp_occluded.cov << kp_feedback.cov[0], kp_feedback.cov[1], kp_feedback.cov[1], kp_feedback.cov[2];
      kp_occluded.conf = 0.f; // set score to 0, so that occluded keypoint will not be used in triangulation..
    }
  }
}

Detection::BBox PoseEstimator::calc_crop_info(const Detection &det){
  const auto& input_shape = engine.pose_input_tensor_shape();
  float aspect_ratio = (float)input_shape[2] / input_shape[1]; // width / height
  float width = det.bbox.width();
  float height = det.bbox.height();
  cv::Point2f center(det.bbox.xmin + width * 0.5f, det.bbox.ymin + height * 0.5f);
  if(width > aspect_ratio * height)
    height = width / aspect_ratio;
  else if(width < aspect_ratio * height)
    width = height * aspect_ratio;
  cv::Point2f scale(width * 1.25f, height * 1.25f);

  return Detection::BBox{center.y - 0.5f * scale.y, center.x - 0.5f * scale.x,
                         center.y + 0.5f * scale.y, center.x + 0.5f * scale.x}; //y0, x0, y1, x1
}

void PoseEstimator::crop_bbox(cv::Mat& cropped_img, const cv::Mat& orig_img, const Detection::BBox& crop_info){
  const auto& input_shape = engine.pose_input_tensor_shape();
  cv::Size input_size(input_shape[2], input_shape[1]);

  float transform_data[6] = {input_size.width / crop_info.width(), 0.f, (0.5f - crop_info.cx() / crop_info.width()) * input_size.width , 0.f, input_size.height / crop_info.height(), (0.5f - crop_info.cy() / crop_info.height()) * input_size.height};
  cv::Mat warp_mat(2, 3, CV_32FC1, (void*)transform_data);

  cv::warpAffine(orig_img, cropped_img, warp_mat, input_size);
}

Detection::BBox PoseEstimator::crop_bbox(cv::Mat &cropped_img, const cv::Mat &orig_img, const Detection &det){
  const auto& input_shape = engine.pose_input_tensor_shape();
  float aspect_ratio = (float)input_shape[2] / input_shape[1]; // width / height
  float width = det.bbox.width();
  float height = det.bbox.height();
  cv::Point2f center(det.bbox.xmin + width * 0.5f, det.bbox.ymin + height * 0.5f);
  if(width > aspect_ratio * height)
    height = width / aspect_ratio;
  else if(width < aspect_ratio * height)
    width = height * aspect_ratio;
  cv::Point2f scale(width * 1.25f, height * 1.25f);
  cv::Size input_size(input_shape[2], input_shape[1]);

  float transform_data[6] = {input_size.width / scale.x, 0.f, (0.5f - center.x / scale.x) * input_size.width , 0.f, input_size.height / scale.y, (0.5f - center.y / scale.y) * input_size.height};
  cv::Mat warp_mat(2, 3, CV_32FC1, (void*)transform_data);

  cv::warpAffine(orig_img, cropped_img, warp_mat, input_size);

  return Detection::BBox{center.y - 0.5f * scale.y, center.x - 0.5f * scale.x,
                         center.y + 0.5f * scale.y, center.x + 0.5f * scale.x}; //y0, x0, y1, x1
}

void PoseEstimator::render_heatmaps(cv::Mat &heatmaps, const person_msgs::Person2DOcclusion &human, const Detection::BBox &cropinfo, std::vector<cv::Point2i>* hm_fb_kps_centers){
  heatmaps.create(heatmap_size, CV_32FC(heatmap_channels));
  heatmaps = cv::Scalar::all(0.0);

  float scale_x = cropinfo.width() / heatmap_size.width;
  float scale_y = cropinfo.height() / heatmap_size.height;
  if(scale_x == 0.0f || scale_y == 0.0f)
    return;

  if (hm_fb_kps_centers != nullptr)
    *hm_fb_kps_centers = std::vector<cv::Point2i>(heatmap_channels, cv::Point2i(-2*max_env_swap, -2*max_env_swap));

  for(int part_idx = 0; part_idx < heatmap_channels; ++part_idx) {
    const auto& kp = human.keypoints[part_idx];
    if(kp.score == 0.0f)
      continue;

    Eigen::Matrix2f cov;
    cov << kp.cov[0] / (scale_x*scale_x), kp.cov[1] / (scale_x*scale_y), // scale covariance to img crop coordinates
           kp.cov[1] / (scale_x*scale_y), kp.cov[2] / (scale_y*scale_y);
    cov += noise_offset;

    float real_x = flip ? humans_fb_img_size.width - kp.x : kp.x;
    float real_y = flip ? humans_fb_img_size.height - kp.y : kp.y;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> esolver(cov, Eigen::DecompositionOptions::EigenvaluesOnly);
    float sigma_fb = sqrt(esolver.eigenvalues()(1));
    float sigma = std::min(sigma_max, sigma_fb);
    float tmp_factor = hm_fb_sigma / sigma;

    int tmp_size = (int)std::ceil(sigma * 3);
    int mu_x = std::lround((real_x - cropinfo.xmin) / scale_x);
    int mu_y = std::lround((real_y - cropinfo.ymin) / scale_y);
    int ulx = mu_x - tmp_size;
    int uly = mu_y - tmp_size;
    int brx = mu_x + tmp_size + 1; // past-the-end index
    int bry = mu_y + tmp_size + 1;

    if (hm_fb_kps_centers != nullptr){
      hm_fb_kps_centers->at(part_idx).x = mu_x;
      hm_fb_kps_centers->at(part_idx).y = mu_y;
    }

    if(ulx >= heatmap_size.width || uly >= heatmap_size.height || brx <= 0 || bry <= 0)
      continue;

    float sub_x = ((real_x - cropinfo.xmin) / scale_x) - mu_x;
    float sub_y = ((real_y - cropinfo.ymin) / scale_y) - mu_y;
    cv::Mat g = get_gaussian_kernel_cov(tmp_size, sub_x, sub_y, cov(0,0), cov(1,1), cov(0,1)) * (double) tmp_factor; //render directional covariance!

    cv::Mat hm_reshaped = heatmaps.reshape(1, heatmaps.rows * heatmaps.cols);
    cv::Mat g_reshaped = g.reshape(1, g.rows * g.cols);

    int img_y0 = std::max(0, uly), img_x0 = std::max(0, ulx);
    int g_y0 = std::max(0, -uly), g_x0 = std::max(0, -ulx);
    int img_y1 = std::min(bry, heatmap_size.height);
    int roi_width = std::min(brx, heatmap_size.width) - img_x0;
    for(int yy = img_y0, gyy = g_y0; yy < img_y1; ++yy, ++gyy){
      int gxx0 = g_x0 + gyy * g.cols;
      int xx0 =  img_x0 + yy * heatmaps.cols;
      g_reshaped.rowRange(gxx0, gxx0 + roi_width).copyTo(hm_reshaped.col(part_idx).rowRange(xx0, xx0 + roi_width));
    }
  }
}

void PoseEstimator::swap_hm_channels(cv::Mat &heatmaps, const std::vector<cv::Point2i> &hm_fb_kps_centers, const std::vector<std::vector<int> > &kps_symm_pairs){
  for (const auto& pair : kps_symm_pairs) {
    const auto& pl = hm_fb_kps_centers.at(pair[0]);
    const auto& pr = hm_fb_kps_centers.at(pair[1]);
    double dist = cv::norm(pl - pr);
    int min_dist = pair[0] <= 9 ? min_dist_swap[0] : min_dist_swap[1];
    int env_size = std::min(max_env_swap, (int)(0.5 * (dist - min_dist)));
    if(env_size <= 0) // python-script: if env_size < 0:
      continue;

    cv::Point2i ull(pl.x - env_size, pl.y - env_size);
    cv::Point2i brl(pl.x + env_size + 1, pl.y + env_size + 1);
    cv::Point2i ulr(pr.x - env_size, pr.y - env_size);
    cv::Point2i brr(pr.x + env_size + 1, pr.y + env_size + 1);

    if(ull.x >= heatmaps.cols || ull.y >= heatmaps.rows || brl.x < 0 || brl.y < 0 ||
       ulr.x >= heatmaps.cols || ulr.y >= heatmaps.rows || brr.x < 0 || brr.y < 0)
      continue;

    cv::Mat hm_reshaped = heatmaps.reshape(1, heatmaps.rows * heatmaps.cols);
    double sumll = 0.0, sumlr = 0.0, sumrl = 0.0, sumrr = 0.0;

    cv::Range hm_xl(std::max(0, ull.x), std::min(brl.x, heatmaps.cols));
    cv::Range hm_yl(std::max(0, ull.y), std::min(brl.y, heatmaps.rows));
    for(int yy = hm_yl.start; yy < hm_yl.end; ++yy){
      sumll += cv::sum(hm_reshaped.col(pair[0]).rowRange(hm_xl.start + yy * heatmaps.cols, hm_xl.end + yy * heatmaps.cols))[0];
      sumlr += cv::sum(hm_reshaped.col(pair[1]).rowRange(hm_xl.start + yy * heatmaps.cols, hm_xl.end + yy * heatmaps.cols))[0];
    }
    bool swapl = sumll < sumlr;

    cv::Range hm_xr(std::max(0, ulr.x), std::min(brr.x, heatmaps.cols));
    cv::Range hm_yr(std::max(0, ulr.y), std::min(brr.y, heatmaps.rows));
    for(int yy = hm_yr.start; yy < hm_yr.end; ++yy){
      sumrr += cv::sum(hm_reshaped.col(pair[1]).rowRange(hm_xr.start + yy * heatmaps.cols, hm_xr.end + yy * heatmaps.cols))[0];
      sumrl += cv::sum(hm_reshaped.col(pair[0]).rowRange(hm_xr.start + yy * heatmaps.cols, hm_xr.end + yy * heatmaps.cols))[0];
    }
    bool swapr = sumrr < sumrl;

    if(swapl && swapr){
      ROS_INFO_THROTTLE(1.0, "Left-Right swap detected! joint %d <-> %d. exchanging heatmap channels", pair[0], pair[1]);
//      ROS_INFO("Left-Right swap detected! joint %d <-> %d. exchanging heatmap channels", pair[0], pair[1]);
      cv::Mat tmp;
      hm_reshaped.col(pair[0]).copyTo(tmp);
      hm_reshaped.col(pair[1]).copyTo(hm_reshaped.col(pair[0]));
      tmp.copyTo(hm_reshaped.col(pair[1]));
    }
  }
}

PoseEstimator::Human PoseEstimator::parse_skeleton(const cv::Mat &heatmaps_, const Detection::BBox &crop_info, float part_thresh){
  int num_kps = heatmaps_.channels();
  int hm_width = heatmaps_.cols;
  int hm_height = heatmaps_.rows;
  MapTypeFloat heatmaps(heatmaps_.ptr<float>(0), hm_height * hm_width, num_kps);

  cv::Mat hm_thresh; // TODO: is this necessary ? Can this be done more efficiently (i.e. only for really concerned pixels) ?
  cv::threshold(heatmaps_, hm_thresh, hm_cov_thresh, 1.0, CV_THRESH_TOZERO);

  Human human;
  human.kps.resize(num_kps);
  float ymin = std::numeric_limits<float>::max();
  float xmin = ymin;
  float ymax = std::numeric_limits<float>::lowest();
  float xmax = ymax;
  float cum_conf = 0.0f;
  for(int kp_idx = 0; kp_idx < num_kps; ++kp_idx){
    Eigen::MatrixXf::Index maxIndex;
    float conf = heatmaps.col(kp_idx).maxCoeff(&maxIndex);
    if(conf > part_thresh){
      float x = (float)(maxIndex % hm_width);
      float y = (float)(maxIndex / hm_width);

      //post-processing
      int px = (maxIndex % hm_width);
      int py = (maxIndex / hm_width);
      int signx = 0, signy = 0;
      if(1 < px && px < hm_width - 1 && 1 < py && py < hm_height - 1){
        float dx = heatmaps(py * hm_width + px + 1, kp_idx) - heatmaps(py * hm_width + px - 1, kp_idx);
        float dy = heatmaps((py+1) * hm_width + px, kp_idx) - heatmaps((py-1) * hm_width + px, kp_idx);
        signx = sign(dx);
        signy = sign(dy);
      }
      x += signx * 0.25f;
      y += signy * 0.25f;

      int ulx = px - PoseEstimatorParams::hm_cov_env_size;
      int uly = py - PoseEstimatorParams::hm_cov_env_size;
      int brx = px + PoseEstimatorParams::hm_cov_env_size + 1; // past-the-end index
      int bry = py + PoseEstimatorParams::hm_cov_env_size + 1;
      int img_y0 = std::max(0, uly), img_x0 = std::max(0, ulx);
      int g_y0 = std::max(0, -uly), g_x0 = std::max(0, -ulx);
      int roi_height = std::min(bry, hm_height) - img_y0;
      int roi_width = std::min(brx, hm_width) - img_x0;

      MapTypeFloatCh hm_thresh_ch(hm_thresh.ptr<float>(0) + kp_idx, hm_thresh.rows, hm_thresh.cols, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(hm_width * num_kps, num_kps));
      const auto& weights = hm_thresh_ch.block(img_y0, img_x0, roi_height, roi_width);
      float cov_norm = conf * weights.count(); // normalization factor
      float scale_x = crop_info.width()  / hm_width; //scale factors to original pixel coordinates
      float scale_y = crop_info.height() / hm_height;

      float cov_xx = kernel_cov_xx[cov_kernel_idx(signx)].block(g_y0, g_x0, roi_height, roi_width).cwiseProduct(weights).sum() / cov_norm * scale_x * scale_x;
      float cov_yy = kernel_cov_yy[cov_kernel_idx(signy)].block(g_y0, g_x0, roi_height, roi_width).cwiseProduct(weights).sum() / cov_norm * scale_y * scale_y;
      float cov_xy = kernel_cov_xy[crosscov_kernel_idx(signx, signy)].block(g_y0, g_x0, roi_height, roi_width).cwiseProduct(weights).sum() / cov_norm * scale_x * scale_y;
      //cout << "joint: " << kp_idx << ": cov_xx: " << cov_xx << ", cov_yy: " << cov_yy << ", cov_xy: " << cov_xy << endl;

      human.kps[kp_idx].cov << cov_xx, cov_xy,
                               cov_xy, cov_yy;
      human.kps[kp_idx].conf = conf;
      human.kps[kp_idx].x = x * scale_x + crop_info.xmin;
      human.kps[kp_idx].y = y * scale_y + crop_info.ymin;

      cum_conf += conf;
      human.num_valid_kps++;

      if(human.kps[kp_idx].y < ymin){ymin = human.kps[kp_idx].y;}
      if(human.kps[kp_idx].x < xmin){xmin = human.kps[kp_idx].x;}
      if(human.kps[kp_idx].y > ymax){ymax = human.kps[kp_idx].y;}
      if(human.kps[kp_idx].x > xmax){xmax = human.kps[kp_idx].x;}
    }
    else {
      human.kps[kp_idx].conf = 0.0f;
    }
  }

  if(human.num_valid_kps > 0){
    human.conf = cum_conf / human.num_valid_kps;
    human.bbox = Detection::BBox{ymin, xmin, ymax, xmax};
  }
  else
    human.conf = 0.0f;

  return human;
}

cv::Mat PoseEstimator::get_gaussian_kernel(int size, float x_sub, float y_sub, float sigma_x, float sigma_y, float angle){
  int tmp_size = 2 * size + 1;
  cv::Mat gauss_y(tmp_size, 1, CV_32FC1);
  cv::Mat gauss_x(1, tmp_size, CV_32FC1);
  float sx = 2 * sigma_x * sigma_x;
  float sy = 2 * sigma_y * sigma_y;
  float mux = size + x_sub;
  float muy = size + y_sub;

  for(int i = 0; i < tmp_size; ++i){
    gauss_y.at<float>(i, 0) = std::exp(-(i - muy) * (i - muy) / sy);
    gauss_x.at<float>(0, i) = std::exp(-(i - mux) * (i - mux) / sx);
  }

  return gauss_y * gauss_x;
}

cv::Mat PoseEstimator::get_gaussian_kernel_cov(int size, float x_sub, float y_sub, float cov_xx, float cov_yy, float cov_xy){
  int tmp_size = 2 * size + 1;
  cv::Mat gauss(tmp_size, tmp_size, CV_32FC1);
  float mux = size + x_sub;
  float muy = size + y_sub;
  float factor = -1.0f / (2*(cov_xx*cov_yy-cov_xy*cov_xy));
  for(int y = 0; y < tmp_size; ++y){
    for(int x = 0; x < tmp_size; ++x){
      gauss.at<float>(y,x) = std::exp(factor * (cov_yy * (x-mux) * (x-mux) - 2*cov_xy * (x-mux) * (y-muy) + cov_xx * (y - muy) * (y-muy)));
    }
  }
  return gauss;
}

int PoseEstimator::cov_kernel_idx(int sign){
  if (sign < 0)
    return 0;
  else if (sign == 0)
    return 1;
  else
    return 2;
}

int PoseEstimator::crosscov_kernel_idx(int signx, int signy){
  if(signx < 0){
    if(signy < 0)
      return 0;
    else if (signy == 0)
      return 1;
    else
      return 2;
  }
  else if(signx == 0){
    if(signy < 0)
      return 3;
    else if (signy == 0)
      return 4;
    else
      return 5;
  }
  else {
    if(signy < 0)
      return 6;
    else if (signy == 0)
      return 7;
    else
      return 8;
  }
}

void PoseEstimator::set_cov_kernels(){
  int tmp_size = 2 * PoseEstimatorParams::hm_cov_env_size + 1;
  float sub_x[3] = {-0.25f, 0.0f, 0.25f};
  float sub_y[3] = {-0.25f, 0.0f, 0.25f};

  for (int subx_idx = 0; subx_idx < 3; ++subx_idx){
    float mux = PoseEstimatorParams::hm_cov_env_size + sub_x[subx_idx];
    for (int x = 0; x < tmp_size; ++x) {
      for (int y = 0; y < tmp_size; ++y) {
        kernel_cov_xx[subx_idx](y, x) = (x - mux) * (x - mux);
      }
    }
  }

  for (int suby_idx = 0; suby_idx < 3; ++suby_idx){
    float muy = PoseEstimatorParams::hm_cov_env_size + sub_y[suby_idx];
    for (int x = 0; x < tmp_size; ++x) {
      for (int y = 0; y < tmp_size; ++y) {
        kernel_cov_yy[suby_idx](y, x) = (y - muy) * (y - muy);
      }
    }
  }

  for (int subx_idx = 0; subx_idx < 3; ++subx_idx){
    for (int suby_idx = 0; suby_idx < 3; ++suby_idx) {
      float mux = PoseEstimatorParams::hm_cov_env_size + sub_x[subx_idx];
      float muy = PoseEstimatorParams::hm_cov_env_size + sub_y[suby_idx];
      for (int x = 0; x < tmp_size; ++x) {
        for (int y = 0; y < tmp_size; ++y) {
          kernel_cov_xy[subx_idx * 3 + suby_idx](y, x) = (x - mux) * (y - muy);
        }
      }
    }
  }
}

void PoseEstimator::plot_covariance(cv::Mat &img, const Human &human, const Detection::BBox &crop_info, const cv::Mat &colors){
  for (int kp_idx = 0; kp_idx < human.kps.size(); ++kp_idx) {
    const auto& kp = human.kps[kp_idx];
    if (kp.conf == 0.0f)
      continue;

    float scale_x = crop_info.width()  / img.cols; //scale factors to original pixel coordinates
    float scale_y = crop_info.height() / img.rows;
    float x = (kp.x - crop_info.xmin) / scale_x;
    float y = (kp.y - crop_info.ymin) / scale_y;
    img.at<cv::Vec3b>(std::lround(y), std::lround(x)) = 255 * colors.at<cv::Vec3f>(kp_idx);
    //cv::circle(img, cv::Point2f(x, y), 1, 255*colors.at<cv::Vec3f>(kp_idx), -1);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> esolver(kp.cov / (scale_x * scale_x)); // scale covariance to image crop (assume equal scaling of x and y); // eigenvalues are sorted in increasing order
    //Calculate the angle between the largest eigenvector and the x-axis
    double angle = (double)std::atan2(esolver.eigenvectors()(1,1), esolver.eigenvectors()(0,1)) * 180.0 / M_PI;
    //Calculate the size of the minor and major axes
    float halfmajoraxissize=2.4477f*sqrt(esolver.eigenvalues()(1)); //2.4477 (2-sigma), 3.0349 (3-sigma)
    float halfminoraxissize=2.4477f*sqrt(esolver.eigenvalues()(0));
    cv::ellipse(img, cv::Point2f(x, y), cv::Size(std::lround(halfmajoraxissize), std::lround(halfminoraxissize)), angle, 0, 360, 255*colors.at<cv::Vec3f>(kp_idx), 1, cv::LINE_4);
  }
}

void PoseEstimator::align_depth(cv::Mat& depth_thermal){
  depth_aligned.create(depth_size, CV_16UC1);
  depth_aligned = cv::Scalar::all(0.0);
  const bool transform_to_thermal = (depth_thermal.cols == thermal_size.width && depth_thermal.rows == thermal_size.height);
  if(transform_to_thermal)
    depth_thermal = cv::Scalar::all(0.0);
  //MapTypeDepth depth_aligned_map(depth_aligned.ptr<uint16_t>(0), 1, depth_aligned.cols * depth_aligned.rows);
  MapTypeDepthConst depth_map(depth_input_img.ptr<uint16_t>(0), 1, depth_input_img.cols * depth_input_img.rows);
  Eigen::Matrix<bool, 1, Eigen::Dynamic, Eigen::RowMajor> depth_mask = (depth_map.array() > 0);
  Eigen::Array<float, 1, Eigen::Dynamic, Eigen::RowMajor> depth_scaled = depth_map.cast<float>().array() / m_depth_scale;
  Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor> points3D = m_depth_pixel_rays.array().rowwise() * depth_scaled; // Pixel-rays: 3 x HW, Depth: 1 x HW
//  Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor> points3D_br = g_depth_pixel_rays_br.array().rowwise() * depth_scaled; // Pixel-rays: 3 x HW, Depth: 1 x HW
  
  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> P = K_color_scaled * m_depth_to_color.affine();
  Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor> transformed_depth_homogenous = P.cast<float>() * points3D.colwise().homogeneous(); // P: 3x4, points3D 3xHW -> 4xHW (x, y, z) -> (x, y, z, 1)
//  Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor> transformed_depth_homogenous_br = P.cast<float>() * points3D_br.colwise().homogeneous(); // P: 3x4, points3D 3xHW -> 4xHW (x, y, z) -> (x, y, z, 1)
  Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor> transformed_depth_pixels = transformed_depth_homogenous.colwise().hnormalized(); // (u', v') shape 2xHW
//  Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor> transformed_depth_pixels_br = transformed_depth_homogenous_br.colwise().hnormalized(); // (u', v') shape 2xHW

  Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor> transformed_depth_pixels_thermal;
  if(transform_to_thermal){
    Eigen::Matrix<double, 3, 4, Eigen::RowMajor> Pthermal = K_thermal * m_depth_to_thermal.affine();
    Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor> transformed_depth_thermal_homogenous = Pthermal.cast<float>() * points3D.colwise().homogeneous();
    transformed_depth_pixels_thermal = transformed_depth_thermal_homogenous.colwise().hnormalized();
  }

  for(int idx = 0; idx < depth_size.width * depth_size.height; ++idx){ // Loops through the pixels
    if(depth_mask(idx) > 0){ // skip pixels with no valid depth data
      const int aligned_u = std::lround(transformed_depth_pixels.col(idx).x()); // Round to nearest integer.
      const int aligned_v = std::lround(transformed_depth_pixels.col(idx).y());
//      int aligned_u1 = std::lround(transformed_depth_pixels_br.col(idx).x()); // Round to nearest integer.
//      int aligned_v1 = std::lround(transformed_depth_pixels_br.col(idx).y());

      if (aligned_u >= 0 && aligned_v >= 0 && aligned_u < depth_size.width && aligned_v < depth_size.height) //if (aligned_u0 < 0 || aligned_v0 < 0 || aligned_u1 >= g_depth_size.width || aligned_v1 >= g_depth_size.height)
        depth_aligned.at<uint16_t>(aligned_v, aligned_u) = depth_aligned.at<uint16_t>(aligned_v, aligned_u) ? std::min(depth_aligned.at<uint16_t>(aligned_v, aligned_u), depth_map(idx)) : depth_map(idx);
      //int aligned_idx =  aligned_u * depth_size.width + aligned_v;
      //depth_aligned_map(aligned_idx) = depth_aligned_map(aligned_idx) ? std::min(depth_aligned_map(aligned_idx), depth_map(idx)) : depth_map(idx);

//      for(int v = aligned_v0; v <= aligned_v1; ++v){
//        for(int u = aligned_u0; u <= aligned_u1; ++u){
//          int aligned_idx =  v * g_depth_size.width + u;
//          depth_aligned_map(aligned_idx) = depth_aligned_map(aligned_idx) ? std::min(depth_aligned_map(aligned_idx), depth_map(idx)) : depth_map(idx);
//        }
//      }
      if(transform_to_thermal){
        const int aligned_u_thermal = std::lround(transformed_depth_pixels_thermal.col(idx).x()); // Round to nearest integer.
        const int aligned_v_thermal = std::lround(transformed_depth_pixels_thermal.col(idx).y());
        if (aligned_u_thermal >= 0 && aligned_v_thermal >= 0 && aligned_u_thermal < thermal_size.width && aligned_v_thermal < thermal_size.height)
          depth_thermal.at<uint16_t>(aligned_v_thermal, aligned_u_thermal) = depth_thermal.at<uint16_t>(aligned_v_thermal, aligned_u_thermal) ? std::min(depth_thermal.at<uint16_t>(aligned_v_thermal, aligned_u_thermal), depth_map(idx)) : depth_map(idx);
      }
    }
  }
}
