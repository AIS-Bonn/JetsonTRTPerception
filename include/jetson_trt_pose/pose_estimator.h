#pragma once
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <person_msgs/Person2DOcclusionList.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <opencv2/core.hpp>
#include <eigen3/Eigen/Core>

#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "jetson_trt_pose/pose_det_engine_jetson.h"

using std::string;

struct PoseEstimatorParams{
    bool pub_raw = false;
    bool debug_heatmap = false;
    string camera = "camera";
    string model_pose, model_det = "", model_segm = "", model_det_thermal = "";
    uint32_t precision = 1;
    string calib_data_dir = "", calib_data_dir_thermal = "";
    float det_thresh = 0.5;
    int top_k = 10;
    double max_delta_t_det = 0.5;
    float prev_crop_factor = 1.25f;
    float nms_threshold = 0.40f;
    string feedback_type = "";
    double max_fb_delay = 1.0;
    float sigma = 1.5f;
    double fb_min_iou = 0.40;
    //float sigma_score_factor = 7.5f;
    float sigma_max = 10.0f;
    bool do_swap_hm_channels = false;
    int min_dist_swap[2] = {3, 6};
    int max_env_swap = 2;
    bool h36m = false;
    bool flip = false;
    double hm_cov_thresh = 0.05;
    static const int hm_cov_env_size = 15;
    static constexpr float sigma_meas_depth_min = 0.15,
           sigma_meas_depth_max = 0.75,
           sigma_meas_depth_steepness = 20,
           sigma_meas_depth_steep_point = 0.65;
};

class PoseEstimator{
protected:
    struct Keypoint{
      float x;
      float y;
      float conf;
      Eigen::Matrix2f cov;
    };

    struct Human{
      std::vector<Keypoint> kps;
      std::vector<Keypoint> debug_occ_kps_orig;
      std::vector<uint8_t> kp_occluded;
      bool estimate_depth = true;
      std::vector<float> depth_est;
      std::vector<float> depth_sigma;
      float conf = 0.0f;
      int num_valid_kps = 0;
      int num_occluded_kps = 0;
      Detection::BBox bbox;
      void set_bbox();
      static const int num_kps = 17;
      float oks_dist(const Human& other);
    };

    //kappa = (2*sigma)^2, oks_sigmas[17] = {0.026f,  0.025f,  0.025f,  0.035f,  0.035f,  0.079f, 0.079f, 0.072f,  0.072f,  0.062f,  0.062f,  0.107f,  0.107f,  0.087f,  0.087f,  0.089f,  0.089f};
    static constexpr float oks_kappas[17] = {0.0027f, 0.0025f, 0.0025f, 0.0049f, 0.0049f, 0.025f, 0.025f, 0.0207f, 0.0207f, 0.0154f, 0.0154f, 0.0458f, 0.0458f, 0.0303f, 0.0303f, 0.0317f, 0.0317f};

    typedef Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MapTypeFloat;
    typedef Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> > MapTypeFloatCh;
    typedef Eigen::Map<const Eigen::Matrix<uint16_t, 1, Eigen::Dynamic, Eigen::RowMajor>> MapTypeDepthConst;
    typedef Eigen::Map<Eigen::Matrix<uint16_t, 1, Eigen::Dynamic, Eigen::RowMajor>> MapTypeDepth;

    bool pub_debug_heatmap, pub_raw_heatmap;
    string camera;
    float det_thresh;
    int top_k;
    double max_delta_t_det;
    float prev_crop_factor;
    float nms_threshold;
    string feedback_type;
    double max_fb_delay;
    double w_feedback, w_mult;
    float hm_fb_sigma, sigma_max; //sigma_score_factor
    double fb_min_iou;
    bool do_swap_hm_channels;
    int min_dist_swap[2];
    int max_env_swap;
    bool flip;
    double hm_cov_thresh;

    ros::NodeHandle& nh;
    ros::Publisher publisher_skeleton, publisher_obj, publisher_segm, publisher_det_thermal;
    ros::Publisher publisher_debug_img, publisher_segm_overlay;
    ros::Publisher publisher_raw_hm;

    PoseDetEngine engine;
    std::unique_ptr<ThermalDetEngine> thermal_engine_ptr;
    cv::Size heatmap_size;
    cv::Size image_size;
    int heatmap_channels;

    person_msgs::Person2DOcclusionList fb_skeletons;
    std::vector<person_msgs::Person2DOcclusion> fb_skeletons_occluded;
    ros::Subscriber fb_skel_sub;
    sensor_msgs::Image pred_heatmaps;
    ros::Subscriber pred_hm_sub;

    std::thread feedback_hm_thread;
    std::mutex fb_hm_mutex, humans_fb_mutex;
    std::condition_variable fb_hm_cv, humans_fb_cv;
    bool fb_hm_updated, humans_fb_updated;
    std::vector<cv::Mat> heatmaps_feedback;
    std::vector<std::vector<cv::Point2i> > hm_fb_kps_centers;
    std::vector<Detection::BBox> humans_fb_cropinfos;
    cv::Size humans_fb_img_size;
    std::vector<int> humans_fb_assignment;
    std::vector<person_msgs::Person2DOcclusion> humans_fb;

    std::thread segmentation_inference_thread;
    std::mutex segm_mutex;
    std::condition_variable segm_cv;
    bool segm_input_updated;
    cv::Mat segm_input_img;
    std_msgs::Header segm_input_header;
    cv::Mat segm_logits, segm_depth;
    std::mutex segm_logits_depth_mutex;

    std::thread depth_alignment_thread;
    std::mutex depth_mutex;
    std::condition_variable depth_cv;
    bool depth_input_updated;
    cv::Mat depth_input_img, depth_aligned;
    ros::Time depth_input_ts;
    cv::Size depth_size;
    float depth_color_scale;
    Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor> m_pixels_depth; //, g_pixels_depth_tl, g_pixels_depth_br;
    Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor> m_depth_pixel_rays; //, g_depth_pixel_rays_tl, g_depth_pixel_rays_br;
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_depth, K_color_scaled;
    Eigen::Affine3d m_depth_to_color;
    bool m_align_depth;
    static constexpr float m_depth_scale = 1000.0;
    static constexpr uint16_t m_min_depth = 300;
    static constexpr uint16_t m_max_depth = 10000;
    static const int m_depth_env_delta = 2;
    static const int m_depth_env_size = 2 * m_depth_env_delta + 1;
    ros::Publisher aligned_depth_debug_pub;

    Eigen::Affine3d m_depth_to_thermal;
    Eigen::Matrix<float, 3, 4> P_thermal_to_color;
    std::mutex depth_thermal_mutex;
    cv::Mat depth_aligned_thermal;
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_thermal;
    cv::Size thermal_size;
    std::vector<Detection> thermal_dets_transformed_to_color;
    bool thermal_dets_updated;

    Eigen::Matrix2f noise_offset;
    Eigen::Matrix<float, 2 * PoseEstimatorParams::hm_cov_env_size + 1, 2 * PoseEstimatorParams::hm_cov_env_size + 1> kernel_cov_xx[3], kernel_cov_yy[3], kernel_cov_xy[9];

    std::thread debug_hm_thread;
    std::mutex debug_hm_mutex;
    std::condition_variable debug_hm_cv;
    bool debug_hm_updated;
    ros::Time debug_hm_timestamp;
    cv::Mat debug_heatmaps, debug_heatmaps_feedback;
    Human debug_human;
    Detection::BBox debug_crop_info;

    void associate_feedback(int* assignment, const std::vector<Detection>& dets);
    void associate_humans(int* assignment, const std::vector<Human>& humans_prev, const std::vector<Human>& humans);
    void update_prev_humans(std::vector<Human>& humans_prev, const std::vector<Human>& humans, const std::vector<int>& assignment);

    Detection::BBox calc_crop_info(const Detection& det);
    Detection::BBox crop_bbox(cv::Mat& cropped_img, const cv::Mat& orig_img, const Detection& det);
    void crop_bbox(cv::Mat& cropped_img, const cv::Mat& orig_img, const Detection::BBox& crop_info);
    float bbox_iou(const Detection::BBox& bb1, const Detection::BBox& bb2);

    void humans_kps_depth_est(std::vector<Human> &humans);
    void nms_humans(std::vector<Human> &humans, const float &nms_threshold_);
    void humans_to_dets(std::vector<Detection>& dets_out, const std::vector<Human>& humans);
    void humans_to_msg(person_msgs::Person2DOcclusionList& msg_out, const std::vector<Human>& humans, const ros::Time& curr_time, const bool add_occluded_fb = false);
    void human_check_occlusion(Human& human, const person_msgs::Person2DOcclusion& human_feedback);

    Human parse_skeleton(const cv::Mat& heatmaps, const Detection::BBox& crop_info, float part_thresh = 0.30f);
    void swap_hm_channels(cv::Mat& heatmaps, const std::vector<cv::Point2i>& hm_fb_kps_centers, const std::vector<std::vector<int>>& kps_symm_pairs);

    void align_depth(cv::Mat& depth_thermal);
    
    //Only derived classes should be instantiated
    PoseEstimator(ros::NodeHandle& nh_, const PoseEstimatorParams& params = PoseEstimatorParams());
    ~PoseEstimator();

private:
    void debug_heatmap_cb();
    void feedback_hm_cb();
    void depth_alignment_cb();
    void segmentation_inference_callback();

    void fb_skeleton_cb(const person_msgs::Person2DOcclusionListConstPtr& humans);
    void render_heatmaps(cv::Mat& heatmaps, const person_msgs::Person2DOcclusion& human, const Detection::BBox& cropinfo, std::vector<cv::Point2i>* hm_fb_kps_centers = nullptr);
    void pred_hm_cb(const sensor_msgs::ImageConstPtr& pred_result);

    float oks_dist(const Detection& person_prev, const person_msgs::Person2DOcclusion& person_fb);
    void enlarge_bbox(std::vector<Detection>& dets);
    void nms_bbox(std::vector<Detection>& dets, const float& nms_threshold_);

    void set_cov_kernels();
    int cov_kernel_idx(int sign);
    int crosscov_kernel_idx(int signx, int signy);
    cv::Mat get_gaussian_kernel(int size, float x_sub, float y_sub, float sigma_x, float sigma_y, float angle = 0.0f);
    cv::Mat get_gaussian_kernel_cov(int size, float x_sub, float y_sub, float cov_xx, float cov_yy, float cov_xy);
    void plot_covariance(cv::Mat& img, const Human& human, const Detection::BBox& crop_info, const cv::Mat& colors);

public:
    static constexpr int max_num_timings = 10;
    std::vector<double> runtimes, feedback_delays;
    std::vector<int> runtimes_cnt, fb_delays_cnt;
    double runtime_det, runtime_det_thermal;
    int runtime_cnt_det, runtime_cnt_det_thermal;
};
