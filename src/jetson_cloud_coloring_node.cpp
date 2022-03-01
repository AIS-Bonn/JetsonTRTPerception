#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <edgetpu_segmentation_msgs/DetectionList.h>
#include <edgetpu_segmentation_msgs/BoundingBox.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Core>

#include <jetson_trt_pose/VoxelFilter.h>
#include <jetson_trt_pose/datasets.h>
#include <jetson_trt_pose/nanoflann.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
//#include <message_filters/sync_policies/approximate_time.h>

#include <string>
#include <vector>
#include <functional>
#include <chrono>

using std::string;
using std::cout;
using std::endl;

static Eigen::Matrix<uint8_t, Eigen::Dynamic, 3, Eigen::RowMajor> g_colormap;
static std::vector<int> g_det2segm_class;
static std::vector<int> g_class2pub_idx;

static bool g_flip = false;
static bool g_use_bbox_dets = true;
static int num_classes = 32;
static string g_color_frame, g_depth_frame, g_thermal_frame;
static cv::Size g_color_size, g_depth_size, g_thermal_size;

const float g_depth_scale = 1000.0;
const uint16_t g_min_depth = 300;
const uint16_t g_max_depth = 12000;
static float g_depth_vert_res = 0.005f;
static const string g_base_frame = "base";
static const float g_ground_plane_tolerance = 0.13f;
static Eigen::Vector4f g_ground_plane_coeffs;
static int g_class_idx_floor = -1;

static edgetpu_segmentation_msgs::DetectionList g_last_thermal_dets;

static sensor_msgs::PointField g_field_x, g_field_y, g_field_z, g_field_rgb, g_field_semantic;
static sensor_msgs::PointCloud2 g_cloud_xyz_rgb_semantic_initial, g_cloud_xyz_rgb_initial;

static Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor> g_pixels_depth;
static Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor> g_depth_pixel_rays;

typedef Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MapTypeFloat;
typedef Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>> MapTypeFloatRowVec;
typedef Eigen::Map<const Eigen::Matrix<uint16_t, 1, Eigen::Dynamic, Eigen::RowMajor>> MapTypeDepthConst;
typedef Eigen::Map<Eigen::Matrix<uint16_t, 1, Eigen::Dynamic, Eigen::RowMajor>> MapTypeDepth;

typedef Eigen::Map<const Eigen::Matrix<uint8_t, Eigen::Dynamic, 3, Eigen::RowMajor>> MapTypeRGBconst;
typedef Eigen::Map<Eigen::Matrix<uint8_t, Eigen::Dynamic, 3, Eigen::RowMajor>> MapTypeRGB;
typedef Eigen::Map<Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>> MapTypeGray;

typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix3Xf, 3, nanoflann::metric_L2_Simple, false> my_kd_tree_t;

bool init_dataset_dependent(int num_classes);

struct Config{
    //TODO: config server not available for publication. Using hardcoded default values instead..
    float voxelFilter_side_length = 20.f;
    float voxelFilter_voxel_length = 0.05f;
    bool use_outlier_filter = true;
    int outlier_meanKs = 10;
    float outlier_stdDevs = 0.5f;
//   config_server::Parameter<float> voxelFilter_side_length{"/voxel_filter/side_length", 0, 1, 100, 20};
//   config_server::Parameter<float> voxelFilter_voxel_length{"/voxel_filter/voxel_length", 0.01f, 0.01f, 1.f, 0.05f}; // default res: 5cm
// 
//   config_server::Parameter<bool> use_outlier_filter{"/outlier_filter/enable", false};
//   config_server::Parameter<int> outlier_meanKs{"/outlier_filter/mean_k", 2, 1, 100, 10};
//   config_server::Parameter<float> outlier_stdDevs{"/outlier_filter/std_dev_factor", 0.f, 0.01f, 10.f, 0.5f};
//   
// //   voxel_filter:
// //     side_length: 20
// //     voxel_length: 0.05
// //   outlier_filter:
// //     enable: 1
// //     mean_k: 10
// //     std_dev_factor: 0.5
};

struct PointRGB{
  union{
    struct{
      uint8_t b;
      uint8_t g;
      uint8_t r;
      uint8_t a;
    };
    float rgb;
  };
};

struct DetPointInfo{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3f pt;
  int ptIdx;
  float r;
  float w_det;
  int det_class;
  int segm_class;
};

struct DetPointInfoInit{
  int detPointIdx;
  float r;
  int det_class;
};

struct DetPointCloud{
  std::vector<DetPointInfo> pts;
  inline size_t kdtree_get_point_count() const { return pts.size(); }
  inline float kdtree_get_pt(const size_t idx, const size_t dim) const
  {
    if (dim == 0) return pts[idx].pt(0);
    else if (dim == 1) return pts[idx].pt(1);
    else return pts[idx].pt(2);
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, DetPointCloud> , DetPointCloud, 3 /* dim */> my_kd_tree_bbox_t;
void extractEuclideanCluster(const my_kd_tree_bbox_t& tree, float tolerance, int seed_idx, std::vector<bool>& idx_in_cluster){
  std::vector<bool> processed (tree.dataset.kdtree_get_point_count(), false);
  std::vector<std::pair<uint32_t, float> > nn_matches;

  std::vector<uint32_t> seed_queue;
  int sq_idx = 0;
  seed_queue.push_back(seed_idx);
  processed[seed_idx] = true;

  while (sq_idx < static_cast<int> (seed_queue.size ()))
  {
    // Search for sq_idx
    size_t n_matches = tree.radiusSearch(tree.dataset.pts[seed_queue[sq_idx]].pt.data(), tolerance, nn_matches, nanoflann::SearchParams()); // results are sorted
    if (n_matches <= 1)
    {
      sq_idx++;
      continue;
    }

    for (std::size_t j = 1; j < n_matches; ++j)             // results are sorted -> first point [idx 0] is query point -> start from result idx 1
    {
      if (processed[nn_matches[j].first])        // Has this point been processed before ?
        continue;

      // Perform a simple Euclidean clustering
      seed_queue.push_back (nn_matches[j].first);
      processed[nn_matches[j].first] = true;
    }

    sq_idx++;
  }

  idx_in_cluster = std::vector<bool>(tree.dataset.kdtree_get_point_count(), false);
  for (std::size_t j = 0; j < seed_queue.size (); ++j)
    idx_in_cluster[seed_queue[j]] = true;
}

template<int NUM_CLASSES>
Eigen::Matrix<float, NUM_CLASSES, 1> addLogProb(const Eigen::Matrix<float, NUM_CLASSES, 1>& prob_a, const Eigen::Matrix<float, NUM_CLASSES, 1>& prob_b){
  Eigen::Matrix<float, NUM_CLASSES, 1> prob_fused = prob_a + prob_b;

  typename Eigen::Matrix<float, NUM_CLASSES, 1>::Index maxIndex;
  const double log_max = prob_fused.maxCoeff(&maxIndex);
  double log_sum; // The max element is not part of the exp sum
  if (maxIndex == 0)
    log_sum = log_max + std::log1p(double((prob_fused.template tail<NUM_CLASSES-1>().array()-log_max).exp().sum()));
  else if (maxIndex == NUM_CLASSES-1)
    log_sum = log_max + std::log1p(double((prob_fused.template head<NUM_CLASSES-1>().array()-log_max).exp().sum()));
  else
    log_sum = log_max + std::log1p(double((prob_fused.segment(0,maxIndex).array()-log_max).exp().sum()) + double((prob_fused.segment(maxIndex+1,NUM_CLASSES-maxIndex-1).array()-log_max).exp().sum()));

  prob_fused.array() -= (float)log_sum;
  return prob_fused;
}

double bbox_iou(const edgetpu_segmentation_msgs::BoundingBox& bb1, const edgetpu_segmentation_msgs::BoundingBox& bb2){
  double x0 = std::max(bb1.xmin, bb2.xmin);
  double y0 = std::max(bb1.ymin, bb2.ymin);
  double x1 = std::min(bb1.xmax, bb2.xmax);
  double y1 = std::min(bb1.ymax, bb2.ymax);

  double interArea = std::max(0.0, x1 - x0) * std::max(0.0, y1 - y0);
  double bb1_area = std::max(0.0, bb1.xmax - bb1.xmin) * std::max(0.0, bb1.ymax - bb1.ymin);
  double bb2_area = std::max(0.0, bb2.xmax - bb2.xmin) * std::max(0.0, bb2.ymax - bb2.ymin);

  return interArea / (bb1_area + bb2_area - interArea + std::numeric_limits<double>::epsilon());
}

double bbox_overlap(const edgetpu_segmentation_msgs::BoundingBox& bb1, const edgetpu_segmentation_msgs::BoundingBox& bb2){
  double x0 = std::max(bb1.xmin, bb2.xmin);
  double y0 = std::max(bb1.ymin, bb2.ymin);
  double x1 = std::min(bb1.xmax, bb2.xmax);
  double y1 = std::min(bb1.ymax, bb2.ymax);

  double interArea = std::max(0.0, x1 - x0) * std::max(0.0, y1 - y0);
  double bb1_area = std::max(0.0, bb1.xmax - bb1.xmin) * std::max(0.0, bb1.ymax - bb1.ymin);

  return interArea / (bb1_area + std::numeric_limits<double>::epsilon());
}

double bbox_area(const edgetpu_segmentation_msgs::BoundingBox& bb){
  return std::max(0.0, bb.xmax - bb.xmin) * std::max(0.0, bb.ymax - bb.ymin);
}

void thermalDetsCallback(const edgetpu_segmentation_msgs::DetectionList& thermal_dets_msg){
    g_last_thermal_dets = thermal_dets_msg;
}

void generateStatistics (double& mean, double& variance, double& stddev, std::vector<float>& distances, const my_kd_tree_t& tree, const int& mean_k_, const Eigen::Matrix3Xf& cloud){

  // Allocate enough space to hold the result
  std::vector<size_t> nn_indexes(mean_k_);
  std::vector<float> nn_dists_sqr (mean_k_);

  const int num_points = cloud.cols();
  distances.resize (num_points);
  int valid_distances = 0;
  // Go over all the points and calculate the mean or smallest distance
  for (std::size_t cp = 0; cp < num_points; ++cp)
  {
    if(!cloud.col(cp).allFinite()){
      distances[cp] = 0;
      continue;
    }

    nanoflann::KNNResultSet<float> resultSet(mean_k_);
    resultSet.init(&nn_indexes[0], &nn_dists_sqr[0]);

    if ( !tree.index->findNeighbors(resultSet, cloud.col(cp).data(), nanoflann::SearchParams()))
    {
      distances[cp] = 0;
      ROS_WARN ("Searching for the closest %d neighbors failed.\n", mean_k_);
      continue;
    }

    // Minimum distance (if mean_k_ == 2) or mean distance
    double dist_sum = 0;
    for (int j = 1; j < mean_k_; ++j)
      dist_sum += sqrt(nn_dists_sqr[j]);
    distances[cp] = static_cast<float> (dist_sum / (mean_k_ - 1));
    valid_distances++;
  }

  // Estimate the mean and the standard deviation of the distance vector
  double sum = 0, sq_sum = 0;
  for (const float &distance : distances)
  {
    sum += distance;
    sq_sum += distance * distance;
  }

  mean = sum / static_cast<double>(valid_distances);
  variance = (sq_sum - sum * sum / static_cast<double>(valid_distances)) / (static_cast<double>(valid_distances) - 1);
  stddev = sqrt (variance);
}

template<int NUM_CLASSES>
void depthSemanticCallback(const sensor_msgs::ImageConstPtr& depth_msg, const sensor_msgs::ImageConstPtr& logits_msg, const edgetpu_segmentation_msgs::DetectionListConstPtr& rgb_dets_msg,
                           const Eigen::Matrix3d& K_color,  const Eigen::Matrix3d& K_thermal, const Eigen::Affine3d& depth_to_color, const Eigen::Affine3d& depth_to_thermal, Config& params,
                           const ros::Publisher& pub_cloud, const std::vector<ros::Publisher>& pubs_cloud_classes){
  typedef Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, NUM_CLASSES, Eigen::RowMajor> > MapTypeLogitsConst;
  typedef Eigen::Map<Eigen::Matrix<float, 1, NUM_CLASSES, Eigen::RowMajor> > MapVecLogits;
  typedef Eigen::Matrix<float, 1, NUM_CLASSES, Eigen::RowMajor> VecLogits;
  auto t0 = std::chrono::high_resolution_clock::now();

  cv_bridge::CvImageConstPtr cv_ptr_depth(new cv_bridge::CvImage);
  cv_bridge::CvImageConstPtr cv_ptr_logits(new cv_bridge::CvImage);
  cv::Mat depth, logits;
  try {
    cv_ptr_depth = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    cv_ptr_logits = cv_bridge::toCvShare(logits_msg);
    depth = cv_ptr_depth->image;
    logits = cv_ptr_logits->image;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s. aborting!", e.what());
    return;
  }

  const int num_points = depth.cols * depth.rows;
  const int num_classes = logits.channels();
  MapTypeLogitsConst logits_eigen(logits.ptr<float>(0), logits.cols * logits.rows, NUM_CLASSES);
  if(NUM_CLASSES != num_classes){
    ROS_ERROR("received logits num classes %d different from template num classes %d!", num_classes, NUM_CLASSES);
  }

  MapTypeDepthConst depth_map(depth.ptr<uint16_t>(0), 1, num_points);
  Eigen::Matrix<bool, 1, Eigen::Dynamic, Eigen::RowMajor> depth_mask = ((depth_map.array() > g_min_depth) && (depth_map.array() < g_max_depth));
  Eigen::Array<float, 1, Eigen::Dynamic, Eigen::RowMajor> depth_scaled = depth_map.cast<float>().array() / g_depth_scale;
  Eigen::Matrix3Xf points3D = g_depth_pixel_rays.array().rowwise() * depth_scaled; // Pixel-rays: 3 x HW, Depth: 1 x HW
  auto t1 = std::chrono::high_resolution_clock::now();

  //Voxel Filter
  auto voxel_filter = VoxelGrid::Create(params.voxelFilter_side_length, params.voxelFilter_voxel_length);
  voxel_filter->addCloud(points3D, depth_mask);
  Eigen::Matrix3Xf point3Dfiltered;
  voxel_filter->getFusedVoxel(point3Dfiltered);
  auto t2 = std::chrono::high_resolution_clock::now();

  //Outlier Filter
  if(params.use_outlier_filter){
    my_kd_tree_t tree(3, std::cref(point3Dfiltered), 10);
    tree.index->buildIndex();
    double mean, variance, stddev;
    std::vector<float> distances;
    generateStatistics(mean, variance, stddev, distances, tree, params.outlier_meanKs, point3Dfiltered);

    double const distance_threshold = mean + params.outlier_stdDevs * stddev; // a distance that is bigger than this signals an outlier
    const int num_points = distances.size();
    Eigen::Matrix3Xf points3DoutlierFiltered(3, num_points);
    int num_points_outlier_filtered = 0;
    for (int cp = 0; cp < num_points; ++cp){
      if(distances[cp] <= distance_threshold){ // inlier -> copy to output
        points3DoutlierFiltered.col(num_points_outlier_filtered) = point3Dfiltered.col(cp);
        ++num_points_outlier_filtered;
      }
    }

    //points3DoutlierFiltered.conservativeResize(Eigen::NoChange_t::NoChange, num_points_outlier_filtered);
    point3Dfiltered = points3DoutlierFiltered.leftCols(num_points_outlier_filtered);
  }

  const int num_points_filtered = point3Dfiltered.cols();
  const int point_step_semantic = g_cloud_xyz_rgb_semantic_initial.point_step;
  const int point_step_rgb = g_cloud_xyz_rgb_initial.point_step;

  // new cloud
  sensor_msgs::PointCloud2::Ptr semantic_cloud_msg ( new sensor_msgs::PointCloud2() );
  semantic_cloud_msg->header = depth_msg->header;
  semantic_cloud_msg->data.resize(num_points_filtered * point_step_semantic);
  semantic_cloud_msg->width = num_points_filtered;
  semantic_cloud_msg->height = 1;
  semantic_cloud_msg->fields = g_cloud_xyz_rgb_semantic_initial.fields;
  semantic_cloud_msg->is_bigendian = false;
  semantic_cloud_msg->point_step = point_step_semantic;
  semantic_cloud_msg->row_step = point_step_semantic * semantic_cloud_msg->width;
  semantic_cloud_msg->is_dense = false;

  const int num_clouds_classes = pubs_cloud_classes.size();
  std::vector<bool> clouds_classes_used(num_clouds_classes, false);
  std::vector<size_t> clouds_classes_ptIdx(num_clouds_classes, 0);
  std::vector<sensor_msgs::PointCloud2::Ptr> semantic_cloud_msgs_classes(num_clouds_classes, nullptr);
  for (int i = 0; i < num_clouds_classes; ++i){
    if(pubs_cloud_classes[i].getNumSubscribers() > 0){
      clouds_classes_used[i] = true;
      auto& cloud_msg = semantic_cloud_msgs_classes[i];
      cloud_msg.reset(new sensor_msgs::PointCloud2());
      cloud_msg->header = depth_msg->header;
      cloud_msg->data.resize(num_points_filtered * point_step_rgb);
      cloud_msg->width = num_points_filtered;
      cloud_msg->height = 1;
      cloud_msg->fields = g_cloud_xyz_rgb_initial.fields;
      cloud_msg->is_bigendian = false;
      cloud_msg->point_step = point_step_rgb;
      cloud_msg->row_step = point_step_rgb * cloud_msg->width;
      cloud_msg->is_dense = false;
    }
  }

  double scale_x = (double) (logits_msg->width - 1) /  (double) (g_color_size.width); // -1 to imitate tensorflow "align-corners" behaviour:
  double scale_y = (double) (logits_msg->height - 1) / (double) (g_color_size.height);
  Eigen::Matrix3d K_logits = K_color;
  K_logits.row(0) *= scale_x;
  K_logits.row(1) *= scale_y;
  float scale_x_dets = 1.f / (float) scale_x;
  float scale_y_dets = 1.f / (float) scale_y;

  Eigen::Matrix<uint8_t, 1, 3> rgb = Eigen::Matrix<uint8_t, 1, 3>::Zero();
  std::vector<DetPointCloud> dets_rgb_pts, dets_thermal_pts;
  std::vector<std::vector<DetPointInfoInit> > dets_rgb_pts_init, dets_thermal_pts_init;
  int n_rgb_dets = 0, n_thermal_dets = 0;
  bool transform_thermal = !g_last_thermal_dets.detections.empty();
  if(!rgb_dets_msg->detections.empty()){
    n_rgb_dets = rgb_dets_msg->detections.size();
    dets_rgb_pts.resize(n_rgb_dets);
    dets_rgb_pts_init.resize(n_rgb_dets);
  }
  if(transform_thermal){
    n_thermal_dets = g_last_thermal_dets.detections.size();
    dets_thermal_pts.resize(n_thermal_dets);
    dets_thermal_pts_init.resize(n_thermal_dets);
  }
  const float softmax_temperature = 0.25f;

  // do projection:
  for ( int ptIdx = 0; ptIdx < num_points_filtered; ++ptIdx ){
    const uint32_t point_start_semantic = semantic_cloud_msg->point_step * ptIdx;
    const Eigen::Vector3f pt_depth = point3Dfiltered.col(ptIdx);
    Eigen::Map<Eigen::Vector3f>(reinterpret_cast<float*>(&semantic_cloud_msg->data[point_start_semantic])) = pt_depth;

    //TODO: update ground-plane coeffs from obstacle-clustering node, if available..
    double dist_to_plane = std::abs(g_ground_plane_coeffs[0] * pt_depth.x() + g_ground_plane_coeffs[1] * pt_depth.y() + g_ground_plane_coeffs[2] * pt_depth.z() + g_ground_plane_coeffs[3]); // fabs(a * p.x + b * p.y + c * p.z + d)
    if(dist_to_plane < g_ground_plane_tolerance){ // fix ground plane points to ground class..
      VecLogits prob_floor;
      prob_floor.setZero();
      prob_floor(g_class_idx_floor) = 1.f;

      rgb = g_colormap.row(g_class_idx_floor);
      PointRGB pt_rgb;
      pt_rgb.r = rgb(0);
      pt_rgb.g = rgb(1);
      pt_rgb.b = rgb(2);
      pt_rgb.a = 0;
      *reinterpret_cast<PointRGB*>(&semantic_cloud_msg->data[point_start_semantic + g_field_rgb.offset]) = pt_rgb;
      memcpy(&semantic_cloud_msg->data[point_start_semantic + g_field_semantic.offset], prob_floor.data(), NUM_CLASSES * sizeof(float));

      if(g_class_idx_floor < g_class2pub_idx.size() && g_class2pub_idx[g_class_idx_floor] >= 0 && g_class2pub_idx[g_class_idx_floor] < num_clouds_classes){
        const int cloud_idx = g_class2pub_idx[g_class_idx_floor];
        if(clouds_classes_used[cloud_idx]){
          auto& cloud_msg = semantic_cloud_msgs_classes[cloud_idx];
          const uint32_t point_start_class = cloud_msg->point_step * clouds_classes_ptIdx[cloud_idx];
          Eigen::Map<Eigen::Vector3f>(reinterpret_cast<float*>(&cloud_msg->data[point_start_class])) = pt_depth;
          *reinterpret_cast<PointRGB*>(&cloud_msg->data[point_start_class + g_field_rgb.offset]) = pt_rgb;

          ++clouds_classes_ptIdx[cloud_idx];
        }
      }
    }

    else{
      Eigen::Vector2f kp_image = ((K_logits * depth_to_color).cast<float>() * pt_depth).hnormalized();
      Eigen::Vector2f kp_image_logits = g_flip ? Eigen::Vector2f(logits.cols, logits.rows) - kp_image : kp_image;

      typename VecLogits::Index maxIndex = -1;
      if ( kp_image_logits.x() >= 0 && kp_image_logits.x() <= logits.cols - 1 && kp_image_logits.y() >= 0 && kp_image_logits.y() <= logits.rows - 1)
      {
        //Bilinear Interpolation!
        const int v_up = static_cast<int>(std::floor(kp_image_logits.y()));
        const int v_down = static_cast<int>(std::ceil(kp_image_logits.y()));
        const int u_left = static_cast<int>(std::floor(kp_image_logits.x()));
        const int u_right = static_cast<int>(std::ceil(kp_image_logits.x()));
        const int idx_ul = v_up * logits.cols + u_left;
        const int idx_ur = v_up * logits.cols + u_right;
        const int idx_bl = v_down * logits.cols + u_left;
        const int idx_br = v_down * logits.cols + u_right;

        const float alpha_u = u_right - kp_image_logits.x();
        const float alpha_v = v_down - kp_image_logits.y();

        if(alpha_u < 0 || alpha_u > 1 || alpha_v < 0 || alpha_v > 1){
          ROS_ERROR("Interpolation factors out of bounds! u: %f, v: %f, alpha_u: %f, alpha_v: %f", kp_image_logits.x(), kp_image_logits.y(), alpha_u, alpha_v);
        }

        const VecLogits logit_up = alpha_u * logits_eigen.row(idx_ul) + (1.f - alpha_u) * logits_eigen.row(idx_ur);
        const VecLogits logit_down = alpha_u * logits_eigen.row(idx_bl) + (1.f - alpha_u) * logits_eigen.row(idx_br);
        VecLogits logit = alpha_v * logit_up + (1.f - alpha_v) * logit_down;
        float max = logit.maxCoeff(&maxIndex);

        //SoftMax -> Use confidence values as weights for fusion
        logit.array() -= max; // This is for numeric stability
        VecLogits logit_exp = (logit.array() * softmax_temperature).exp();
        logit_exp /= logit_exp.sum();

        rgb = g_colormap.row(maxIndex);
        PointRGB pt_rgb;
        pt_rgb.r = rgb(0);
        pt_rgb.g = rgb(1);
        pt_rgb.b = rgb(2);
        pt_rgb.a = 0;
        *reinterpret_cast<PointRGB*>(&semantic_cloud_msg->data[point_start_semantic + g_field_rgb.offset]) = pt_rgb;
        memcpy(&semantic_cloud_msg->data[point_start_semantic + g_field_semantic.offset], logit_exp.data(), NUM_CLASSES * sizeof(float));

        if(maxIndex < g_class2pub_idx.size() && g_class2pub_idx[maxIndex] >= 0 && g_class2pub_idx[maxIndex] < num_clouds_classes){
          const int cloud_idx = g_class2pub_idx[maxIndex];
          if(clouds_classes_used[cloud_idx]){
            auto& cloud_msg = semantic_cloud_msgs_classes[cloud_idx];
            const uint32_t point_start_class = cloud_msg->point_step * clouds_classes_ptIdx[cloud_idx];
            Eigen::Map<Eigen::Vector3f>(reinterpret_cast<float*>(&cloud_msg->data[point_start_class])) = pt_depth;
            *reinterpret_cast<PointRGB*>(&cloud_msg->data[point_start_class + g_field_rgb.offset]) = pt_rgb;

            ++clouds_classes_ptIdx[cloud_idx];
          }
        }

        if(!rgb_dets_msg->detections.empty() && g_use_bbox_dets){ // These are person + obj detections.
          float u_det = kp_image.x() * scale_x_dets; // abs coordinates in original 848x480 resolution
          float v_det = kp_image.y() * scale_y_dets; // abs coordinates in original 848x480 resolution

          int det_candidate_idx = -1;
          for(int det_idx = 0; det_idx < n_rgb_dets; ++det_idx){
            const auto& det = rgb_dets_msg->detections[det_idx];

            if(u_det > det.bbox.xmin && u_det < det.bbox.xmax && v_det > det.bbox.ymin && v_det < det.bbox.ymax){ // inside bbox
              if(det_candidate_idx < 0){
                det_candidate_idx = det_idx;
                if(g_det2segm_class[det.label] == ADE20K_INDOOR::person) // directly abort search, if point falls into person bbox
                    break;
              }
              else{
                const auto& det_candidate = rgb_dets_msg->detections[det_candidate_idx];
                if(g_det2segm_class[det.label] == ADE20K_INDOOR::person || (det.label != det_candidate.label && bbox_area(det.bbox) < bbox_area(det_candidate.bbox) && bbox_overlap(det.bbox, det_candidate.bbox) > 0.5)){ // new box should have person label or have different label and be smaller and inside first box (sub-part of the first box)
                  det_candidate_idx = det_idx;
                  if(g_det2segm_class[det.label] == ADE20K_INDOOR::person) // abort search, if point falls into person bbox
                    break;
                  //ROS_INFO("Updated bbox index! new class: %d, old class: %d", det.label, det_candidate.label);
                }
              }
            }
          }

          if(det_candidate_idx >= 0){
            const auto& det = rgb_dets_msg->detections[det_candidate_idx];
            float bbox_sigma_x = (float)(det.bbox.xmax - det.bbox.xmin) * 0.5f;
            float bbox_sigma_y = (float)(det.bbox.ymax - det.bbox.ymin) * 0.5f;
            float bbox_dx = (u_det - (float)(det.bbox.xmax + det.bbox.xmin) * 0.5f) / bbox_sigma_x; // (x - mu_x) / sigma_x
            float bbox_dy = (v_det - (float)(det.bbox.ymax + det.bbox.ymin) * 0.5f) / bbox_sigma_y; // (y - mu_y) / sigma_y
            float w_det = std::exp(-0.5f * (bbox_dx * bbox_dx + bbox_dy * bbox_dy)) * det.score;
            dets_rgb_pts[det_candidate_idx].pts.emplace_back(DetPointInfo{pt_depth, ptIdx, point3Dfiltered.col(ptIdx).norm(), w_det, det.label, (int)maxIndex});
            if(g_det2segm_class[det.label] == (int)maxIndex)
              dets_rgb_pts_init[det_candidate_idx].emplace_back(DetPointInfoInit{(int)dets_rgb_pts[det_candidate_idx].pts.size() - 1, point3Dfiltered.col(ptIdx).norm(), det.label});
          }
        }
      }
      else {
        std::fill(reinterpret_cast<float*>(&semantic_cloud_msg->data[point_start_semantic + g_field_semantic.offset]), reinterpret_cast<float*>(&semantic_cloud_msg->data[point_start_semantic + g_field_semantic.offset]) + NUM_CLASSES, 1.0f / NUM_CLASSES);
        PointRGB pt_rgb;
        pt_rgb.rgb = 0.0f;
        *reinterpret_cast<PointRGB*>(&semantic_cloud_msg->data[point_start_semantic + g_field_rgb.offset]) = pt_rgb;
      }

      if(transform_thermal && g_use_bbox_dets){
        Eigen::Vector2f kp_thermal = ((K_thermal * depth_to_thermal).cast<float>() * pt_depth).hnormalized();

        if ( kp_thermal.x() >= 0 && kp_thermal.x() <= g_thermal_size.width - 1 && kp_thermal.y() >= 0 && kp_thermal.y() <= g_thermal_size.height - 1){ // inside image
          for(int det_idx = 0; det_idx < n_thermal_dets; ++det_idx){
            const auto& det = g_last_thermal_dets.detections[det_idx];

            if(kp_thermal.x() > det.bbox.xmin && kp_thermal.x() < det.bbox.xmax && kp_thermal.y() > det.bbox.ymin && kp_thermal.y() < det.bbox.ymax){ // inside bbox
              const float bbox_sigma_x = (float)(det.bbox.xmax - det.bbox.xmin) * 0.75f;
              const float bbox_sigma_y = (float)(det.bbox.ymax - det.bbox.ymin) * 0.75f;
              const float bbox_dx = (kp_thermal.x() - (float)(det.bbox.xmax + det.bbox.xmin) * 0.5f) / bbox_sigma_x; // (x - mu_x) / sigma_x
              const float bbox_dy = (kp_thermal.y() - (float)(det.bbox.ymax + det.bbox.ymin) * 0.5f) / bbox_sigma_y; // (y - mu_y) / sigma_y
              const float w_det = std::exp(-0.5f * (bbox_dx * bbox_dx + bbox_dy * bbox_dy)) * det.score;
              dets_thermal_pts[det_idx].pts.emplace_back(DetPointInfo{pt_depth, ptIdx, point3Dfiltered.col(ptIdx).norm(), w_det, det.label, (int)maxIndex});
              if(g_det2segm_class[det.label] == (int)maxIndex)
                dets_thermal_pts_init[det_idx].emplace_back(DetPointInfoInit{(int)dets_thermal_pts[det_idx].pts.size() - 1, point3Dfiltered.col(ptIdx).norm(), det.label});
              break; //this will use the first (most confident) bbox the point falls in.
            }
          }
        }
      }
    }
  }

  auto t3 = std::chrono::high_resolution_clock::now();

  //Handle points inside rgb + thermal bounding boxes
  if(g_use_bbox_dets){
    static constexpr float cluster_tol_factor[14] = {0.f, 2.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f}; // {"NONE", "person", "cycle", "vehicle", "animal", "chair", "couch", "table", "tv", "laptop", "microwave", "oven", "fridge", "book"};
    std::vector<std::vector<DetPointCloud>* > bbox_pts_infos = {&dets_rgb_pts, &dets_thermal_pts};
    std::vector<std::vector<std::vector<DetPointInfoInit>>* > bbox_pts_init_infos = {&dets_rgb_pts_init, &dets_thermal_pts_init};
    static constexpr int range_quantile = 4; //[3] = {4, 4, 2}; // person, cycle, vehicle
    for(int det_type_idx = 0; det_type_idx < bbox_pts_infos.size(); ++det_type_idx){
      const int n_dets = bbox_pts_infos[det_type_idx]->size();
      for(int det_idx = 0; det_idx < n_dets; ++det_idx){
        auto& bbox_pts = (*bbox_pts_infos[det_type_idx])[det_idx];
        if(bbox_pts.pts.empty())
          continue;
        const int num_points = bbox_pts.pts.size();
        auto& bbox_pts_init = (*bbox_pts_init_infos[det_type_idx])[det_idx];
        float median_range = 0.f;
        int median_range_ptIdx = -1;
        if(bbox_pts_init.size() > 0.20 * num_points){ // there are enough overlapping points -> init median range only from points where det and segm. class agree.
          std::sort(bbox_pts_init.begin(), bbox_pts_init.end(), [](const DetPointInfoInit& a, const DetPointInfoInit& b){return a.r < b.r;}); // sort in ascending range
          median_range = bbox_pts_init[bbox_pts_init.size() / range_quantile].r; // 25% quantile
          median_range_ptIdx = bbox_pts_init[bbox_pts_init.size() / range_quantile].detPointIdx;
        }
        else{
          std::sort(bbox_pts.pts.begin(), bbox_pts.pts.end(), [](const DetPointInfo& a, const DetPointInfo& b){return a.r < b.r;}); // sort in ascending range
          median_range_ptIdx = num_points / range_quantile;
          median_range = bbox_pts.pts[median_range_ptIdx].r; // 25% quantile
        }

        const float cluster_tolerance = std::max(params.voxelFilter_voxel_length, g_depth_vert_res * median_range) * cluster_tol_factor[bbox_pts.pts[0].det_class]; // factor * vertical depth resolution at determined median range
        my_kd_tree_bbox_t tree(3, bbox_pts, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree.buildIndex();
        std::vector<bool> idx_in_cluster;
        extractEuclideanCluster(tree, cluster_tolerance*cluster_tolerance, median_range_ptIdx, idx_in_cluster);

        for (int pt_idx = 0; pt_idx < num_points; ++pt_idx) {
          const auto& pt = bbox_pts.pts[pt_idx];
          //std::cout << "ptIdx (rgb): " << pt.ptIdx << ", class: " << pt.det_class << ", range-delta: " << std::abs(pt.r - median_range) << std::endl;
          const uint32_t point_start_semantic = semantic_cloud_msg->point_step * pt.ptIdx;
          MapVecLogits logits_exp(reinterpret_cast<float*>(&semantic_cloud_msg->data[point_start_semantic + g_field_semantic.offset]));
          typename MapVecLogits::Index maxIndex_segm = pt.segm_class;
          if(maxIndex_segm < 0) // this should only be the case for a point in a thermal bounding box, which was somehow not labeled by color segmentation
            logits_exp.maxCoeff(&maxIndex_segm);

          if(idx_in_cluster[pt_idx]){
            if(pt.w_det > 1.f / NUM_CLASSES && (g_det2segm_class[pt.det_class] == ADE20K_INDOOR::person || maxIndex_segm != ADE20K_INDOOR::person)){ // don't overwrite segmented persons with other class...
//              const Eigen::Map<const Eigen::Vector3f> point ( reinterpret_cast<const float*>(&semantic_cloud_msg->data[point_start_semantic+offset_x]));
//              const float ground_plane_dist = point.dot(ground_plane_normal) + ground_plane_offset;
//              //std::cout << "ptIdx (rgb): " << pt.ptIdx << ", class: " << pt.det_class << ", ground-plane dist: " << ground_plane_dist << ", pt: " << point.transpose() << ", normal: " <<  ground_plane_normal.transpose() << ", offset: " << ground_plane_offset << std::endl;

              Eigen::Matrix<float, NUM_CLASSES, 1> logits_det_exp;
              logits_det_exp.setConstant((1 - pt.w_det)/ (NUM_CLASSES - 1)); // Distribute remaining probabilities mass equally between all other N-1 classes
              logits_det_exp(g_det2segm_class[pt.det_class]) = pt.w_det;
              logits_exp = addLogProb<NUM_CLASSES>(logits_exp.array().log(), logits_det_exp.array().log()).array().exp();

              typename MapVecLogits::Index maxIndex;
              logits_exp.maxCoeff(&maxIndex);

              if(maxIndex != maxIndex_segm){ // If ArgMax class changed through fusion with detection
                rgb = g_colormap.row(maxIndex);
                PointRGB pt_rgb;
                pt_rgb.r = rgb(0);
                pt_rgb.g = rgb(1);
                pt_rgb.b = rgb(2);
                pt_rgb.a = 0;
                *reinterpret_cast<PointRGB*>(&semantic_cloud_msg->data[point_start_semantic + g_field_rgb.offset]) = pt_rgb;

                if(maxIndex < g_class2pub_idx.size() && g_class2pub_idx[maxIndex] >= 0 && g_class2pub_idx[maxIndex] < num_clouds_classes){
                  const int cloud_idx = g_class2pub_idx[maxIndex];
                  const int initial_cloud_idx = maxIndex_segm < g_class2pub_idx.size() ? g_class2pub_idx[maxIndex_segm] : -1;
                  if(cloud_idx != initial_cloud_idx && clouds_classes_used[cloud_idx]){
                    auto& cloud_msg = semantic_cloud_msgs_classes[cloud_idx];
                    const uint32_t point_start_class = cloud_msg->point_step * clouds_classes_ptIdx[cloud_idx];
                    Eigen::Map<Eigen::Vector3f>(reinterpret_cast<float*>(&cloud_msg->data[point_start_class])) = Eigen::Map<Eigen::Vector3f>(reinterpret_cast<float*>(&semantic_cloud_msg->data[point_start_semantic]));
                    *reinterpret_cast<PointRGB*>(&cloud_msg->data[point_start_class + g_field_rgb.offset]) = pt_rgb;

                    ++clouds_classes_ptIdx[cloud_idx];
                  }
                }
              }
            }
          }
          else if (g_det2segm_class[pt.det_class] == maxIndex_segm){ // point excluded (background), reset to background
            PointRGB pt_rgb;
            logits_exp.setConstant(1.f / NUM_CLASSES);
            pt_rgb.rgb = 0.f;
            *reinterpret_cast<PointRGB*>(&semantic_cloud_msg->data[point_start_semantic + g_field_rgb.offset]) = pt_rgb;
          }
        }
      }
    }
  }

  auto t4 = std::chrono::high_resolution_clock::now();

  if(pub_cloud.getNumSubscribers() > 0)
    pub_cloud.publish(semantic_cloud_msg);

  for (int i = 0; i < num_clouds_classes; ++i){
    if(clouds_classes_used[i]){
      auto& cloud_msg = semantic_cloud_msgs_classes[i];
      cloud_msg->data.resize(clouds_classes_ptIdx[i] * point_step_rgb);
      cloud_msg->width = clouds_classes_ptIdx[i];
      cloud_msg->row_step = point_step_rgb * cloud_msg->width;

      pubs_cloud_classes[i].publish(cloud_msg);
    }
  }

  auto t5 = std::chrono::high_resolution_clock::now();

  long duration_proj3D = std::chrono::duration_cast<std::chrono::microseconds>( t1 - t0 ).count();
  long duration_filter = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  long duration_semantic = std::chrono::duration_cast<std::chrono::microseconds>( t3 - t2 ).count();
  long duration_bbox = std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();
  long duration_pub = std::chrono::duration_cast<std::chrono::microseconds>( t5 - t4 ).count();
  long duration_total = std::chrono::duration_cast<std::chrono::microseconds>( t5 - t0 ).count();

  cout << "Duration proj3D: " << duration_proj3D / 1000. << "ms (" << num_points << " points), duration filter: " << duration_filter / 1000. << "ms (" << num_points_filtered << " points remaining), duration semantic: " << duration_semantic / 1000. << "ms, duration bbox: " << duration_bbox / 1000. << "ms, duration pub: " << duration_pub / 1000. << "ms, duration total: " << duration_total / 1000. << "ms.\r";
  cout.flush();
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "jetson_cloud_coloring_node");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  
  nh_private.param("flip", g_flip, false);
  nh_private.param<bool>("use_bbox_dets", g_use_bbox_dets, true);
  nh_private.param("n_classes", num_classes, 32);
  std::string camera;
  nh_private.param<std::string>("camera", camera, "d455");

  Config config;

  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);
  
  string depth_topic = "/" + camera + "/depth/image_rect_raw";
  string depth_info_topic = "/" + camera + "/depth/camera_info";
  
  string logits_topic = "/" + camera + "/logits";
  string dets_rgb_topic = "/" + camera + "/dets_obj";
  string image_info_topic = "/" + camera + "/color/camera_info";
  
  string dets_thermal_topic = "/" + camera + "/dets_thermal";
  string thermal_info_topic = "/" + camera + "/lepton/camera_info";

  string semantic_cloud_topic = "/" + camera + "/cloud_coloring/semantic_cloud";
  std::vector<string> semantic_cloud_topics_classes = {"/" + camera + "/cloud_coloring/semantic_clouds/floor",
                                                       "/" + camera + "/cloud_coloring/semantic_clouds/wall",
                                                      "/" + camera + "/cloud_coloring/semantic_clouds/table",
                                                      "/" + camera + "/cloud_coloring/semantic_clouds/chair",
                                                      "/" + camera + "/cloud_coloring/semantic_clouds/computer",
                                                      "/" + camera + "/cloud_coloring/semantic_clouds/person",
                                                      "/" + camera + "/cloud_coloring/semantic_clouds/other"};

  message_filters::Subscriber<sensor_msgs::Image> sub_depth(nh, depth_topic, 1);
  message_filters::Subscriber<sensor_msgs::Image> sub_logits(nh, logits_topic, 1);
  message_filters::Subscriber<edgetpu_segmentation_msgs::DetectionList> sub_dets_rgb(nh, dets_rgb_topic, 1);
  ros::Subscriber sub_dets_thermal = nh.subscribe(dets_thermal_topic, 1, thermalDetsCallback);

  ros::Publisher pub_cloud = nh.advertise<sensor_msgs::PointCloud2>(semantic_cloud_topic, 1);
  std::vector<ros::Publisher> pubs_cloud_classes;
  pubs_cloud_classes.reserve(semantic_cloud_topics_classes.size());
  for(const auto& topic: semantic_cloud_topics_classes)
    pubs_cloud_classes.push_back(nh.advertise<sensor_msgs::PointCloud2>(topic, 1));

  init_dataset_dependent(num_classes);
  
  // get intrinsics
  ROS_INFO("Waiting for camera info message on topic \"%s\"...", depth_info_topic.c_str());
  const auto& caminfo_depth_msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(depth_info_topic, nh);
  const auto& caminfo_color_msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(image_info_topic, nh);
  const auto& caminfo_thermal_msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(thermal_info_topic, nh); //TODO: thermal calibration
  if(caminfo_depth_msg == nullptr || caminfo_color_msg == nullptr){
    ROS_ERROR("no camera info message received!");
    return -1;
  }

  Eigen::Matrix3d K_depth, K_color, K_thermal;
  K_depth << caminfo_depth_msg->K[0], 0.0, caminfo_depth_msg->K[2], 0.0, caminfo_depth_msg->K[4], caminfo_depth_msg->K[5], 0.0, 0.0, 1.0;
  K_color << caminfo_color_msg->K[0], 0.0, caminfo_color_msg->K[2], 0.0, caminfo_color_msg->K[4], caminfo_color_msg->K[5], 0.0, 0.0, 1.0;
  K_thermal << caminfo_thermal_msg->K[0], 0.0, caminfo_thermal_msg->K[2], 0.0, caminfo_thermal_msg->K[4], caminfo_thermal_msg->K[5], 0.0, 0.0, 1.0;
  g_depth_size = cv::Size(caminfo_depth_msg->width, caminfo_depth_msg->height);
  g_color_size = cv::Size(caminfo_color_msg->width, caminfo_color_msg->height);
  g_thermal_size = cv::Size(caminfo_thermal_msg->width, caminfo_thermal_msg->height);
  g_depth_vert_res = 58.f * M_PIf32 / 180 / g_depth_size.height; // 58 deg. vertical FoV

  cout << "Received depth camera intrinsics: size: (" << g_depth_size.width << ", " << g_depth_size.height << ")" << endl << K_depth << endl;
  cout << "Received color camera intrinsics: size: (" << g_color_size.width << ", " << g_color_size.height << ")" << endl << K_color << endl;
  cout << "Received thermal camera intrinsics: size: (" << g_thermal_size.width << ", " << g_thermal_size.height << ")" << endl << K_thermal << endl;

  g_pixels_depth = Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor>(2, g_depth_size.width * g_depth_size.height); // shape = 2 x HW
  for (int u = 0; u < g_depth_size.width; ++u) {
    for (int v = 0; v < g_depth_size.height; ++v) {
      g_pixels_depth.col(v * g_depth_size.width + u) = Eigen::Vector2f(u, v);
    }
  }

  g_depth_pixel_rays = K_depth.cast<float>().inverse() * g_pixels_depth.colwise().homogeneous(); // shape 3 x HW // .homogenous(): (u,v) -> (u, v, 1)
  
  g_depth_frame = caminfo_depth_msg->header.frame_id;
  g_color_frame = caminfo_color_msg->header.frame_id;
  g_thermal_frame = caminfo_thermal_msg->header.frame_id;
  ROS_INFO("depth frame: %s, color frame: %s, thermal frame: %s.", g_depth_frame.c_str(), g_color_frame.c_str(), g_thermal_frame.c_str());

  Eigen::Affine3d depth_to_color_eigen = Eigen::Affine3d::Identity();
  Eigen::Affine3d depth_to_thermal_eigen = Eigen::Affine3d::Identity();
  Eigen::Affine3d sensor_to_base_eigen = Eigen::Affine3d::Identity();
  geometry_msgs::TransformStamped depth_to_color, depth_to_thermal, depth_to_base;
  while(1){
    try {
        depth_to_color = tfBuffer.lookupTransform(g_color_frame, g_depth_frame, ros::Time(0), ros::Duration(1.0));
        depth_to_thermal = tfBuffer.lookupTransform(g_thermal_frame, g_depth_frame, ros::Time(0), ros::Duration(1.0));
        depth_to_base = tfBuffer.lookupTransform(g_base_frame, g_depth_frame, ros::Time(0), ros::Duration(1.0));
    }
    catch(tf2::TransformException &ex) {
        ROS_ERROR("%s",ex.what());
        continue;
    }
    break;
  }

  depth_to_color_eigen = tf2::transformToEigen(depth_to_color);
  depth_to_thermal_eigen = tf2::transformToEigen(depth_to_thermal);
  cout << "Depth to color extrinsics: translation: " << depth_to_color_eigen.translation().transpose() << endl << "rotation: " << endl << depth_to_color_eigen.linear() << endl;
  cout << "Depth to thermal extrinsics: translation: " << depth_to_thermal_eigen.translation().transpose() << endl << "rotation: " << endl << depth_to_thermal_eigen.linear() << endl;

  sensor_to_base_eigen = tf2::transformToEigen(depth_to_base);
  Eigen::Vector3d normal_base(0,0,1);
  Eigen::Vector3d normal_sensor = sensor_to_base_eigen.inverse().linear() * normal_base;
  g_ground_plane_coeffs.head<3>() = normal_sensor.cast<float>();
  g_ground_plane_coeffs(3) = static_cast<float>(sensor_to_base_eigen.translation().z());
  if(camera.find("_3") != string::npos || camera.find("_4") != string::npos)
    g_ground_plane_coeffs(3) += 0.1; // TODO why is this necessary ? - calibration issue ?!
  cout << "Ground plane coeffs: " << g_ground_plane_coeffs.transpose() << endl;
  
//  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, edgetpu_segmentation_msgs::DetectionList> mySyncPolicy;
//  message_filters::Synchronizer<mySyncPolicy> sync(mySyncPolicy(5), sub_depth, sub_logits, sub_dets_rgb);
  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, edgetpu_segmentation_msgs::DetectionList> sync(sub_depth, sub_logits, sub_dets_rgb, 5);

  switch (num_classes) {
  case ADE20K_INDOOR::NUM_CLASSES:
    sync.registerCallback(std::bind(depthSemanticCallback<ADE20K_INDOOR::NUM_CLASSES>, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::cref(K_color), std::cref(K_thermal), std::cref(depth_to_color_eigen), std::cref(depth_to_thermal_eigen), std::ref(config), std::cref(pub_cloud), std::cref(pubs_cloud_classes)));
    break;
  default:
    return -1;
  }
  
  ROS_INFO("spinning...");
  ros::spin();
}
  
bool init_dataset_dependent(int num_classes){
  g_colormap = Eigen::Matrix<uint8_t, Eigen::Dynamic, 3, Eigen::RowMajor>::Zero(num_classes, 3);

  g_field_x.name = "x";               g_field_x.datatype = sensor_msgs::PointField::FLOAT32;        g_field_x.count = 1;                  g_field_x.offset = 0;
  g_field_y.name = "y";               g_field_y.datatype = sensor_msgs::PointField::FLOAT32;        g_field_y.count = 1;                  g_field_y.offset = 4;
  g_field_z.name = "z";               g_field_z.datatype = sensor_msgs::PointField::FLOAT32;        g_field_z.count = 1;                  g_field_z.offset = 8;
  g_field_rgb.name = "rgb";           g_field_rgb.datatype = sensor_msgs::PointField::FLOAT32;      g_field_rgb.count = 1;                g_field_rgb.offset = 12;
  g_field_semantic.name = "semantic"; g_field_semantic.datatype = sensor_msgs::PointField::FLOAT32; g_field_semantic.count = num_classes; g_field_semantic.offset = 16;
  g_cloud_xyz_rgb_semantic_initial.fields = {g_field_x, g_field_y, g_field_z, g_field_rgb, g_field_semantic};
  g_cloud_xyz_rgb_semantic_initial.point_step = (4 + num_classes) * sizeof(float);
  g_cloud_xyz_rgb_initial.fields = {g_field_x, g_field_y, g_field_z, g_field_rgb};
  g_cloud_xyz_rgb_initial.point_step = 4 * sizeof(float);

  switch(num_classes){
  case ADE20K_INDOOR::NUM_CLASSES:
    ROS_INFO("Model-Type: ADE20K-Indoor: %d classes", num_classes);
    //DETECTOR class_names = {"NONE", "person", "cycle", "vehicle", "animal", "chair", "couch", "table", "tv", "laptop", "microwave", "oven", "fridge", "book"};
    //SEGMENTATION:     background,     wall,    floor,    ceiling,    window,    door,    column,    stairs,    table,    chair,    seat,    cabinet,    shelf,    lamp,    person,    animal,    vehicle,    bike,    poster,    box,    book,    toy,    fridge,    dishwasher,    oven,    trashbin,    computer,    TV,    screen,   glass,      bottle,    food

    g_colormap.row( 0) = Eigen::Matrix<uint8_t, 1, 3>(0, 0, 0); //  0 background (0)
    g_colormap.row( 1) = Eigen::Matrix<uint8_t, 1, 3>(217, 83, 25); //  1 wall (1, 19) ## 19
    g_colormap.row( 2) = Eigen::Matrix<uint8_t, 1, 3>(158, 158, 158); //  2 floor (4, 29, 41, 102) # 41, 102
    g_colormap.row( 3) = Eigen::Matrix<uint8_t, 1, 3>(0, 114, 189); //  3 ceiling (6)
    g_colormap.row( 4) = Eigen::Matrix<uint8_t, 1, 3>(128, 64, 0); //  4 window (9)
    g_colormap.row( 5) = Eigen::Matrix<uint8_t, 1, 3>(255, 255, 64); //  5 door (15, 59)
    g_colormap.row( 6) = Eigen::Matrix<uint8_t, 1, 3>(217, 83, 25); //  6 column (41, 43, 94)
    g_colormap.row( 7) = Eigen::Matrix<uint8_t, 1, 3>(162, 20, 47); //  7 stairs (54, 60, 97, 122)
    g_colormap.row( 8) = Eigen::Matrix<uint8_t, 1, 3>(222,184,135);//  8 table (16, 34, 46, 57, 65, 71, 78) ## 78
    g_colormap.row( 9) = Eigen::Matrix<uint8_t, 1, 3>(126, 47, 142);//  9 chair (20, 31, 76, 111)
    g_colormap.row(10) = Eigen::Matrix<uint8_t, 1, 3>(126, 47, 142);//  10 seat (8, 24, 32, 40, 58, 70, 98, 132) ## 8, 58, 132 ---> chair
    g_colormap.row(11) = Eigen::Matrix<uint8_t, 1, 3>(222,184,135);//  11 cabinet (11, 36, 45, 74, 100) ---> table
    g_colormap.row(12) = Eigen::Matrix<uint8_t, 1, 3>(222,184,135);//  12 shelf (25, 63) ---> table
    g_colormap.row(13) = Eigen::Matrix<uint8_t, 1, 3>(77, 190, 238); //  13 lamp (37, 83, 86, 88, 135, 137) # 88, 137
    g_colormap.row(14) = Eigen::Matrix<uint8_t, 1, 3>(128, 128, 0); //  14 person (13)
    g_colormap.row(15) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);   // 15 animal --> other
    g_colormap.row(16) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250); //  16 vehicle (21, 81, 84, 103) --> other
    g_colormap.row(17) = Eigen::Matrix<uint8_t, 1, 3>(196, 64, 128); //  17 bike (117, 128)
    g_colormap.row(18) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250); //  18 poster (23, 101, 124, 145, 149, 150) # 124, 149, 150 --> other
    g_colormap.row(19) = Eigen::Matrix<uint8_t, 1, 3>(127,255,0);//  19 box (42, 112, 56, 113, 116, 126, 136); #112, 126, 136
    g_colormap.row(20) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);//  20 book (68) --> other
    g_colormap.row(21) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);//  21 toy --> other
    g_colormap.row(22) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);//  22 fridge --> other
    g_colormap.row(23) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);//  23 dishwasher --> other
    g_colormap.row(24) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);//  24 oven --> other
    g_colormap.row(25) = Eigen::Matrix<uint8_t, 1, 3>(0, 128, 128);//  25 trashbin
    g_colormap.row(26) = Eigen::Matrix<uint8_t, 1, 3>(255,0,255);//  26 computer (75)
    g_colormap.row(27) = Eigen::Matrix<uint8_t, 1, 3>(255,0,255);//  27 TV (75) --> computer
    g_colormap.row(28) = Eigen::Matrix<uint8_t, 1, 3>(255,0,255);//  28 screen (90, 131, 142, 144) --> computer
    g_colormap.row(29) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);//  29 glass --> other
    g_colormap.row(30) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);//  30 bottle --> other
    g_colormap.row(31) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);//  31 food (121) --> other

    g_det2segm_class = {0, 14, 17, 16, 15, 9, 10, 8, 27, 26, 24, 24, 22, 20};
    g_class2pub_idx = {-1, 1, 0, 6, 1, 1, 1, 6, 2, 3, 3, 2, 2, 6, 5, 6, 6, 6, 2, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 6, 6, 6}; // floor, wall, table, chair, computer, person, other
    g_class_idx_floor = 2;
    break;

    //TODO Reduced (output) classes:
//    g_colormap.row( 0) = Eigen::Matrix<uint8_t, 1, 3>(0, 0, 0); //  0 background (0)
//    g_colormap.row( 1) = Eigen::Matrix<uint8_t, 1, 3>(217, 83, 25); //  1 wall (1, 19) + column
//    g_colormap.row( 2) = Eigen::Matrix<uint8_t, 1, 3>(158, 158, 158); //  2 floor (4, 29, 41, 102)
//    g_colormap.row( 3) = Eigen::Matrix<uint8_t, 1, 3>(0, 114, 189); //  3 ceiling (6) + (13) lamp
//    g_colormap.row( 4) = Eigen::Matrix<uint8_t, 1, 3>(128, 64, 0); //  4 window (9)
//    g_colormap.row( 5) = Eigen::Matrix<uint8_t, 1, 3>(255, 255, 64); //  5 door (15, 59)
//    g_colormap.row( 6) = Eigen::Matrix<uint8_t, 1, 3>(162, 20, 47); //  7 stairs (54, 60, 97, 122)
//    g_colormap.row( 7) = Eigen::Matrix<uint8_t, 1, 3>(222,184,135);//  8 table (16, 34, 46, 57, 65, 71, 78) + 11 (cabinet) + 12 (shelf)
//    g_colormap.row( 8) = Eigen::Matrix<uint8_t, 1, 3>(126, 47, 142);//  9 chair (20, 31, 76, 111) + 10 (seat)
//    g_colormap.row( 9) = Eigen::Matrix<uint8_t, 1, 3>(77, 190, 238);  13 lamp
//    g_colormap.row(10) = Eigen::Matrix<uint8_t, 1, 3>(128, 128, 0); //  14 person (13)
//    g_colormap.row(11) = Eigen::Matrix<uint8_t, 1, 3>(196, 64, 128); //  17 bike (117, 128)
//    g_colormap.row(12) = Eigen::Matrix<uint8_t, 1, 3>(127,255,0);//  19 box (42, 112, 56, 113, 116, 126, 136); #112, 126, 13
//    g_colormap.row(13) = Eigen::Matrix<uint8_t, 1, 3>(0, 128, 128);//  25 trashbin
//    g_colormap.row(14) = Eigen::Matrix<uint8_t, 1, 3>(255,0,255);//  26 computer (75) + TV + screen
//    g_colormap.row(15) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250); // other...
  
  default:
    ROS_ERROR("Unsupported number of classes %d!", num_classes);
    return false;
  }
  
  return true;
}
