#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <vector>

#include "datasets.h"

#include "trt_common/BatchStream.h"
#include "trt_common/EntropyCalibrator.h"
#include "trt_common/argsParser.h"
#include "trt_common/buffers.h"
#include "trt_common/common.h"
#include "trt_common/logger.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <eigen3/Eigen/Core>
#include <opencv2/core.hpp>

struct Detection{
  int id;
  int label;
  float score;

  struct BBox{
    float ymin, xmin, ymax, xmax;
    float width() const { return xmax - xmin; }
    float height() const { return ymax - ymin; }
    float cx() const { return (xmin + xmax) / 2.0f; }
    float cy() const { return (ymin + ymax) / 2.0f; }
    float area() const { return width() * height(); }
    bool valid() const { return xmin <= xmax && ymin <= ymax; }
  } bbox;

  struct Keypoint{
    float x, y, conf;
  };
  std::vector<Keypoint> kps;
};

struct EngineParams{
  uint32_t batchSize;
  std::vector<uint32_t> batchSizes_pose;
  int32_t dlaCore;
  bool int8, fp16;
  std::vector<std::string> inputTensorNames, outputTensorNames;
  std::vector<std::string> inputTensorNames_pose, outputTensorNames_pose;
  std::vector<std::string> inputTensorNames_segm, outputTensorNames_segm;
  std::vector<std::string> calibDataDirs, calibDataDirs_pose;
};

class  PoseDetEngine{
public:
  template <typename T>
  using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

  PoseDetEngine(const std::string& model_path_pose, const std::string& model_path_det = "", const std::string& model_path_segm = "", bool h36m = false, uint32_t precision = 1, const std::string& calib_data_dir = "");

  // PoseDetEngine is neither copyable nor movable
  PoseDetEngine(const PoseDetEngine&) = delete;
  PoseDetEngine& operator=(const PoseDetEngine&) = delete;

  ~PoseDetEngine();

  void process_det_input(const cv::Mat& img);
  void process_segm_input(const cv::Mat& img, const samplesCommon::BufferManager& buffers);
  void process_pose_input(const cv::Mat& crop);
  void process_pose_input(const std::vector<cv::Mat> &crops, const int start_crop, const int batchsize, const int id);

  const std::vector<int>& det_input_tensor_shape() const {return input_tensor_shape_det;}
  const int& det_input_size() const {return input_size_det;}
  const std::vector<std::vector<int>>& det_output_tensor_shapes() const {return output_tensor_shapes_det;}
  const std::vector<int>& det_output_sizes() const {return output_sizes_det;}
  const std::vector<int>& pose_input_tensor_shape() const {return input_tensor_shape_pose;}
  const int& pose_input_size() const {return input_size_pose;}
  const std::vector<std::vector<int>>& pose_output_tensor_shapes() const {return output_tensor_shapes_pose;}
  const std::vector<int>& pose_output_sizes() const {return output_sizes_pose;}
  int pose_output_stride() const {return output_stride_pose;}
  const std::vector<std::vector<int>>& get_kps_symmetry() const {return kps_symmetry;}

  std::shared_ptr<nvinfer1::ICudaEngine> get_segm_engine() const {return mEngineSegm;}
  const std::vector<int>& get_segm_input_dims() const {return input_tensor_shape_segm;}
  const std::vector<int>& get_segm_output_dims() const {return output_tensor_shape_segm;}

  void invoke_det();
  void invoke_pose(const int batchsize = 1, const int id = 0);
  void invoke_segm(samplesCommon::BufferManager& buffers, SampleUniquePtr<IExecutionContext> &context);
  void get_det_result(std::vector<Detection>& res, float thresh, int top_k, const cv::Size& img_size = cv::Size(1.0f, 1.0f), int x0 = 0, int y0 = 0, float scale = 1.0f);
  void get_det_result(std::vector<Detection>& res, std::vector<Detection>& res_obj, float thresh, int top_k, const cv::Size& img_size = cv::Size(1.0f, 1.0f), int x0 = 0, int y0 = 0, float scale = 1.0f);
  void get_heatmaps(cv::Mat& heatmaps);
  void get_heatmaps(std::vector<cv::Mat> &heatmaps_vec, const int batchsize, const int id);
  void get_segmentation(cv::Mat& logits, const samplesCommon::BufferManager& buffers);
  void infer_batch_pose(std::vector<cv::Mat> &heatmaps_vec, const std::vector<cv::Mat> &crops);

  static constexpr double alpha_overlay = 0.70;
  static Eigen::Matrix<uint8_t, ADE20K_INDOOR::NUM_CLASSES, 3, Eigen::RowMajor> colormap_ade20k;
  static void def_colormap();

private:
  PoseDetEngine() = default;

  bool build_det(std::string model_path);
  bool build_segm(std::string model_path);
  bool build_pose(std::string model_path, const int id = 0);

  EngineParams mParams;
  std::shared_ptr<nvinfer1::ICudaEngine> mEngineDet, mEngineSegm; //!< The TensorRT engine used to run the network
  std::vector<std::shared_ptr<nvinfer1::ICudaEngine> > mEnginesPose;
  SampleUniquePtr<nvinfer1::IExecutionContext> mContextDet; //, mContextSegm;
  std::vector<SampleUniquePtr<nvinfer1::IExecutionContext> > mContextsPose;
  std::shared_ptr<samplesCommon::BufferManager> mBuffersDet; //, mBuffersSegm;
  std::vector<std::shared_ptr<samplesCommon::BufferManager> > mBuffersPose;
  nvinfer1::Dims mInputDims, mInputDimsPose, mOutputDimsPose, mInputDimsSegm, mOutputDimsSegm;
  std::vector<nvinfer1::Dims> mOutputDims;

  std::vector<int> input_tensor_shape_det, input_tensor_shape_pose, input_tensor_shape_segm, output_tensor_shape_segm;
  int input_size_det, input_size_pose;
  std::vector<std::vector<int>> output_tensor_shapes_det, output_tensor_shapes_pose;
  std::vector<int> output_sizes_det, output_sizes_pose;
  int output_stride_pose;
  std::vector<std::vector<int>> kps_symmetry;
  std::vector<std::string> class_names;
  bool segm_nhwc;
};

struct ThermalEngineParams{
  uint32_t batchSize;
  int32_t dlaCore;
  bool int8, fp16;
  std::vector<std::string> inputTensorNames, outputTensorNames;
  std::vector<std::string> calibDataDirs;
};

class  ThermalDetEngine{
public:
  template <typename T>
  using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

  ThermalDetEngine(const std::string& model_path_det, uint32_t precision = 1, const std::string& calib_data_dir = "");

  // ThermalDetEngine is neither copyable nor movable
  ThermalDetEngine(const PoseDetEngine&) = delete;
  ThermalDetEngine& operator=(const PoseDetEngine&) = delete;

  ~ThermalDetEngine();

  void process_det_input(const cv::Mat& img);

  const std::vector<int>& det_input_tensor_shape() const {return input_tensor_shape_det;}
  const int& det_input_size() const {return input_size_det;}
  const std::vector<std::vector<int>>& det_output_tensor_shapes() const {return output_tensor_shapes_det;}
  const std::vector<int>& det_output_sizes() const {return output_sizes_det;}

  void invoke_det();
  void get_det_result(std::vector<Detection>& res, float thresh, int top_k, const cv::Size& img_size = cv::Size(1.0f, 1.0f), int x0 = 0, int y0 = 0, float scale = 1.0f);

private:
  ThermalDetEngine() = default;

  bool build_det(std::string model_path);
  bool build_det_onnx(std::string model_path);

  ThermalEngineParams mParams;
  std::shared_ptr<nvinfer1::ICudaEngine> mEngineDet; //!< The TensorRT engine used to run the network
  SampleUniquePtr<nvinfer1::IExecutionContext> mContextDet;
  std::shared_ptr<samplesCommon::BufferManager> mBuffersDet;
  nvinfer1::Dims mInputDims;
  std::vector<nvinfer1::Dims> mOutputDims;
  bool mONNX;

  std::vector<int> input_tensor_shape_det;
  int input_size_det;
  std::vector<std::vector<int>> output_tensor_shapes_det;
  std::vector<int> output_sizes_det;
  std::vector<std::string> class_names;
};
