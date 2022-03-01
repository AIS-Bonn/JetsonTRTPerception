#include "jetson_trt_pose/pose_det_engine_jetson.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/filesystem.hpp>

#include "NvUffParser.h"
#include "NvOnnxParser.h"

#include <iostream>
#include <queue>
#include <algorithm>

using std::string;

namespace sample
{
Logger gLogger{Logger::Severity::kINFO};
LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLogger)};
LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
LogStreamConsumer gLogWarning{LOG_WARN(gLogger)};
LogStreamConsumer gLogError{LOG_ERROR(gLogger)};
LogStreamConsumer gLogFatal{LOG_FATAL(gLogger)};

void setReportableSeverity(Logger::Severity severity)
{
    gLogger.setReportableSeverity(severity);
    gLogVerbose.setReportableSeverity(severity);
    gLogInfo.setReportableSeverity(severity);
    gLogWarning.setReportableSeverity(severity);
    gLogError.setReportableSeverity(severity);
    gLogFatal.setReportableSeverity(severity);
}
} // namespace sample

namespace {
struct DetComparator {
  bool operator()(const Detection& lhs, const Detection& rhs) const {
    return std::tie(lhs.score, lhs.id) > std::tie(rhs.score, rhs.id);
  }
};

std::string to_string(const nvinfer1::Dims &d)
{
    std::string s;
    for (int64_t i = 0; i < d.nbDims; i++) {
        if (!s.empty()) { s += ", "; }
        s += std::to_string(d.d[i]);
    }
    return "(" + s + ")";
}

std::string to_string(const nvinfer1::DataType dtype)
{
    return std::to_string(int(dtype));
}

class MyBatchStream : public IBatchStream
{
public:
    MyBatchStream(int batchSize, nvinfer1::Dims dims, std::vector<std::string> directory, bool rgb = false, bool nhwc = false, int increment = 1)
        : mBatchSize(batchSize > 0 ? batchSize : dims.d[0])
        , mIncrement(increment > 0 ? increment : 1)
        , mRGB(rgb)
        , mNHWC(nhwc)
        , mDims(dims)
        , mDataDir(directory)
    {
        mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);

        // get all .jpg files from directory
        for (const auto& dir : directory){
          sample::gLogInfo << "Calibration files directory: " << dir << std::endl;
          if (boost::filesystem::exists(dir) && boost::filesystem::is_directory(dir)){
            for (auto const & entry : boost::filesystem::directory_iterator(dir)){
              if (boost::filesystem::is_regular_file(entry) && (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")){
                mFilenames.push_back(entry.path().string());
              }
            }
          }
        }

        mMaxBatches = mFilenames.size() / mIncrement / mBatchSize;
        reset(0);

        sample::gLogInfo << "got " << mFilenames.size() << " calibration images (" << mMaxBatches << " batches)." << std::endl;
    }

    // Resets data members
    void reset(int firstBatch) override
    {
        mBatchCount = 0;
        mImgCount = 0;
        skip(firstBatch);
    }

    // Advance to next batch and return true, or return false if there is no batch left.
    bool next() override
    {
        if (mBatchCount == mMaxBatches || mImgCount >= mFilenames.size())
        {
            return false;
        }

        const int32_t inputC = mDims.d[mNHWC ? 3 : 1];
        const int32_t inputH = mDims.d[mNHWC ? 1 : 2];
        const int32_t inputW = mDims.d[mNHWC ? 2 : 3];

        for (int32_t i = 0, volImg = inputC * inputH * inputW; i < mBatchSize && mImgCount < mFilenames.size(); ++i)
        {
            sample::gLogInfo << "Calibration: reading: " << mFilenames[mImgCount]  << "; (Image " << mImgCount + 1 << " / " << mFilenames.size() << ", Batch " << mBatchCount + 1 << " / " << mMaxBatches << ")" << std::endl;
            cv::Mat img = cv::imread(mFilenames[mImgCount]);
            if(mRGB)
              cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            cv::resize(img, img, cv::Size(inputW, inputH));

            if(mNHWC){
              cv::Mat normalized;
              img.convertTo(normalized, CV_32F, 2.0 / 255.0, -1.0);
              std::memcpy((mBatch.data() + i * volImg), normalized.data, volImg * sizeof (float));
            }
            else{
              for (int32_t c = 0; c < inputC; ++c)
              {
                  for (uint32_t j = 0, volChl = inputH * inputW; j < volChl; ++j)
                  {
                      mBatch[i * volImg + c * volChl + j]
                          = (2.0f / 255.0f) * float(img.data[j * inputC + c]) - 1.0f;
                  }
              }
            }

            mImgCount += mIncrement;
        }

        mBatchCount++;
        return true;
    }

    // Skips the batches
    void skip(int skipCount) override
    {
        int x = mBatchCount;
        for (int i = 0; i < skipCount; i++)
        {
            next();
        }
        mBatchCount = x;
    }

    float* getBatch() override
    {
        return mBatch.data();
    }

    float* getLabels() override
    {
        return mLabels.data();
    }

    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    int getBatchSize() const override
    {
        return mBatchSize;
    }

    nvinfer1::Dims getDims() const override
    {
        return mDims;
    }

private:
    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};
    int mImgCount{0};
    int mImageSize{0};
    int mIncrement{1};
    bool mRGB{false};
    bool mNHWC{false};
    std::vector<float> mBatch;         //!< Data for the batch
    std::vector<float> mLabels;        //!< Labels for the batch
    nvinfer1::Dims mDims;              //!< Input dimensions
    std::vector<std::string> mFilenames, mDataDir; //!< Directories where the files can be found
};

}

Eigen::Matrix<uint8_t, ADE20K_INDOOR::NUM_CLASSES, 3, Eigen::RowMajor> PoseDetEngine::colormap_ade20k;

PoseDetEngine::PoseDetEngine(const string& model_path_pose, const string& model_path_det, const std::string& model_path_segm, bool h36m, uint32_t precision, const std::string &calib_data_dir) {

  mParams.inputTensorNames.push_back("Input");
  mParams.outputTensorNames.push_back("NMS");
  mParams.outputTensorNames.push_back("NMS_1");
  mParams.inputTensorNames_pose.push_back("Input");
  mParams.outputTensorNames_pose.push_back("tower_0/out/BiasAdd");

  mParams.dlaCore = (precision & 0x4) ? 0 : -1;
  mParams.int8 = (precision & 0x2);
  mParams.fp16 = (precision & 0x1);

  mParams.batchSize = 1;
  if(mParams.dlaCore < 0)
    mParams.batchSizes_pose = {3, 7, 15};
  else
    mParams.batchSizes_pose = {3, 7};
  const int num_models_pose = mParams.batchSizes_pose.size();

  if(mParams.int8){
    if(calib_data_dir == "")
      sample::gLogError << "Using INT8 precision but no calib data directory given!" << std::endl;
    mParams.calibDataDirs.push_back(calib_data_dir);
    mParams.calibDataDirs_pose.push_back(calib_data_dir + "/crops");
  }

  if(mParams.int8)
    sample::gLogInfo << "Using INT8 precision." << std::endl;
  else if (mParams.fp16)
    sample::gLogInfo << "Using FP16 precision." << std::endl;
  else
    sample::gLogInfo << "Using FP32 precision." << std::endl;
  if(mParams.dlaCore >= 0)
    sample::gLogInfo << "Using DLA." << std::endl;

  initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

  mEnginesPose.resize(num_models_pose);
  mBuffersPose.resize(num_models_pose);
  mContextsPose.resize(num_models_pose);
  for (int id = 0; id < num_models_pose; ++id) {
    build_pose(model_path_pose, id);
    mBuffersPose[id] = std::make_shared<samplesCommon::BufferManager>(mEnginesPose[id], mParams.batchSizes_pose[id]);
    mContextsPose[id] = SampleUniquePtr<nvinfer1::IExecutionContext>(mEnginesPose[id]->createExecutionContext());
  }


  if(model_path_det != ""){
    build_det(model_path_det);
    mBuffersDet = std::make_shared<samplesCommon::BufferManager>(mEngineDet, mParams.batchSize);
    mContextDet =  SampleUniquePtr<nvinfer1::IExecutionContext>(mEngineDet->createExecutionContext());
  }

  if(model_path_segm != ""){
    build_segm(model_path_segm);
    //initialize Buffers and Context in inference thread
  }

  assert(mInputDimsPose.nbDims == 3);
  input_size_pose = 1;
  input_tensor_shape_pose.resize(mInputDimsPose.nbDims + 1);
  input_tensor_shape_pose[0] = 1; // TODO: Batch-Size pose. Expected to be in NHWC order!
  input_tensor_shape_pose[1] = mInputDimsPose.d[1];
  input_tensor_shape_pose[2] = mInputDimsPose.d[2];
  input_tensor_shape_pose[3] = mInputDimsPose.d[0];
  for (int i = 0; i < mInputDimsPose.nbDims; ++i) {
    input_size_pose *= mInputDimsPose.d[i];
  }

  output_sizes_pose.push_back(1);
  output_tensor_shapes_pose.resize(1);
  output_tensor_shapes_pose[0].resize(mOutputDimsPose.nbDims + 1);
  output_tensor_shapes_pose[0][0] = 1; // TODO: Batch-Size pose
  for (int i = 0; i < mOutputDimsPose.nbDims; ++i) {
    output_tensor_shapes_pose[0][i+1] = mOutputDimsPose.d[i];
    output_sizes_pose[0] *= mOutputDimsPose.d[i];
  }

  output_stride_pose = input_tensor_shape_pose[1] / output_tensor_shapes_pose[0][1];

  if(model_path_det != ""){
    assert(mInputDims.nbDims == 3);
    input_size_det = 1;
    input_tensor_shape_det.resize(mInputDims.nbDims + 1);
    input_tensor_shape_det[0] = 1; //Batch-Size Det. Expected to be in NHWC order!
    input_tensor_shape_det[1] = mInputDims.d[1];
    input_tensor_shape_det[2] = mInputDims.d[2];
    input_tensor_shape_det[3] = mInputDims.d[0];
    for (int i = 0; i < mInputDims.nbDims; ++i) {
      input_size_det *= mInputDims.d[i];
    }

    output_tensor_shapes_det.resize(mOutputDims.size());
    output_sizes_det.resize(mOutputDims.size());
    for (int j = 0; j < mOutputDims.size(); ++j) {
      output_sizes_det[j] = 1;
      output_tensor_shapes_det[j].resize(mOutputDims[j].nbDims + 1);
      output_tensor_shapes_det[j][0] = 1; // Batch-Size Det
      for (int i = 0; i < mOutputDims[j].nbDims; ++i) {
        output_tensor_shapes_det[j][i+1] = mOutputDims[j].d[i];
        output_sizes_det[j] *= mOutputDims[j].d[i];
      }
    }
  }

  if(model_path_segm != ""){
    assert(mInputDimsSegm.nbDims == 4);
    input_tensor_shape_segm.resize(mInputDimsSegm.nbDims);
    input_tensor_shape_segm[0] = mInputDimsSegm.d[0]; //Expected to be in NHWC order!
    input_tensor_shape_segm[1] = mInputDimsSegm.d[segm_nhwc ? 1 : 2];
    input_tensor_shape_segm[2] = mInputDimsSegm.d[segm_nhwc ? 2 : 3];
    input_tensor_shape_segm[3] = mInputDimsSegm.d[segm_nhwc ? 3 : 1];

    output_tensor_shape_segm.resize(mOutputDimsSegm.nbDims);
    output_tensor_shape_segm[0] = mOutputDimsSegm.d[0]; //Expected to be in NHWC order!
    output_tensor_shape_segm[1] = mOutputDimsSegm.d[1];
    output_tensor_shape_segm[2] = mOutputDimsSegm.d[2];
    output_tensor_shape_segm[3] = mOutputDimsSegm.d[3];
  }

  if(h36m)
    kps_symmetry = {{5, 6}, {7, 8}, {9, 10}, {11, 12}, {13, 14}, {15, 16}};
  else
    kps_symmetry = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}, {13, 14}, {15, 16}};

  class_names = {"NONE", "person", "cycle", "vehicle", "animal", "chair", "couch", "table", "tv", "laptop", "microwave", "oven", "fridge", "book"};
}

bool PoseDetEngine::build_det(std::string model_path){
  auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));

  std::stringstream gieModelStream;
  gieModelStream.seekg(0, gieModelStream.beg);

  std::stringstream cache_path_ss;
  cache_path_ss << model_path << "_" << mParams.batchSize << "_" << (mParams.int8 ? "uint8" : (mParams.fp16 ? "fp16" : "fp32")) << ".engine";
  std::string cache_path = cache_path_ss.str();
  std::ifstream cache( cache_path.c_str() );

  if(!cache){
      auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork()); //builder->createNetworkV2(0U)
      auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
      auto parser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());

      parser->registerInput(mParams.inputTensorNames[0].c_str(), DimsCHW(3, 480, 848), nvuffparser::UffInputOrder::kNCHW);
      parser->registerOutput(mParams.outputTensorNames[0].c_str());

      auto parsed = parser->parse(model_path.c_str(), *network, DataType::kFLOAT);
      if (!parsed)
      {
          return false;
      }

      sample::gLogInfo << "Detector batch size: " << mParams.batchSize << std::endl;

      builder->setMaxBatchSize(mParams.batchSize);
      config->setMaxWorkspaceSize(1_GiB); //TODO: more memory may enable better optimizations, increase workspace size, if GPU memory is available and TRT warnings about insufficient workspace occur.
      if (mParams.fp16)
      {
          config->setFlag(BuilderFlag::kFP16);
      }

      // Calibrator life time needs to last until after the engine is built.
      std::unique_ptr<IInt8Calibrator> calibrator;

      if (mParams.int8)
      {
        sample::gLogInfo << "Using Entropy Calibrator 2" << std::endl;
        const int32_t imageC = 3;
        const int32_t imageH = 480;
        const int32_t imageW = 848;
        const int32_t batchsize = mParams.batchSize;
        const nvinfer1::DimsNCHW imageDims{batchsize, imageC, imageH, imageW};
        MyBatchStream calibrationStream(batchsize, imageDims, mParams.calibDataDirs, true);
        calibrator.reset(new Int8EntropyCalibrator2<MyBatchStream>(
            calibrationStream, 0, "UffSSD", mParams.inputTensorNames[0].c_str()));
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
      }

      // No DLA for detector as results are very bad...
//      sample::gLogInfo << "Default Device Type: " << int(config->getDefaultDeviceType()) << std::endl;
//      if(builder->getNbDLACores() > 0 && mParams.dlaCore >= 0)
//      {
//          samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore + 1);
//          sample::gLogInfo << "Using DLA core: " << mParams.dlaCore + 1 << std::endl;
//      }

      mEngineDet = std::shared_ptr<nvinfer1::ICudaEngine>(
          builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
      if (!mEngineDet)
      {
          return false;
      }

      assert(network->getNbInputs() == 1);
      mInputDims = network->getInput(0)->getDimensions();
      assert(mInputDims.nbDims == 3);
      assert(network->getNbOutputs() == 2);
      mOutputDims.push_back(network->getOutput(0)->getDimensions());
      mOutputDims.push_back(network->getOutput(1)->getDimensions());

      nvinfer1::IHostMemory* serMem = mEngineDet->serialize();

      if( !serMem ){
          sample::gLogInfo << "Failed to serialize CUDA engine" << std::endl;
          return true;
      }

      gieModelStream.write((const char*)serMem->data(), serMem->size());

      sample::gLogInfo << "writing engine cache to: " << cache_path << std::endl;;

      std::ofstream outFile;
      outFile.open(cache_path);
      outFile << gieModelStream.rdbuf();
      outFile.close();
      gieModelStream.seekg(0, gieModelStream.beg);
      sample::gLogInfo << "Completed writing engine cache." << std::endl;

      return true;
  }
  else{

      sample::gLogInfo << "loading network profile from engine cache: " << cache_path << std::endl;
      gieModelStream << cache.rdbuf();
      cache.close();

      nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
      if (!infer){
          sample::gLogInfo <<  "Failed to create Infer Runtime" << std::endl;;
          return false;
      }

      gieModelStream.seekg(0, std::ios::end);
      const int modelSize = gieModelStream.tellg();
      gieModelStream.seekg(0, std::ios::beg);

      void* modelMem = malloc(modelSize);

      if( !modelMem ){
          sample::gLogError << "Failed to allocate memory to deserialize buffer" << std::endl;
          return false;
      }

      gieModelStream.read((char*)modelMem, modelSize);
      sample::gLogInfo << "deserializing..." << std::endl;
      mEngineDet = std::shared_ptr<nvinfer1::ICudaEngine>(infer->deserializeCudaEngine(modelMem, modelSize), samplesCommon::InferDeleter());
      free(modelMem);
      if (!mEngineDet)
      {
          return false;
      }
      sample::gLogInfo << "Success" << std::endl;

      const int n = mEngineDet->getNbBindings();
      for (int i = 0; i < n; ++i) {
          const nvinfer1::Dims dims = mEngineDet->getBindingDimensions(i);
          const nvinfer1::DataType dtype = mEngineDet->getBindingDataType(i);
          const std::string name(mEngineDet->getBindingName(i));
          const bool input = mEngineDet->bindingIsInput(i);
          if(input)
              mInputDims = dims;
          else
              mOutputDims.push_back(dims);

          sample::gLogInfo << "binding " << i << ":" << " name: " << name << ", is input: " << input << ", type: " << to_string(dtype) << " " << to_string(dims) << std::endl;
      }
      assert(mInputDims.nbDims == 3);

      return true;
  }
}

bool PoseDetEngine::build_segm(std::string model_path){
  std::stringstream gieModelStream;
  gieModelStream.seekg(0, gieModelStream.beg);
  std::stringstream cache_path_ss;
  cache_path_ss << model_path << "_" << mParams.batchSize << "_" << (mParams.int8 ? "uint8" : (mParams.fp16 ? "fp16" : "fp32")) << ".engine";
  std::string cache_path = cache_path_ss.str();
  std::ifstream cache( cache_path.c_str() );

  if(!cache){
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));

    auto parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    auto input_dims = network->getInput(0)->getDimensions();
    if(input_dims.nbDims != 4){
      sample::gLogError << "Expected 4-dimensional input tensor but got " << input_dims.nbDims << ". Aborting!" << std::endl;
      return false;
    }

    if(input_dims.d[1] == 3){
      segm_nhwc = false;
      sample::gLogInfo << "Detected NCHW segmentation model input" << std::endl;
    }
    else if (input_dims.d[3] == 3){
      segm_nhwc = true;
      sample::gLogInfo << "Detected NHWC segmentation model input" << std::endl;
    }
    else {
      sample::gLogError << "Expected 3-channel input for segmentation model but got " << input_dims.d[1] << "(NCHW), resp. " << input_dims.d[3] << "(NHWC). Aborting!" << std::endl;
      return false;
    }

    sample::gLogInfo << "Original input dimensions: [" << input_dims.d[0] << ", " << input_dims.d[1] << ", " << input_dims.d[2] << ", " << input_dims.d[3] << "]" << std::endl;
    if(segm_nhwc)
      network->getInput(0)->setDimensions(Dims4{1, 481, 849, 3}); // For inference: context.setBindingDimensions(0, Dims4{1, 481, 849, 3})
    else
      network->getInput(0)->setDimensions(Dims4{1, 3, 481, 849}); // For inference: context.setBindingDimensions(0, Dims4{1, 3, 481, 849})

    input_dims = network->getInput(0)->getDimensions();
    sample::gLogInfo << "Fixed input dimensions: [" << input_dims.d[0] << ", " << input_dims.d[1] << ", " << input_dims.d[2] << ", " << input_dims.d[3] << "]" << std::endl;

    assert(network->getNbInputs() == 1);
    assert(input_dims.nbDims == 4);
    mParams.inputTensorNames_segm.push_back(network->getInput(0)->getName());
    mInputDimsSegm = input_dims;

    assert(network->getNbOutputs() == 1);
    mParams.outputTensorNames_segm.push_back(network->getOutput(0)->getName());

    sample::gLogInfo << "Batch size: " << mParams.batchSize << std::endl;
    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(1_GiB); //TODO: more memory may enable better optimizations, increase workspace size, if GPU memory is available and TRT warnings about insufficient workspace occur.

    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

    if (mParams.int8)
    {
      sample::gLogInfo << "Using Entropy Calibrator 2" << std::endl;
      int slash_idx = model_path.rfind('/');
      int dot_idx = model_path.rfind('.');
      std::string file_suffix = model_path.substr(slash_idx + 1, dot_idx - slash_idx - 1);
      MyBatchStream calibrationStream(0, mInputDimsSegm, mParams.calibDataDirs, true, segm_nhwc);
      calibrator.reset(new Int8EntropyCalibrator2<MyBatchStream>(
          calibrationStream, 0, ("UffSegm_" + file_suffix).c_str(), mParams.inputTensorNames_segm[0].c_str()));
      config->setFlag(BuilderFlag::kINT8);
      config->setInt8Calibrator(calibrator.get());
    }

    // Segmentation doesn't work with dla...
//    sample::gLogInfo << "Default Device Type: " << int(config->getDefaultDeviceType()) << std::endl;
//    if(builder->getNbDLACores() > 0 && mParams.dlaCore >= 0)
//    {
//        samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
//        sample::gLogInfo << "Using DLA core: " << mParams.dlaCore << std::endl;
//    }

    mEngineSegm = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngineSegm)
    {
        return false;
    }

    auto output_dims = mEngineSegm->getBindingDimensions(1);
    sample::gLogInfo << "Engine output dimensions: [" << output_dims.d[0] << ", " << output_dims.d[1] << ", " << output_dims.d[2] << ", " << output_dims.d[3] << "]" << std::endl;
    assert(output_dims.nbDims == 4);
    mOutputDimsSegm = output_dims;

    nvinfer1::IHostMemory* serMem = mEngineSegm->serialize();

    if( !serMem ){
        sample::gLogInfo << "Failed to serialize CUDA engine" << std::endl;
        return true;
    }

    gieModelStream.write((const char*)serMem->data(), serMem->size());

    sample::gLogInfo << "writing engine cache to: " << cache_path << std::endl;;

    std::ofstream outFile;
    outFile.open(cache_path);
    outFile << gieModelStream.rdbuf();
    outFile.close();
    gieModelStream.seekg(0, gieModelStream.beg);
    sample::gLogInfo << "Completed writing engine cache." << std::endl;

    return true;
  }
  else{

      sample::gLogInfo << "loading network profile from engine cache: " << cache_path << std::endl;
      gieModelStream << cache.rdbuf();
      cache.close();

      nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
      if (!infer){
          sample::gLogInfo <<  "Failed to create Infer Runtime" << std::endl;;
          return false;
      }

      gieModelStream.seekg(0, std::ios::end);
      const int modelSize = gieModelStream.tellg();
      gieModelStream.seekg(0, std::ios::beg);

      void* modelMem = malloc(modelSize);

      if( !modelMem ){
          sample::gLogInfo << "Failed to allocate memory to deserialize buffer" << std::endl;
          return false;
      }

      gieModelStream.read((char*)modelMem, modelSize);
      sample::gLogInfo << "deserializing..." << std::endl;
      mEngineSegm = std::shared_ptr<nvinfer1::ICudaEngine>(infer->deserializeCudaEngine(modelMem, modelSize), samplesCommon::InferDeleter());
      free(modelMem);
      if (!mEngineSegm)
      {
          return false;
      }
      sample::gLogInfo << "Success" << std::endl;

      const int n = mEngineSegm->getNbBindings();
      for (int i = 0; i < n; ++i) {
          const nvinfer1::Dims dims = mEngineSegm->getBindingDimensions(i);
          const nvinfer1::DataType dtype = mEngineSegm->getBindingDataType(i);
          const std::string name(mEngineSegm->getBindingName(i));
          const bool input = mEngineSegm->bindingIsInput(i);
          if(input){
              mParams.inputTensorNames_segm.push_back(name);
              mInputDimsSegm = dims;
          }
          else {
            mParams.outputTensorNames_segm.push_back(name);
            mOutputDimsSegm = dims;
          }

          sample::gLogInfo << "binding " << i << ":" << " name: " << name << ", is input: " << input << ", type: " << to_string(dtype) << " " << to_string(dims) << std::endl;
      }
      assert(mInputDimsSegm.nbDims == 4);

      if(mInputDimsSegm.nbDims != 4){
        sample::gLogError << "Expected 4-dimensional input tensor but got " << mInputDimsSegm.nbDims << ". Aborting!" << std::endl;
        return false;
      }

      if(mInputDimsSegm.d[1] == 3){
        segm_nhwc = false;
        sample::gLogInfo << "Detected NCHW segmentation model input" << std::endl;
      }
      else if (mInputDimsSegm.d[3] == 3){
        segm_nhwc = true;
        sample::gLogInfo << "Detected NHWC segmentation model input" << std::endl;
      }
      else {
        sample::gLogError << "Expected 3-channel input for segmentation model but got " << mInputDimsSegm.d[1] << "(NCHW), resp. " << mInputDimsSegm.d[3] << "(NHWC). Aborting!" << std::endl;
        return false;
      }

      return true;
  }
}

bool PoseDetEngine::build_pose(std::string model_path, const int id){
  auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
  sample::gLogInfo << "Has " << builder->getNbDLACores() << " DLA cores, DLA max batchsize: " << builder->getMaxDLABatchSize() << std::endl;

  std::stringstream gieModelStream;
  gieModelStream.seekg(0, gieModelStream.beg);

  std::stringstream cache_path_ss;
  cache_path_ss << model_path << "_" << mParams.batchSizes_pose[id] << "_" << (mParams.int8 ? "uint8" : (mParams.fp16 ? "fp16" : "fp32")) << ((builder->getNbDLACores() > 0 && mParams.dlaCore >= 0) ? "_DLA" + std::to_string(mParams.dlaCore) : "") << ".engine";
  std::string cache_path = cache_path_ss.str();
  std::ifstream cache( cache_path.c_str() );

  if(!cache){
      auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork()); //builder->createNetworkV2(0U)
      auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
      auto parser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());

      parser->registerInput(mParams.inputTensorNames_pose[0].c_str(), DimsCHW(3, 256, 192), nvuffparser::UffInputOrder::kNCHW);
      parser->registerOutput(mParams.outputTensorNames_pose[0].c_str());

      auto parsed = parser->parse(model_path.c_str(), *network, DataType::kFLOAT);
      if (!parsed)
      {
          return false;
      }

      sample::gLogInfo << "Batch size: " << mParams.batchSizes_pose[id] << std::endl;

      builder->setMaxBatchSize(mParams.batchSizes_pose[id]);
      config->setMaxWorkspaceSize(768_MiB); //TODO: more memory may enable better optimizations, increase workspace size, if GPU memory is available and TRT warnings about insufficient workspace occur.
      if (mParams.fp16)
      {
          config->setFlag(BuilderFlag::kFP16);
      }

      // Calibrator life time needs to last until after the engine is built.
      std::unique_ptr<IInt8Calibrator> calibrator;

      if (mParams.int8)
      {
        sample::gLogInfo << "Using Entropy Calibrator 2" << std::endl;
        const int32_t imageC = 3;
        const int32_t imageH = 256;
        const int32_t imageW = 192;
        const int32_t batchsize = mParams.batchSizes_pose[id];
        nvinfer1::DimsNCHW imageDims{batchsize, imageC, imageH, imageW};
        MyBatchStream calibrationStream(batchsize, imageDims, mParams.calibDataDirs_pose, false);
        calibrator.reset(new Int8EntropyCalibrator2<MyBatchStream>(
            calibrationStream, 0, ("UffPose_" + std::to_string(batchsize)).c_str(), mParams.inputTensorNames_pose[0].c_str()));
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
      }

      sample::gLogInfo << "Default Device Type: " << int(config->getDefaultDeviceType()) << std::endl;
      if(builder->getNbDLACores() > 0 && mParams.dlaCore >= 0)
      {
          samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
          sample::gLogInfo << "Using DLA core: " << mParams.dlaCore << std::endl;
      }

      mEnginesPose[id] = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
      if (!mEnginesPose[id])
      {
          return false;
      }

      assert(network->getNbInputs() == 1);
      mInputDimsPose = network->getInput(0)->getDimensions();
      assert(mInputDims.nbDims == 3);
      assert(network->getNbOutputs() == 1);
      mOutputDimsPose = network->getOutput(0)->getDimensions();

      nvinfer1::IHostMemory* serMem = mEnginesPose[id]->serialize();

      if( !serMem ){
          sample::gLogInfo << "Failed to serialize CUDA engine" << std::endl;
          return true;
      }

      gieModelStream.write((const char*)serMem->data(), serMem->size());

      sample::gLogInfo << "writing engine cache to: " << cache_path << std::endl;;

      std::ofstream outFile;
      outFile.open(cache_path);
      outFile << gieModelStream.rdbuf();
      outFile.close();
      gieModelStream.seekg(0, gieModelStream.beg);
      sample::gLogInfo << "Completed writing engine cache." << std::endl;

      return true;
  }
  else{

      sample::gLogInfo << "loading network profile from engine cache: " << cache_path << std::endl;
      gieModelStream << cache.rdbuf();
      cache.close();

      nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
      if (!infer){
          sample::gLogInfo <<  "Failed to create Infer Runtime" << std::endl;;
          return false;
      }

      gieModelStream.seekg(0, std::ios::end);
      const int modelSize = gieModelStream.tellg();
      gieModelStream.seekg(0, std::ios::beg);

      void* modelMem = malloc(modelSize);

      if( !modelMem ){
          sample::gLogInfo << "Failed to allocate memory to deserialize buffer" << std::endl;
          return false;
      }

      gieModelStream.read((char*)modelMem, modelSize);
      sample::gLogInfo << "deserializing..." << std::endl;
      mEnginesPose[id] = std::shared_ptr<nvinfer1::ICudaEngine>(infer->deserializeCudaEngine(modelMem, modelSize), samplesCommon::InferDeleter());
      free(modelMem);
      if (!mEnginesPose[id])
      {
          return false;
      }
      sample::gLogInfo << "Success" << std::endl;

      const int n = mEnginesPose[id]->getNbBindings();
      for (int i = 0; i < n; ++i) {
          const nvinfer1::Dims dims = mEnginesPose[id]->getBindingDimensions(i);
          const nvinfer1::DataType dtype = mEnginesPose[id]->getBindingDataType(i);
          const std::string name(mEnginesPose[id]->getBindingName(i));
          const bool input = mEnginesPose[id]->bindingIsInput(i);
          if(input)
              mInputDimsPose = dims;
          else
              mOutputDimsPose = dims;

          sample::gLogInfo << "binding " << i << ":" << " name: " << name << ", is input: " << input << ", type: " << to_string(dtype) << " " << to_string(dims) << std::endl;
      }
      assert(mInputDims.nbDims == 3);

      return true;
  }
}

PoseDetEngine::~PoseDetEngine(){
  nvuffparser::shutdownProtobufLibrary();
}

void PoseDetEngine::process_det_input(const cv::Mat& img){
  const int32_t inputC = mInputDims.d[0];
  const int32_t inputH = mInputDims.d[1];
  const int32_t inputW = mInputDims.d[2];

  float* hostDataBuffer = static_cast<float*>(mBuffersDet->getHostBuffer(mParams.inputTensorNames[0])); // Host memory for input buffer

  //Batch-Size 1
  for (int32_t c = 0; c < inputC; ++c)
  {
      // The color image to input should be in RGB order
      for (uint32_t j = 0, volChl = inputH * inputW; j < volChl; ++j)
      {
          hostDataBuffer[c * volChl + j]
              = (2.0f / 255.0f) * float(img.data[j * inputC + c]) - 1.0f;
      }
  }
}

void PoseDetEngine::process_segm_input(const cv::Mat &img, const samplesCommon::BufferManager &buffers){
  const int32_t inputC = mInputDimsSegm.d[segm_nhwc ? 3 : 1];
  const int32_t inputH = mInputDimsSegm.d[segm_nhwc ? 1 : 2];
  const int32_t inputW = mInputDimsSegm.d[segm_nhwc ? 2 : 3];
  const int32_t batchSize = mInputDimsSegm.d[0];
  if(batchSize != 1){
    sample::gLogError << "Segmentation batchsize expected to be 1 but is: " << batchSize << ". Will only run inference on one input image!" << std::endl;
  }

  float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames_segm[0])); // Host memory for input buffer

  if(segm_nhwc){
    cv::Mat normalized(img.rows, img.cols, CV_32FC(img.channels()), (void*)(hostDataBuffer));
    img.convertTo(normalized, CV_32F, 2.0 / 255.0, -1.0);
  }
  else {
    for (int32_t c = 0; c < inputC; ++c)
    {
        // The color image to input should be in RGB order
        for (uint32_t j = 0, volChl = inputH * inputW; j < volChl; ++j)
        {
            hostDataBuffer[c * volChl + j]
                = (2.0f / 255.0f) * float(img.data[j * inputC + c]) - 1.0f;
        }
    }
  }
}

void PoseDetEngine::process_pose_input(const cv::Mat& crop){
  const int32_t inputC = mInputDimsPose.d[0];
  const int32_t inputH = mInputDimsPose.d[1];
  const int32_t inputW = mInputDimsPose.d[2];

  float* hostDataBuffer = static_cast<float*>(mBuffersPose[0]->getHostBuffer(mParams.inputTensorNames_pose[0])); // Host memory for input buffer

  //Batch-Size 1
  for (int32_t c = 0; c < inputC; ++c)
  {
      // The color image to input should be in RGB order -> channels inversed to BGR
      for (uint32_t j = 0, volChl = inputH * inputW; j < volChl; ++j)
      {
          hostDataBuffer[c * volChl + j]
              = (2.0f / 255.0f) * float(crop.data[j * inputC + inputC - c - 1]) - 1.0f;
      }
  }
}

void PoseDetEngine::process_pose_input(const std::vector<cv::Mat> &crops, const int start_crop, const int batchsize, const int id){
  const int32_t inputC = mInputDimsPose.d[0];
  const int32_t inputH = mInputDimsPose.d[1];
  const int32_t inputW = mInputDimsPose.d[2];

  //assert(batchsize <= mParams.batchSizes_pose[id]);
  if(batchsize > mParams.batchSizes_pose[id]){
    sample::gLogError << "Error: batchsize " << batchsize << " > max. batchsize of model: " << mParams.batchSizes_pose[id] << std::endl;
    return;
  }

  //assert(start_crop + batchsize <= crops.size());
  if(start_crop + batchsize > crops.size()){
    sample::gLogError << "Error: max crop index " << start_crop + batchsize << " > number of crops: " << crops.size() << std::endl;
    return;
  }

  float* hostDataBuffer = static_cast<float*>(mBuffersPose[id]->getHostBuffer(mParams.inputTensorNames_pose[0]));
  // Host memory for input buffer
  for (int32_t i = 0, volImg = inputC * inputH * inputW; i < batchsize; ++i)
  {
      for (int32_t c = 0; c < inputC; ++c)
      {
          // The color image to input should be in RGB order -> channels inversed to BGR
          for (uint32_t j = 0, volChl = inputH * inputW; j < volChl; ++j)
          {
              hostDataBuffer[i * volImg + c * volChl + j]
                  = (2.0f / 255.0f) * float(crops[start_crop + i].data[j * inputC + inputC - c - 1]) - 1.0f;
          }
      }
  }
}

void PoseDetEngine::invoke_det(){
  mBuffersDet->copyInputToDevice();
  const bool status = mContextDet->execute(mParams.batchSize, mBuffersDet->getDeviceBindings().data());
  if (!status)
      sample::gLogError << "Detector inference failed!" << std::endl;

  mBuffersDet->copyOutputToHost();
}

void PoseDetEngine::invoke_pose(const int batchsize, const int id){
  mBuffersPose[id]->copyInputToDevice();
  const bool status = mContextsPose[id]->execute((mParams.int8 || mParams.dlaCore >= 0) ? mParams.batchSizes_pose[id] : batchsize, mBuffersPose[id]->getDeviceBindings().data());
  if (!status){
    sample::gLogError << "Pose inference failed! (batchsize: " << ((mParams.int8 || mParams.dlaCore >= 0) ? mParams.batchSizes_pose[id] : batchsize) << ", id: " << id << ")" << std::endl;
  }

  mBuffersPose[id]->copyOutputToHost();
}

void PoseDetEngine::invoke_segm(samplesCommon::BufferManager& buffers, SampleUniquePtr<nvinfer1::IExecutionContext> &context){
  // Memcpy from host input buffers to device input buffers
  buffers.copyInputToDevice();

  const bool status = context->executeV2(buffers.getDeviceBindings().data());
  if (!status){
      sample::gLogError << "Segmentation inference failed!" << std::endl;
  }

  buffers.copyOutputToHost();
}

void PoseDetEngine::get_det_result(std::vector<Detection>& res, float thresh, int top_k, const cv::Size& img_size, int x0, int y0, float scale){
  const float* detectionOut = static_cast<const float*>(mBuffersDet->getHostBuffer(mParams.outputTensorNames[0]));
  const int32_t* keepCount = static_cast<const int32_t*>(mBuffersDet->getHostBuffer(mParams.outputTensorNames[1]));
  int n_det = keepCount[0];

   std::priority_queue<Detection, std::vector<Detection>, DetComparator> q_person;
   for (int i = 0; i < n_det; ++i) {
     // Output format for each detection is stored in the order: [image_id, label, confidence, xmin, ymin, xmax, ymax]
     const float* det = &detectionOut[0] + i * 7;

     const int label = std::lround(det[1]);
     if(label == 1){ // 1 = person
       const int id = i; //std::lround(det[0]);
       const float score = det[2];
       if (score < thresh) continue;

       const float xmin = (std::max(0.0f, det[3]) * img_size.width  - x0) / scale;
       const float ymin = (std::max(0.0f, det[4]) * img_size.height - y0) / scale;
       const float xmax = (std::min(1.0f, det[5]) * img_size.width  - x0) / scale;
       const float ymax = (std::min(1.0f, det[6]) * img_size.height - y0) / scale;

       q_person.push(Detection{id, label, score, Detection::BBox{ymin, xmin, ymax, xmax}, std::vector<Detection::Keypoint>()});
       if (q_person.size() > top_k) q_person.pop();
     }
   }

   res.clear();
   res.reserve(q_person.size());
   while (!q_person.empty()) {
       res.push_back(q_person.top());
       q_person.pop();
   }
   std::reverse(res.begin(), res.end());
}

void PoseDetEngine::get_det_result(std::vector<Detection>& res, std::vector<Detection>& res_obj, float thresh, int top_k, const cv::Size& img_size, int x0, int y0, float scale){
  const float* detectionOut = static_cast<const float*>(mBuffersDet->getHostBuffer(mParams.outputTensorNames[0]));
  const int32_t* keepCount = static_cast<const int32_t*>(mBuffersDet->getHostBuffer(mParams.outputTensorNames[1]));
  int n_det = keepCount[0];

   std::priority_queue<Detection, std::vector<Detection>, DetComparator> q_person, q_object;
   for (int i = 0; i < n_det; ++i) {
     // Output format for each detection is stored in the order [image_id, label, confidence, xmin, ymin, xmax, ymax]
     const float* det = &detectionOut[0] + i * 7;

     const int id = i; //std::lround(det[0]);
     const int label = std::lround(det[1]);
     const float score = det[2];
     if (score < thresh) continue;

     const float xmin = (std::max(0.0f, det[3]) * img_size.width  - x0) / scale;
     const float ymin = (std::max(0.0f, det[4]) * img_size.height - y0) / scale;
     const float xmax = (std::min(1.0f, det[5]) * img_size.width  - x0) / scale;
     const float ymax = (std::min(1.0f, det[6]) * img_size.height - y0) / scale;

     if(label == 1){ // 1 = person
        q_person.push(Detection{id, label, score, Detection::BBox{ymin, xmin, ymax, xmax}, std::vector<Detection::Keypoint>()});
        if (q_person.size() > top_k) q_person.pop();
     }
     else { // objects
       q_object.push(Detection{id, label, score, Detection::BBox{ymin, xmin, ymax, xmax}, std::vector<Detection::Keypoint>()});
       if (q_object.size() > top_k) q_object.pop();
     }
   }

   res.clear();
   res.reserve(q_person.size());
   while (!q_person.empty()) {
       res.push_back(q_person.top());
       q_person.pop();
   }
   std::reverse(res.begin(), res.end());

   res_obj.clear();
   res_obj.reserve(q_object.size());
   while (!q_object.empty()) {
       res_obj.push_back(q_object.top());
       q_object.pop();
   }
   std::reverse(res_obj.begin(), res_obj.end());
}

void PoseDetEngine::get_segmentation(cv::Mat &logits, const samplesCommon::BufferManager &buffers){
  const float* data_segm_out = static_cast<const float*>(buffers.getHostBuffer(mParams.outputTensorNames_segm[0]));

  int batchsize = mOutputDimsSegm.d[0];
  if(batchsize != 1){
    sample::gLogError << "Segmentation batchsize expected to be 1 but is: " << batchsize << ". Will only output segmentation of one input image!" << std::endl;
  }

  int outputH = mOutputDimsSegm.d[1];
  int outputW = mOutputDimsSegm.d[2];
  int outputC = mOutputDimsSegm.d[3];
  //int outputVol = samplesCommon::volume(mOutputDimsSegm);

  logits = cv::Mat(outputH, outputW, CV_32FC(outputC), (void*) (data_segm_out));
}

void PoseDetEngine::get_heatmaps(cv::Mat &heatmaps){
   const float* heatmap_data = static_cast<const float*>(mBuffersPose[0]->getHostBuffer(mParams.outputTensorNames_pose[0]));
   heatmaps = cv::Mat(output_tensor_shapes_pose[0][1], output_tensor_shapes_pose[0][2], CV_32FC(output_tensor_shapes_pose[0][3]), (void*)(heatmap_data)) / 255.;
}

void PoseDetEngine::get_heatmaps(std::vector<cv::Mat> &heatmaps_vec, const int batchsize, const int id){
  //assert(batchsize <= mParams.batchSizes_pose[id]);
  if(batchsize > mParams.batchSizes_pose[id]){
     sample::gLogError << "Error: batchsize " << batchsize << " > max. batchsize of model: " << mParams.batchSizes_pose[id] << std::endl;
     return;
   }

   const float* heatmap_data = static_cast<const float*>(mBuffersPose[id]->getHostBuffer(mParams.outputTensorNames_pose[0]));
   for (int i = 0; i < batchsize; ++i) {
     //heatmaps_vec.emplace_back(output_tensor_shapes_pose[0][1], output_tensor_shapes_pose[0][2], CV_32FC(output_tensor_shapes_pose[0][3]), (void*)(heatmap_data + i * output_sizes_pose[0]));
     heatmaps_vec.push_back(cv::Mat(output_tensor_shapes_pose[0][1], output_tensor_shapes_pose[0][2], CV_32FC(output_tensor_shapes_pose[0][3]), (void*)(heatmap_data + i * output_sizes_pose[0])) / 255.);
   }
}

void PoseDetEngine::infer_batch_pose(std::vector<cv::Mat> &heatmaps_vec, const std::vector<cv::Mat> &crops){
  heatmaps_vec.clear();
  int num_crops = crops.size(), start_batch = 0;
  const int num_pose_engines = mEnginesPose.size();

  while(num_crops > 0){
    int batchsize = 0, id = 0;
    for (; id < num_pose_engines; ++id) {
      if(num_crops <= mParams.batchSizes_pose[id]){
        batchsize = num_crops;
        break;
      }
    }

    if(id >= num_pose_engines){
      id = num_pose_engines - 1;
      batchsize = mParams.batchSizes_pose[id];
    }

    process_pose_input(crops, start_batch, batchsize, id);
    invoke_pose(batchsize, id);
    get_heatmaps(heatmaps_vec, batchsize, id);

    start_batch += batchsize;
    num_crops -= batchsize;
  }

  //assert(heatmaps_vec.size() == crops.size());
  if(heatmaps_vec.size() != crops.size()){
    sample::gLogError << "Error: number of output heatmaps: " << heatmaps_vec.size() << " not equal to number of input crops: " << crops.size() << std::endl;
  }
}

void PoseDetEngine::def_colormap(){
  colormap_ade20k.row( 0) = Eigen::Matrix<uint8_t, 1, 3>(0, 0, 0); //  0 background (0)
  colormap_ade20k.row( 1) = Eigen::Matrix<uint8_t, 1, 3>(217, 83, 25); //  1 wall (1, 19) ## 19
  colormap_ade20k.row( 2) = Eigen::Matrix<uint8_t, 1, 3>(158, 158, 158); //  2 floor (4, 29, 41, 102) # 41, 102
  colormap_ade20k.row( 3) = Eigen::Matrix<uint8_t, 1, 3>(0, 114, 189); //  3 ceiling (6)
  colormap_ade20k.row( 4) = Eigen::Matrix<uint8_t, 1, 3>(128, 64, 0); //  4 window (9)
  colormap_ade20k.row( 5) = Eigen::Matrix<uint8_t, 1, 3>(255, 255, 64); //  5 door (15, 59)
  colormap_ade20k.row( 6) = Eigen::Matrix<uint8_t, 1, 3>(217, 83, 25); //  6 column (41, 43, 94)
  colormap_ade20k.row( 7) = Eigen::Matrix<uint8_t, 1, 3>(162, 20, 47); //  7 stairs (54, 60, 97, 122)
  colormap_ade20k.row( 8) = Eigen::Matrix<uint8_t, 1, 3>(222,184,135);//  8 table (16, 34, 46, 57, 65, 71, 78) ## 78
  colormap_ade20k.row( 9) = Eigen::Matrix<uint8_t, 1, 3>(126, 47, 142);//  9 chair (20, 31, 76, 111)
  colormap_ade20k.row(10) = Eigen::Matrix<uint8_t, 1, 3>(126, 47, 142);//  10 seat (8, 24, 32, 40, 58, 70, 98, 132) ## 8, 58, 132 ---> chair
  colormap_ade20k.row(11) = Eigen::Matrix<uint8_t, 1, 3>(222,184,135);//  11 cabinet (11, 36, 45, 74, 100) ---> table
  colormap_ade20k.row(12) = Eigen::Matrix<uint8_t, 1, 3>(222,184,135);//  12 shelf (25, 63) ---> table
  colormap_ade20k.row(13) = Eigen::Matrix<uint8_t, 1, 3>(77, 190, 238); //  13 lamp (37, 83, 86, 88, 135, 137) # 88, 137
  colormap_ade20k.row(14) = Eigen::Matrix<uint8_t, 1, 3>(128, 128, 0); //  14 person (13)
  colormap_ade20k.row(15) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);   // 15 animal --> other
  colormap_ade20k.row(16) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250); //  16 vehicle (21, 81, 84, 103) --> other
  colormap_ade20k.row(17) = Eigen::Matrix<uint8_t, 1, 3>(196, 64, 128); //  17 bike (117, 128)
  colormap_ade20k.row(18) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250); //  18 poster (23, 101, 124, 145, 149, 150) # 124, 149, 150 --> other
  colormap_ade20k.row(19) = Eigen::Matrix<uint8_t, 1, 3>(127,255,0);//  19 box (42, 112, 56, 113, 116, 126, 136); #112, 126, 136
  colormap_ade20k.row(20) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);//  20 book (68) --> other
  colormap_ade20k.row(21) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);//  21 toy --> other
  colormap_ade20k.row(22) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);//  22 fridge --> other
  colormap_ade20k.row(23) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);//  23 dishwasher --> other
  colormap_ade20k.row(24) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);//  24 oven --> other
  colormap_ade20k.row(25) = Eigen::Matrix<uint8_t, 1, 3>(0, 128, 128);//  25 trashbin
  colormap_ade20k.row(26) = Eigen::Matrix<uint8_t, 1, 3>(255,0,255);//  26 computer (75)
  colormap_ade20k.row(27) = Eigen::Matrix<uint8_t, 1, 3>(255,0,255);//  27 TV (75) --> computer
  colormap_ade20k.row(28) = Eigen::Matrix<uint8_t, 1, 3>(255,0,255);//  28 screen (90, 131, 142, 144) --> computer
  colormap_ade20k.row(29) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);//  29 glass --> other
  colormap_ade20k.row(30) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);//  30 bottle --> other
  colormap_ade20k.row(31) = Eigen::Matrix<uint8_t, 1, 3>(230,230,250);//  31 food (121) --> other
}

ThermalDetEngine::ThermalDetEngine(const string& model_path_det, uint32_t precision, const std::string &calib_data_dir) {
  mONNX = false;
  if(model_path_det.find(".onnx") != string::npos){
    mONNX = true;
  }
  else{
    mParams.inputTensorNames.push_back("Input");
    mParams.outputTensorNames.push_back("NMS");
    mParams.outputTensorNames.push_back("NMS_1");
  }

  mParams.dlaCore = (precision & 0x4) ? 0 : -1;
  mParams.int8 = (precision & 0x2);
  mParams.fp16 = (precision & 0x1);

  mParams.batchSize = 1;

  if(mParams.int8){
    if(calib_data_dir == "")
      sample::gLogError << "Using INT8 precision but no calib data directory given!" << std::endl;
    mParams.calibDataDirs.push_back(calib_data_dir);
  }

  if(mParams.int8)
    sample::gLogInfo << "Using INT8 precision, calib data: " <<  calib_data_dir << std::endl;
  else if (mParams.fp16)
    sample::gLogInfo << "Using FP16 precision." << std::endl;
  else
    sample::gLogInfo << "Using FP32 precision." << std::endl;
  if(mParams.dlaCore >= 0)
    sample::gLogInfo << "Using DLA." << std::endl;

  if(mONNX){
    build_det_onnx(model_path_det);
    mBuffersDet = std::make_shared<samplesCommon::BufferManager>(mEngineDet);
    mContextDet = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngineDet->createExecutionContext());

    if (!mContextDet){
        sample::gLogError << "Could not create thermal det onnx inference context!" << std::endl;
        return;
    }

    sample::gLogInfo << "thermal det onnx context created" << std::endl;
    auto input_dims = mContextDet->getBindingDimensions(0);
    sample::gLogInfo << "Thermal ONNX Context input dimensions: [" << input_dims.d[0] << ", " << input_dims.d[1] << ", " << input_dims.d[2] << ", " << input_dims.d[3] << "]" << std::endl;
    auto output_dims = mContextDet->getBindingDimensions(1);
    sample::gLogInfo << "Thermal ONNX Context output dimensions (1): [" << output_dims.d[0] << ", " << output_dims.d[1] << ", " << output_dims.d[2] << ", " << output_dims.d[3] << "]" << std::endl;
    output_dims = mContextDet->getBindingDimensions(2);
    sample::gLogInfo << "Thermal ONNX Context output dimensions (2): [" << output_dims.d[0] << ", " << output_dims.d[1] << ", " << output_dims.d[2] << ", " << output_dims.d[3] << "]" << std::endl;
  }
  else {
    build_det(model_path_det);
    mBuffersDet = std::make_shared<samplesCommon::BufferManager>(mEngineDet, mParams.batchSize);
    mContextDet =  SampleUniquePtr<nvinfer1::IExecutionContext>(mEngineDet->createExecutionContext());
  }

  input_size_det = 1;
  input_tensor_shape_det.resize(mONNX ? mInputDims.nbDims : mInputDims.nbDims + 1);
  input_tensor_shape_det[0] = mONNX ? mInputDims.d[0] : 1; //Batch-Size Det. Expected to be in NHWC order!
  input_tensor_shape_det[1] = mONNX ? mInputDims.d[2] : mInputDims.d[1];
  input_tensor_shape_det[2] = mONNX ? mInputDims.d[3] : mInputDims.d[2];
  input_tensor_shape_det[3] = mONNX ? mInputDims.d[1] : mInputDims.d[0];
  for (int i = 0; i < mInputDims.nbDims; ++i) {
    input_size_det *= mInputDims.d[i];
  }

  output_tensor_shapes_det.resize(mOutputDims.size());
  output_sizes_det.resize(mOutputDims.size());
  for (int j = 0; j < mOutputDims.size(); ++j) {
    output_sizes_det[j] = 1;
    output_tensor_shapes_det[j].resize(mONNX ? mOutputDims[j].nbDims : mOutputDims[j].nbDims + 1);
    output_tensor_shapes_det[j][0] = 1; // Batch-Size Det
    for (int i = 0; i < mOutputDims[j].nbDims; ++i) {
      output_tensor_shapes_det[j][mONNX ? i : i+1] = mOutputDims[j].d[i];
      output_sizes_det[j] *= mOutputDims[j].d[i];
    }
  }

  class_names = {"NONE", "person"};
}

bool ThermalDetEngine::build_det_onnx(std::string model_path){
  std::stringstream gieModelStream;
  gieModelStream.seekg(0, gieModelStream.beg);

  std::stringstream cache_path_ss;
  cache_path_ss << model_path << "_" << mParams.batchSize << "_" << (mParams.int8 ? "uint8" : (mParams.fp16 ? "fp16" : "fp32")) << ".engine";
  std::string cache_path = cache_path_ss.str();
  std::ifstream cache( cache_path.c_str() );

  if(!cache){
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));

    auto parsed = parser->parseFromFile(model_path.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    sample::gLogInfo << "Thermal ONNX network has " << network->getNbInputs() << "input and " << network->getNbOutputs() << "outputs" << std::endl;
    if(network->getNbInputs() != 1 || network->getNbOutputs() != 2){
      sample::gLogError << "Expected one input an two output tensors but got " << network->getNbInputs() << " inputs and " << network->getNbOutputs() << " outputs. Aborting!" << std::endl;
      return false;
    }
    auto input_dims = network->getInput(0)->getDimensions();
    if(input_dims.nbDims != 4){
      sample::gLogError << "Expected 4-dimensional input tensor but got " << input_dims.nbDims << ". Aborting!" << std::endl;
      return false;
    }

    for (int i = 0; i < network->getNbInputs(); ++i) {
      sample::gLogInfo << "Input " << i << ": " << network->getInput(i)->getName() << " " << to_string(network->getInput(i)->getDimensions()) << std::endl;
    }
    for (int i = 0; i < network->getNbOutputs(); ++i) {
      sample::gLogInfo << "Output " << i << ": " << network->getOutput(i)->getName() << " " << to_string(network->getOutput(i)->getDimensions()) << std::endl;
    }

    network->getInput(0)->setDimensions(Dims4{1, 1, 120, 160}); // For inference: context.setBindingDimensions(0, Dims4{1, 1, 120, 160})
    input_dims = network->getInput(0)->getDimensions();
    sample::gLogInfo << "Fixed input dimensions: [" << input_dims.d[0] << ", " << input_dims.d[1] << ", " << input_dims.d[2] << ", " << input_dims.d[3] << "]" << std::endl;

    mParams.inputTensorNames.push_back(network->getInput(0)->getName());
    mInputDims = input_dims;

    mParams.outputTensorNames.push_back(network->getOutput(0)->getName());
    mParams.outputTensorNames.push_back(network->getOutput(1)->getName());
    mOutputDims.push_back(network->getOutput(0)->getDimensions());
    mOutputDims.push_back(network->getOutput(1)->getDimensions());
    sample::gLogInfo << "output dimensions 1: " << to_string(network->getOutput(0)->getDimensions()) << "; 2: " << to_string(network->getOutput(1)->getDimensions()) << std::endl;

    sample::gLogInfo << "Thermal Detector batch size: " << mParams.batchSize << std::endl;

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(1_GiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

    if (mParams.int8)
    {
      sample::gLogInfo << "Using Entropy Calibrator 2" << std::endl;
      int slash_idx = model_path.rfind('/');
      int dot_idx = model_path.rfind('.');
      std::string file_suffix = model_path.substr(slash_idx + 1, dot_idx - slash_idx - 1);
      MyBatchStream calibrationStream(0, mInputDims, mParams.calibDataDirs, false, false, 5);
      calibrator.reset(new Int8EntropyCalibrator2<MyBatchStream>(
          calibrationStream, 0, ("ONNXSSDThermal_" + file_suffix).c_str(), mParams.inputTensorNames[0].c_str()));
      config->setFlag(BuilderFlag::kINT8);
      config->setInt8Calibrator(calibrator.get());
    }

    // No DLA for detector as results are very bad...
//      sample::gLogInfo << "Default Device Type: " << int(config->getDefaultDeviceType()) << std::endl;
//      if(builder->getNbDLACores() > 0 && mParams.dlaCore >= 0)
//      {
//          samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore + 1);
//          sample::gLogInfo << "Using DLA core: " << mParams.dlaCore + 1 << std::endl;
//      }

    mEngineDet = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngineDet)
    {
        return false;
    }

    auto output_dims_1 = mEngineDet->getBindingDimensions(1);
    sample::gLogInfo << "Engine output dimensions: [" << output_dims_1.d[0] << ", " << output_dims_1.d[1] << ", " << output_dims_1.d[2] << ", " << output_dims_1.d[3] << "]" << std::endl;
    auto output_dims_2 = mEngineDet->getBindingDimensions(2);
    sample::gLogInfo << "Engine output dimensions: [" << output_dims_2.d[0] << ", " << output_dims_2.d[1] << ", " << output_dims_2.d[2] << ", " << output_dims_2.d[3] << "]" << std::endl;

    nvinfer1::IHostMemory* serMem = mEngineDet->serialize();

    if( !serMem ){
        sample::gLogInfo << "Failed to serialize CUDA engine" << std::endl;
        return true;
    }

    gieModelStream.write((const char*)serMem->data(), serMem->size());

    sample::gLogInfo << "writing engine cache to: " << cache_path << std::endl;;

    std::ofstream outFile;
    outFile.open(cache_path);
    outFile << gieModelStream.rdbuf();
    outFile.close();
    gieModelStream.seekg(0, gieModelStream.beg);
    sample::gLogInfo << "Completed writing engine cache." << std::endl;

    return true;
  }
  else{
    sample::gLogInfo << "loading network profile from engine cache: " << cache_path << std::endl;
    gieModelStream << cache.rdbuf();
    cache.close();

    nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
    if (!infer){
        sample::gLogInfo <<  "Failed to create Infer Runtime" << std::endl;;
        return false;
    }

    gieModelStream.seekg(0, std::ios::end);
    const int modelSize = gieModelStream.tellg();
    gieModelStream.seekg(0, std::ios::beg);

    void* modelMem = malloc(modelSize);

    if( !modelMem ){
        sample::gLogError << "Failed to allocate memory to deserialize buffer" << std::endl;
        return false;
    }

    gieModelStream.read((char*)modelMem, modelSize);
    sample::gLogInfo << "deserializing..." << std::endl;
    mEngineDet = std::shared_ptr<nvinfer1::ICudaEngine>(infer->deserializeCudaEngine(modelMem, modelSize), samplesCommon::InferDeleter());
    free(modelMem);
    if (!mEngineDet)
    {
        return false;
    }
    sample::gLogInfo << "Success" << std::endl;

    const int n = mEngineDet->getNbBindings();
    for (int i = 0; i < n; ++i) {
        const nvinfer1::Dims dims = mEngineDet->getBindingDimensions(i);
        const nvinfer1::DataType dtype = mEngineDet->getBindingDataType(i);
        const std::string name(mEngineDet->getBindingName(i));
        const bool input = mEngineDet->bindingIsInput(i);
        if(input){
	    mParams.inputTensorNames.push_back(name);
            mInputDims = dims;
	}
        else{
	    mParams.outputTensorNames.push_back(name);
            mOutputDims.push_back(dims);
	}

        sample::gLogInfo << "binding " << i << ":" << " name: " << name << ", is input: " << input << ", type: " << to_string(dtype) << " " << to_string(dims) << std::endl;
    }

    return true;

  }
}

bool ThermalDetEngine::build_det(std::string model_path){
  auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));

  std::stringstream gieModelStream;
  gieModelStream.seekg(0, gieModelStream.beg);

  std::stringstream cache_path_ss;
  cache_path_ss << model_path << "_" << mParams.batchSize << "_" << (mParams.int8 ? "uint8" : (mParams.fp16 ? "fp16" : "fp32")) << ".engine";
  std::string cache_path = cache_path_ss.str();
  std::ifstream cache( cache_path.c_str() );

  if(!cache){
      auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork()); //builder->createNetworkV2(0U)
      auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
      auto parser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());

      parser->registerInput(mParams.inputTensorNames[0].c_str(), DimsCHW(1, 120, 160), nvuffparser::UffInputOrder::kNCHW);
      parser->registerOutput(mParams.outputTensorNames[0].c_str());

      auto parsed = parser->parse(model_path.c_str(), *network, DataType::kFLOAT);
      if (!parsed)
      {
          return false;
      }

      sample::gLogInfo << "Thermal Detector batch size: " << mParams.batchSize << std::endl;

      builder->setMaxBatchSize(mParams.batchSize);
      config->setMaxWorkspaceSize(1_GiB);
      if (mParams.fp16)
      {
          config->setFlag(BuilderFlag::kFP16);
      }

      // Calibrator life time needs to last until after the engine is built.
      std::unique_ptr<IInt8Calibrator> calibrator;

      if (mParams.int8)
      {
        sample::gLogInfo << "Using Entropy Calibrator 2" << std::endl;
        int slash_idx = model_path.rfind('/');
        int dot_idx = model_path.rfind('.');
        std::string file_suffix = model_path.substr(slash_idx + 1, dot_idx - slash_idx - 1);
        const int32_t imageC = 1;
        const int32_t imageH = 120;
        const int32_t imageW = 160;
        const int32_t batchsize = mParams.batchSize;
        const nvinfer1::DimsNCHW imageDims{batchsize, imageC, imageH, imageW};
        MyBatchStream calibrationStream(batchsize, imageDims, mParams.calibDataDirs, false, false, 5);
        calibrator.reset(new Int8EntropyCalibrator2<MyBatchStream>(
            calibrationStream, 0, ("UffSSDThermal_" + file_suffix).c_str(), mParams.inputTensorNames[0].c_str()));
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
      }

      // No DLA for detector as results are very bad...
//      sample::gLogInfo << "Default Device Type: " << int(config->getDefaultDeviceType()) << std::endl;
//      if(builder->getNbDLACores() > 0 && mParams.dlaCore >= 0)
//      {
//          samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore + 1);
//          sample::gLogInfo << "Using DLA core: " << mParams.dlaCore + 1 << std::endl;
//      }

      mEngineDet = std::shared_ptr<nvinfer1::ICudaEngine>(
          builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
      if (!mEngineDet)
      {
          return false;
      }

      assert(network->getNbInputs() == 1);
      mInputDims = network->getInput(0)->getDimensions();
      assert(mInputDims.nbDims == 3);
      assert(network->getNbOutputs() == 2);
      mOutputDims.push_back(network->getOutput(0)->getDimensions());
      mOutputDims.push_back(network->getOutput(1)->getDimensions());

      nvinfer1::IHostMemory* serMem = mEngineDet->serialize();

      if( !serMem ){
          sample::gLogInfo << "Failed to serialize CUDA engine" << std::endl;
          return true;
      }

      gieModelStream.write((const char*)serMem->data(), serMem->size());

      sample::gLogInfo << "writing engine cache to: " << cache_path << std::endl;;

      std::ofstream outFile;
      outFile.open(cache_path);
      outFile << gieModelStream.rdbuf();
      outFile.close();
      gieModelStream.seekg(0, gieModelStream.beg);
      sample::gLogInfo << "Completed writing engine cache." << std::endl;

      return true;
  }
  else{

      sample::gLogInfo << "loading network profile from engine cache: " << cache_path << std::endl;
      gieModelStream << cache.rdbuf();
      cache.close();

      nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger());
      if (!infer){
          sample::gLogInfo <<  "Failed to create Infer Runtime" << std::endl;;
          return false;
      }

      gieModelStream.seekg(0, std::ios::end);
      const int modelSize = gieModelStream.tellg();
      gieModelStream.seekg(0, std::ios::beg);

      void* modelMem = malloc(modelSize);

      if( !modelMem ){
          sample::gLogError << "Failed to allocate memory to deserialize buffer" << std::endl;
          return false;
      }

      gieModelStream.read((char*)modelMem, modelSize);
      sample::gLogInfo << "deserializing..." << std::endl;
      mEngineDet = std::shared_ptr<nvinfer1::ICudaEngine>(infer->deserializeCudaEngine(modelMem, modelSize), samplesCommon::InferDeleter());
      free(modelMem);
      if (!mEngineDet)
      {
          return false;
      }
      sample::gLogInfo << "Success" << std::endl;

      const int n = mEngineDet->getNbBindings();
      for (int i = 0; i < n; ++i) {
          const nvinfer1::Dims dims = mEngineDet->getBindingDimensions(i);
          const nvinfer1::DataType dtype = mEngineDet->getBindingDataType(i);
          const std::string name(mEngineDet->getBindingName(i));
          const bool input = mEngineDet->bindingIsInput(i);
          if(input)
              mInputDims = dims;
          else
              mOutputDims.push_back(dims);

          sample::gLogInfo << "binding " << i << ":" << " name: " << name << ", is input: " << input << ", type: " << to_string(dtype) << " " << to_string(dims) << std::endl;
      }
      assert(mInputDims.nbDims == 3);

      return true;
  }
}

ThermalDetEngine::~ThermalDetEngine(){
  nvuffparser::shutdownProtobufLibrary();
}

void ThermalDetEngine::process_det_input(const cv::Mat& img){
  if(mONNX){
    int batchsize = mInputDims.d[0];
    if(batchsize != 1){
      sample::gLogError << "Thermal det batchsize expected to be 1 but is: " << batchsize << std::endl;
    }
  }
  const int32_t inputC = mONNX ? mInputDims.d[1] : mInputDims.d[0];
  const int32_t inputH = mONNX ? mInputDims.d[2] : mInputDims.d[1];
  const int32_t inputW = mONNX ? mInputDims.d[3] : mInputDims.d[2];

  float* hostDataBuffer = static_cast<float*>(mBuffersDet->getHostBuffer(mParams.inputTensorNames[0])); // Host memory for input buffer

  //Batch-Size 1
  for (int32_t c = 0; c < inputC; ++c)
  {
      // The color image to input should be in RGB order
      for (uint32_t j = 0, volChl = inputH * inputW; j < volChl; ++j)
      {
          hostDataBuffer[c * volChl + j]
              = (2.0f / 255.0f) * float(img.data[j * inputC + c]) - 1.0f;
      }
  }
}

void ThermalDetEngine::invoke_det(){
  mBuffersDet->copyInputToDevice();
  const bool status = mONNX? mContextDet->executeV2(mBuffersDet->getDeviceBindings().data()) :
                             mContextDet->execute(mParams.batchSize, mBuffersDet->getDeviceBindings().data());
  if (!status)
      sample::gLogError << "Detector inference failed!" << std::endl;

  mBuffersDet->copyOutputToHost();
}

void ThermalDetEngine::get_det_result(std::vector<Detection>& res, float thresh, int top_k, const cv::Size& img_size, int x0, int y0, float scale){
  const float* detectionOut = static_cast<const float*>(mBuffersDet->getHostBuffer(mParams.outputTensorNames[0]));
  const int32_t* keepCount = static_cast<const int32_t*>(mBuffersDet->getHostBuffer(mParams.outputTensorNames[1]));
  int n_det = keepCount[0];

   std::priority_queue<Detection, std::vector<Detection>, DetComparator> q_person;
   for (int i = 0; i < n_det; ++i) {
     // Output format for each detection is stored in the order: [image_id, label, confidence, xmin, ymin, xmax, ymax]
     const float* det = &detectionOut[0] + i * 7;

     const int label = std::lround(det[1]);
     if(label == 1){ // 1 = person
       const int id = i; //std::lround(det[0]);
       const float score = det[2];
       if (score < thresh) continue;

       const float xmin = (std::max(0.0f, det[3]) * img_size.width  - x0) / scale;
       const float ymin = (std::max(0.0f, det[4]) * img_size.height - y0) / scale;
       const float xmax = (std::min(1.0f, det[5]) * img_size.width  - x0) / scale;
       const float ymax = (std::min(1.0f, det[6]) * img_size.height - y0) / scale;

       q_person.push(Detection{id, label, score, Detection::BBox{ymin, xmin, ymax, xmax}, std::vector<Detection::Keypoint>()});
       if (q_person.size() > top_k) q_person.pop();
     }
   }

   res.clear();
   res.reserve(q_person.size());
   while (!q_person.empty()) {
       res.push_back(q_person.top());
       q_person.pop();
   }
   std::reverse(res.begin(), res.end());
}

