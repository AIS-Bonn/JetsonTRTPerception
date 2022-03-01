#include <jetson_trt_pose/VoxelFilter.h>
#include <ros/ros.h>
#include <cmath>

void Voxel::addNewPoint( const Eigen::Vector3f & new_point )
{
    // Accumulate position. (will be devided by m_num in getMean())
    ++m_fused.m_num;
    m_fused.m_pos_acc += new_point;
}

VoxelGrid::Ptr VoxelGrid::Create (const float &side_length, const float &voxel_length)
{
    return std::make_shared<VoxelGrid>(side_length, voxel_length);
}

inline VoxelGrid::IndexType VoxelGrid::toCellIndex ( const Eigen::Vector3f & point, const float & voxel_scale, const IndexVec3Type & middleIndexVector, const IndexType & num_voxels )
{
    const IndexVec3Type idx_vec = (point * voxel_scale).array().floor().cast<IndexType>(); // ((numcells>>1)<<1) // even
    const IndexVec3Type p = idx_vec + middleIndexVector;
    const IndexVec3Type num_cells_per_side ( 1, num_voxels, num_voxels * num_voxels);
    return p.dot(num_cells_per_side);
}

void VoxelGrid::addCloud ( const sensor_msgs::PointCloud2ConstPtr & cloud_msg)
{
    uint32_t offset_x = 0;
    uint32_t offset_y = 0;
    uint32_t offset_z = 0;
    
    for( size_t i = 0; i < cloud_msg->fields.size(); ++i )
    {
        if ( cloud_msg->fields[i].name=="x" ) offset_x = cloud_msg->fields[i].offset;
        if ( cloud_msg->fields[i].name=="y" ) offset_y = cloud_msg->fields[i].offset;
        if ( cloud_msg->fields[i].name=="z" ) offset_z = cloud_msg->fields[i].offset;
    }
    
    const size_t num_points = cloud_msg->data.size() / cloud_msg->point_step;
    //const size_t semantic_length = _N * sizeof(float);

    const float lower_bound = -m_side_length / 2.f;
    const float upper_bound = m_side_length / 2.f;
    const IndexType num_voxels_per_side = IndexType(m_side_length / m_voxel_length);
    const float voxel_scale = num_voxels_per_side / m_side_length;
    static const IndexVec3Type middleIndexVector = getMiddleIndexVector ( num_voxels_per_side );

    //ROS_INFO_STREAM_THROTTLE(1,"side: " << m_side_length << ", res: " << m_voxel_length << " vxPS: " << num_voxels_per_side << " vs: " << voxel_scale << " mi: " << middleIndexVector.transpose());

    for ( size_t idx = 0; idx < num_points; ++idx )
    {
        const size_t point_offset = idx * cloud_msg->point_step;
        const Eigen::Map<const Eigen::Vector3f> point ( (float*) &cloud_msg->data[point_offset+offset_x] );
        if((point.array() < lower_bound).any() || (point.array() >= upper_bound).any())
          continue;

        const IndexType voxel_idx = toCellIndex(point, voxel_scale, middleIndexVector, num_voxels_per_side);
        if ( ! m_voxels[voxel_idx] ) m_voxels[voxel_idx] = Voxel::Create();
        m_voxels[voxel_idx]->addNewPoint( point);
    }
}

void VoxelGrid::addCloud(const Eigen::Matrix3Xf &points3D, const Eigen::Matrix<bool, 1, Eigen::Dynamic, Eigen::RowMajor> &depth_mask){

  const float lower_bound = -m_side_length / 2.f;
  const float upper_bound = m_side_length / 2.f;
  const IndexType num_voxels_per_side = IndexType(m_side_length / m_voxel_length);
  const float voxel_scale = num_voxels_per_side / m_side_length;
  static const IndexVec3Type middleIndexVector = getMiddleIndexVector ( num_voxels_per_side );

  const size_t num_points = points3D.cols();
  for ( int ptIdx = 0; ptIdx < num_points; ++ptIdx )
  {
    if(!depth_mask(ptIdx))
      continue;

    const Eigen::Vector3f& point = points3D.col(ptIdx);
    if((point.array() < lower_bound).any() || (point.array() >= upper_bound).any())
      continue;

    const IndexType voxel_idx = toCellIndex(point, voxel_scale, middleIndexVector, num_voxels_per_side);
    if ( ! m_voxels[voxel_idx] ) m_voxels[voxel_idx] = Voxel::Create();
    m_voxels[voxel_idx]->addNewPoint(point);
  }
}

sensor_msgs::PointCloud2Ptr VoxelGrid::getFusedVoxel()
{
    sensor_msgs::PointCloud2Ptr msg ( new sensor_msgs::PointCloud2() );

    // add fields
    msg->fields.resize(4); // x,y,z,rgb
    msg->fields[0].name = "x";
    msg->fields[1].name = "y";
    msg->fields[2].name = "z";
    msg->fields[3].name = "rgb";
    for ( size_t i = 0; i < 4; ++i )
    {
        msg->fields[i].datatype = sensor_msgs::PointField::FLOAT32;
        msg->fields[i].count = 1;
        msg->fields[i].offset = i*sizeof(float);
    }

    const size_t num_points = m_voxels.size();
    const size_t point_step = 4*sizeof(float);
    const size_t offset_rgb = msg->fields[3].offset;

    msg->header.frame_id = ""; //TODO fixed_frame;
    msg->data.resize(num_points * point_step, 0);
    msg->width = num_points;
    msg->height = 1;

    msg->point_step = point_step;
    msg->row_step = point_step * msg->width;
    msg->is_dense = false;

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
      void setZero(){this->b = 0; this->g = 0; this->r = 0; this->a = 0;}
    };
    PointRGB pt_rgb;
    
    size_t idx = 0, num_nullptrs = 0;
    for ( const std::pair<const IndexType, Voxel::Ptr> & idx_voxel_ptr : m_voxels )
    {
        const Voxel::Ptr & voxel_ptr = idx_voxel_ptr.second;
        if ( ! voxel_ptr ) { ++idx; ++num_nullptrs; continue; }
        Voxel& voxel = *voxel_ptr;

        const size_t point_offset = idx * point_step;
        const Eigen::Vector3f pos = voxel.getMean();
        Eigen::Map<Eigen::Vector3f>((float*)&msg->data[point_offset]) = pos;

        pt_rgb.setZero();
        
        *reinterpret_cast<PointRGB*>(&msg->data[point_offset + offset_rgb]) = pt_rgb;
        ++idx;
    }
    
    return msg;
}

void VoxelGrid::getFusedVoxel(Eigen::Matrix3Xf &points3Dfiltered){
  const size_t num_points = m_voxels.size();
  //ROS_INFO("Filtered cloud has %zu points", num_points);
  points3Dfiltered.resize(3, num_points);

  size_t idx = 0;
  for ( const std::pair<const IndexType, Voxel::Ptr> & idx_voxel_ptr : m_voxels )
  {
      const Voxel::Ptr & voxel_ptr = idx_voxel_ptr.second;
      if ( voxel_ptr){
        //ROS_INFO("Voxel %zu: num observation: %d", idx, voxel_ptr->getNum());
        points3Dfiltered.col(idx) = voxel_ptr->getMean();
      }
      else{
        points3Dfiltered.col(idx) = Eigen::Vector3f::Zero();
        ROS_ERROR("nullpointer in voxel grid filter!");
      }

      ++idx;
  }
}
