#pragma once

#include <sensor_msgs/PointCloud2.h>
#include <memory>
#include <Eigen/Dense>
#include <unordered_map>

class Voxel
{
public:
    typedef std::shared_ptr<Voxel > Ptr;

    static Ptr Create()
    {
        return std::make_shared<Voxel>();
    }
    Voxel() {}

    struct VoxelData
    {
        Eigen::Vector3f m_pos_acc = Eigen::Vector3f::Zero();
        int m_num = 0;
    };

    void addNewPoint( const Eigen::Vector3f & new_point);

    Eigen::Vector3f getMean ( ) const
    {
        if ( m_fused.m_num == 0 )
            return Eigen::Vector3f::Constant(std::numeric_limits<double>::signaling_NaN());
        else
            return m_fused.m_pos_acc / m_fused.m_num;
    }

    int getNum () const
    {
        return m_fused.m_num;
    }

private:
    VoxelData m_fused;
};

class VoxelGrid
{
public:
    
    typedef int64_t IndexType;
    typedef Eigen::Matrix<int64_t,3,1> IndexVec3Type;
    typedef std::shared_ptr<VoxelGrid > Ptr;

    static Ptr Create(const float& side_length, const float& voxel_length);
    void addCloud ( const sensor_msgs::PointCloud2ConstPtr & cloud );
    void addCloud ( const Eigen::Matrix3Xf & points3D, const Eigen::Matrix<bool, 1, Eigen::Dynamic, Eigen::RowMajor>& depth_mask);
    sensor_msgs::PointCloud2Ptr getFusedVoxel();
    void getFusedVoxel(Eigen::Matrix3Xf& points3Dfiltered);

    inline IndexVec3Type getMiddleIndexVector( const IndexType & num_cells ) const
    {
        return IndexVec3Type( num_cells>>1, num_cells>>1, num_cells>>1 );
    }

    inline IndexType toCellIndex (const Eigen::Vector3f & point, const float &voxel_scale, const IndexVec3Type & middleIndexVector, const IndexType & num_voxels );

    VoxelGrid(const float& side_length, const float& voxel_length ) : m_side_length(side_length), m_voxel_length(voxel_length) {}

private:
    const float m_side_length;
    const float m_voxel_length;
    std::unordered_map<IndexType, Voxel::Ptr> m_voxels;
};
