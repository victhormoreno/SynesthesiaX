#pragma once
/**
 * @file   Projector.hpp
 * @brief  Projects 3D point cloud onto semantic image; provides semantic colored clouds and overlays.
 *
 * © 2025, Your Robotics Lab – MIT licence.
 */

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <ros/ros.h>

class Projector
{
public:
  Projector();

  /**
   * @brief Initialize projector parameters and calibration.
   * @param minRange Minimum Euclidean distance for valid points (m).
   * @param maxRange Maximum Euclidean distance for valid points (m).
   * @param minAngFOV Minimum polar angle in degrees.
   * @param maxAngFOV Maximum polar angle in degrees.
   * @return True if calibration parameters loaded successfully.
   */
  bool init(const ros::NodeHandle& nh);

  /**
   * @brief Process input cloud and image to filter points and build depth/index buffers.
   * @param cloud_msg Input point cloud message.
   * @param image_msg Input semantic label image message.
   */
  void project_cloud_onto_image(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                                const sensor_msgs::ImageConstPtr& image_msg);

  /**
   * @brief Get semantic colored point clouds.
   */
  void getSemanticClouds(pcl::PointCloud<pcl::PointXYZRGB>& semanticCloud,
                        pcl::PointCloud<pcl::PointXYZRGB>& travCloud,
                        pcl::PointCloud<pcl::PointXYZRGB>& obstacleCloud) const;

  /**
   * @brief Get semantic overlay image. Computes overlay lazily.
   */
  const cv::Mat& getOverlay();

private:
  // Main modular steps called inside project_cloud_onto_image
  void filterPointCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud_in);
  void createDepthBuffers();

  // Calibration matrices and distortion coefficients
  cv::Mat cameraMatrix_, distCoeffs_, rvec_, tvec_, R_, R_inv_, K_;

  // Semantic color lookup table (BGR)
  cv::Mat semantic_lut_;

  // Filtering parameters
  double minRange_, maxRange_;
  double minAngFOV_, maxAngFOV_;

  static constexpr int CALIB_W = 1440;
  static constexpr int CALIB_H = 1080;
  static constexpr int LABEL_W = 500;
  static constexpr int LABEL_H = 500;

  // 3D points and their 2D projections
  std::vector<cv::Point3f> pts3d_;
  std::vector<cv::Point2f> proj2d_;

  // Buffers for depth and indices per pixel
  cv::Mat depth_buf_;
  cv::Mat idx_buf_;
  cv::Mat labels_;

  // Overlay cache for getOverlay()
  cv::Mat overlay_cache_;
  bool overlay_updated_ = false;
};
