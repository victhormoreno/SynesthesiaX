#include "Projector.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cmath>
#include <limits>

// ** Public methods **

Projector::Projector() : overlay_updated_(false), depth_img_updated_(false)
{}

bool Projector::init(const ros::NodeHandle& nh)
{

    // Load parameters from ROS parameter server
    if (!nh.getParam("min_range", minRange_) || !nh.getParam("max_range", maxRange_)) {
        ROS_ERROR("Failed to get min/max range parameters");
        return false;
    }

    if (!nh.getParam("min_ang_fov", minAngFOV_) || !nh.getParam("max_ang_fov", maxAngFOV_)) {
        ROS_ERROR("Failed to get min/max angular FOV parameters");
        return false;
    }

    // Load camera calibration parameters
    std::vector<double> cam, d, rlc, tlc;
    if (!nh.getParam("camera_matrix", cam) || cam.size() != 9) {
        ROS_ERROR("Invalid or missing camera matrix parameters");
        return false;
    }
    if (!nh.getParam("d", d) || d.size() != 5) {
        ROS_ERROR("Invalid or missing distortion coefficients parameters");
        return false;
    }
    if (!nh.getParam("rlc", rlc) || rlc.size() != 9) {
        ROS_ERROR("Invalid or missing rotation vector parameters");
        return false;
    }
    if (!nh.getParam("tlc", tlc) || tlc.size() != 3) {
        ROS_ERROR("Invalid or missing translation vector parameters");
        return false;
    }

    // Camera matrix 3x3
    cameraMatrix_ = cv::Mat::eye(3, 3, CV_64F);
    cameraMatrix_.at<double>(0, 0) = cam[0];
    cameraMatrix_.at<double>(0, 1) = cam[1];
    cameraMatrix_.at<double>(0, 2) = cam[2];
    cameraMatrix_.at<double>(1, 0) = cam[3];
    cameraMatrix_.at<double>(1, 1) = cam[4];
    cameraMatrix_.at<double>(1, 2) = cam[5];
    cameraMatrix_.at<double>(2, 0) = cam[6];
    cameraMatrix_.at<double>(2, 1) = cam[7];
    cameraMatrix_.at<double>(2, 2) = cam[8];

    // Distortion coefficients as column vector 5x1
    distCoeffs_ = cv::Mat(d).clone().reshape(1, 5);

    cv::Mat RlcMat(3, 3, CV_64F, rlc.data());
    cv::Rodrigues(RlcMat, rvec_);
    cv::Rodrigues(rvec_, R_);
    tvec_  = cv::Mat(3, 1, CV_64F, tlc.data()).clone();
    R_inv_ = R_.t();

    // Semantic LUT (label index to BGR color)
    semantic_lut_ = cv::Mat(1, 3, CV_8UC3);
    semantic_lut_.at<cv::Vec3b>(0, 0) = {0,   0, 255};  // Dynamic – red
    semantic_lut_.at<cv::Vec3b>(0, 1) = {0, 255,   0};  // Obstacle – green
    semantic_lut_.at<cv::Vec3b>(0, 2) = {255, 0,   0};  // Traversable – blue

    // Create depth LUT: 1 row, 256 columns, 3-channel 8-bit (BGR)
    depth_lut_ = cv::Mat(1, 256, CV_8UC3);

    for (int i = 0; i < 256; ++i) {

        float t = static_cast<float>(i) / (256 - 1);  // Normalize [0, 1]

        // Custom warm colormap: brown → red → orange → yellow → white
        uchar r, g, b;

        if (t < 0.25f) { // dark brown to red
            r = static_cast<uchar>(128 + t * 512);
            g = static_cast<uchar>(64 * t);
            b = static_cast<uchar>(32 * (1.0f - t));
        } else if (t < 0.5f) { // red to orange
            r = 255;
            g = static_cast<uchar>(128 * (t - 0.25f) / 0.25f);
            b = 0;
        } else if (t < 0.75f) { // orange to yellow
            r = 255;
            g = static_cast<uchar>(128 + 127 * (t - 0.5f) / 0.25f);
            b = 0;
        } else { // yellow to white
            r = 255;
            g = 255;
            b = static_cast<uchar>(255 * (t - 0.75f) / 0.25f);
        }

        depth_lut_.at<cv::Vec3b>(0, i) = cv::Vec3b(b, g, r);  // BGR
    }

    // Fill resolution to K matrix
    K_ = cameraMatrix_.clone();
    double sx = static_cast<double>(LABEL_W) / CALIB_W;
    double sy = static_cast<double>(LABEL_H) / CALIB_H;
    K_.at<double>(0, 0) *= sx; K_.at<double>(0, 2) *= sx;
    K_.at<double>(1, 1) *= sy; K_.at<double>(1, 2) *= sy;

    return true;
}

void Projector::project_cloud_onto_image(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                                         const sensor_msgs::ImageConstPtr& image_msg)
{
    if (!cloud_msg || !image_msg)
    {
        ROS_ERROR("Received null cloud or image message");
        return;
    }

    // 1. Convert input image to label matrix
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
    cv_ptr = cv_bridge::toCvShare(image_msg);
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR_STREAM("cv_bridge error: " << e.what());
        return;
    }
    cv_ptr->image.convertTo(labels_, CV_8UC1);

    // 2. Convert cloud message to pcl and filter points
    pcl::PointCloud<pcl::PointXYZ> cloud_in;
    pcl::fromROSMsg(*cloud_msg, cloud_in);
    this->filterPointCloud(cloud_in);

    // 3. Project filtered points into 2D image space
    proj2d_.clear();
    cv::projectPoints(pts3d_, rvec_, tvec_, K_, distCoeffs_, proj2d_);
    
    // 4. Create depth and index buffers for projected points
    this->createDepthBuffers();

    overlay_updated_ = false;   // Mark overlay cache as dirty
    depth_img_updated_ = false; // Mark depth cache as dirty
}

// ** Private methods **

void Projector::filterPointCloud(const pcl::PointCloud<pcl::PointXYZ>& cloud_in)
{
  pts3d_.clear();

  const double minAngRad = minAngFOV_ * M_PI / 180.0;
  const double maxAngRad = maxAngFOV_ * M_PI / 180.0;

  for (const auto& pt : cloud_in.points)
  {
    if (pt.x <= 0.0) continue;

    const double range = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
    if (range < minRange_ || range > maxRange_) continue;

    const double angle = std::atan2(std::sqrt(pt.y*pt.y + pt.z*pt.z), pt.x);
    if (angle < minAngRad || angle > maxAngRad) continue;

    pts3d_.emplace_back(pt.x, pt.y, pt.z);
  }
}

void Projector::createDepthBuffers()
{
    // Initialize depth and index buffers
    depth_buf_.create(cv::Size(LABEL_W, LABEL_H), CV_32F);
    idx_buf_.create(cv::Size(LABEL_W, LABEL_H), CV_32S);
    depth_buf_.setTo(std::numeric_limits<float>::infinity());
    idx_buf_.setTo(-1);

    const int H = depth_buf_.rows;
    const int W = depth_buf_.cols;

    for (size_t k = 0; k < proj2d_.size(); ++k)
    {
      const int u = cvRound(proj2d_[k].x);
      const int v = cvRound(proj2d_[k].y);

      if (u < 0 || u >= W || v < 0 || v >= H)
        continue;

      const auto& p = pts3d_[k];

      const double Zc = R_.at<double>(2,0)*p.x + 
                        R_.at<double>(2,1)*p.y + 
                        R_.at<double>(2,2)*p.z + 
                        tvec_.at<double>(2);

      if (Zc <= 0) continue;

      float& depthValue = depth_buf_.at<float>(v, u);
      if (Zc < depthValue)
      {
        depthValue = static_cast<float>(Zc);
        idx_buf_.at<int>(v, u) = static_cast<int>(k);
      }
    }
}

const cv::Mat& Projector::getOverlay()
{
    if (overlay_updated_)
        return overlay_cache_;

    overlay_cache_ = cv::Mat::zeros(LABEL_H, LABEL_W, CV_8UC3);

    for (int v = 0; v < LABEL_H; ++v)
    {
        for (int u = 0; u < LABEL_W; ++u)
        {
            const int idx = idx_buf_.at<int>(v, u);
            if (idx < 0) continue;
            const uchar label = labels_.at<uchar>(v, u);

            int label_idx = label;
            if (label_idx < 0 || label_idx >= semantic_lut_.cols)
                label_idx = 0;

            const cv::Vec3b& color = semantic_lut_.at<cv::Vec3b>(0, label_idx);
            overlay_cache_.at<cv::Vec3b>(v, u) = color;
        }
    }
    overlay_updated_ = true;
    return overlay_cache_;
}

const cv::Mat& Projector::getDepthImage()
{
    if(depth_img_updated_)
        return depth_img_cache_;

    depth_img_cache_ = cv::Mat::zeros(LABEL_H, LABEL_W, CV_8UC3);

    for (int v = 0; v < LABEL_H; ++v)
    {
        for (int u = 0; u < LABEL_W; ++u)
        {
            const int idx = idx_buf_.at<int>(v, u);
            if (idx < 0) continue;

            float z = depth_buf_.at<float>(v, u);

            static const float z_min = static_cast<float>(minRange_);
            static const float z_max = static_cast<float>(maxRange_);

            z = std::max(z_min, std::min(z_max, z)); // clamp
            int lut_idx = static_cast<int>(((z - z_min) / (z_max - z_min)) * 255); // normalize

            const cv::Vec3b& color = depth_lut_.at<cv::Vec3b>(0, lut_idx);
            depth_img_cache_.at<cv::Vec3b>(v, u) = color;
        }
    }
    depth_img_updated_ = true;
    return depth_img_cache_;
}

void Projector::getSemanticClouds(pcl::PointCloud<pcl::PointXYZRGB>& semanticCloud,
                                 pcl::PointCloud<pcl::PointXYZRGB>& travCloud,
                                 pcl::PointCloud<pcl::PointXYZRGB>& obstacleCloud) const
{
    semanticCloud.clear();
    travCloud.clear();
    obstacleCloud.clear();

    const int H = idx_buf_.rows;
    const int W = idx_buf_.cols;

    for (int v = 0; v < H; ++v)
    {
        for (int u = 0; u < W; ++u)
        {
            const int idx = idx_buf_.at<int>(v, u);
            if (idx < 0) continue;

            const uchar label = labels_.at<uchar>(v, u);

            const auto& p = pts3d_[idx];

            pcl::PointXYZRGB pt_rgb;
            pt_rgb.x = p.x;
            pt_rgb.y = p.y;
            pt_rgb.z = p.z;

            if (label < 0 || label >= semantic_lut_.cols)
                continue;

            const cv::Vec3b& color = semantic_lut_.at<cv::Vec3b>(0, label);
            pt_rgb.r = color[2];
            pt_rgb.g = color[1];
            pt_rgb.b = color[0];

            semanticCloud.push_back(pt_rgb);

            if (label == 2)
                travCloud.push_back(pt_rgb);
            else if (label == 1 || label == 0)
                obstacleCloud.push_back(pt_rgb);
        }
    }
}
