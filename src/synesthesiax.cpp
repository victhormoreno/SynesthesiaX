#include "Projector.hpp"

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

ros::Publisher pc_on_img_pub, pc_color_pub, pub_obstacles, pub_traversable, depth_img_pub;

// Projector instance (initialized later)
Projector projector;

using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image>;
std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> pc_sub;
std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> img_sub;
std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_ptr;

void callback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
              const sensor_msgs::ImageConstPtr &image_msg)
{
    // 1. Process the input cloud and image: this will filter points, project the cloud and build depth map
    projector.project_cloud_onto_image(cloud_msg, image_msg);

    // 2. Get the semantic colored clouds and overlay image (optional)
    pcl::PointCloud<pcl::PointXYZRGB> semanticCloud, travCloud, obstacleCloud;
    projector.getSemanticClouds(semanticCloud, travCloud, obstacleCloud);

    // Publish overlay image
    if(pc_on_img_pub.getNumSubscribers() > 0){
        const cv::Mat& overlay = projector.getOverlay();
        sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", overlay).toImageMsg();
        img_msg->header = cloud_msg->header;  // Keep timestamp/header synced
        pc_on_img_pub.publish(img_msg);
    }

    // Publish depth image
    if(depth_img_pub.getNumSubscribers() > 0){
        const cv::Mat& depth = projector.getDepthImage();
        sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", depth).toImageMsg();
        img_msg->header = cloud_msg->header;  // Keep timestamp/header synced
        depth_img_pub.publish(img_msg);
    }

    // Publish semantic colored clouds
    sensor_msgs::PointCloud2 pc_color_msg, pc_obstacles_msg, pc_traversable_msg;
    pcl::toROSMsg(semanticCloud, pc_color_msg);
    pcl::toROSMsg(obstacleCloud, pc_obstacles_msg);
    pcl::toROSMsg(travCloud, pc_traversable_msg);
    pc_color_msg.header = pc_obstacles_msg.header = pc_traversable_msg.header = cloud_msg->header;
    pc_color_pub.publish(pc_color_msg);
    pub_obstacles.publish(pc_obstacles_msg);
    pub_traversable.publish(pc_traversable_msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "synesthesiax");
    ros::NodeHandle nh("~");
    
    std::string cloud_topic, img_topic;
    nh.param("cloud_topic", cloud_topic, std::string("/lidar/points"));
    nh.param("img_topic", img_topic, std::string("/camera/labels"));

    if (!projector.init(nh))
    {
        ROS_FATAL("Unable to load calibration parameters, aborting");
        return -1;
    }

    pc_on_img_pub   = nh.advertise<sensor_msgs::Image>("cloud_onto_img", 1);
    depth_img_pub   = nh.advertise<sensor_msgs::Image>("depth_img", 1);
    pc_color_pub    = nh.advertise<sensor_msgs::PointCloud2>("semantic_cloud", 1);
    pub_obstacles   = nh.advertise<sensor_msgs::PointCloud2>("obstacles", 1);
    pub_traversable = nh.advertise<sensor_msgs::PointCloud2>("traversable", 1);

    pc_sub = std::make_unique<message_filters::Subscriber<sensor_msgs::PointCloud2>>(nh, cloud_topic, 1);
    img_sub = std::make_unique<message_filters::Subscriber<sensor_msgs::Image>>(nh, img_topic, 1);

    sync_ptr = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), *pc_sub, *img_sub);
    sync_ptr->registerCallback(boost::bind(&callback, _1, _2));

    ros::spin();

    return 0;
}
