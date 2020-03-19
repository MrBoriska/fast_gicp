#include <mutex>
#include <memory>
#include <iostream>
#include <boost/circular_buffer.hpp>

#include <ros/ros.h>
#include <ros/time.h>
#include <ros/duration.h>
#include <pcl_ros/point_cloud.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include <std_msgs/Time.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/approximate_voxel_grid.h>

//#include <hdl_graph_slam/ros_utils.hpp>
//#include <hdl_graph_slam/registrations.hpp>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#ifdef USE_VGICP_CUDA
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

#include <hdl_localization/pose_estimator.hpp>

namespace fast_gicp {

class FastGICPOdometryNodelet : public nodelet::Nodelet {
public:
  typedef pcl::PointXYZ PointT;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FastGICPOdometryNodelet() {}
  virtual ~FastGICPOdometryNodelet() {}

  virtual void onInit() {
    NODELET_DEBUG("initializing scan_matching_odometry_nodelet...");
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    initialize_params();
    
    if(use_imu) {
      NODELET_INFO("enable imu-based prediction");
      imu_sub = mt_nh.subscribe("/imu/data", 256, &FastGICPOdometryNodelet::imu_callback, this);
    }

    points_sub = nh.subscribe(points_topic, 256, &FastGICPOdometryNodelet::cloud_callback, this);
    initialpose_sub = nh.subscribe("/initialpose", 8, &FastGICPOdometryNodelet::initialpose_callback, this);
    read_until_pub = nh.advertise<std_msgs::Header>("/scan_matching_odometry/read_until", 32);
    odom_pub = nh.advertise<nav_msgs::Odometry>("/odom", 32);

    predict_trans_pub = nh.advertise<geometry_msgs::Pose>("/predict_trans", 32);
    final_trans_pub = nh.advertise<geometry_msgs::Pose>("/final_trans", 32);
  }

private:
  /**
   * @brief initialize parameters
   */
  void initialize_params() {
    auto& pnh = private_nh;
    points_topic = pnh.param<std::string>("points_topic", "/velodyne_points");
    odom_frame_id = pnh.param<std::string>("odom_frame_id", "odom");
    publish_tf = pnh.param<bool>("publish_tf", true);

    use_imu = private_nh.param<bool>("use_imu", true);
    invert_imu = private_nh.param<bool>("invert_imu", false);

    // The minimum tranlational distance and rotation angle between keyframes.
    // If this value is zero, frames are always compared with the previous frame
    keyframe_delta_trans = pnh.param<double>("keyframe_delta_trans", 0.25);
    keyframe_delta_angle = pnh.param<double>("keyframe_delta_angle", 0.15);
    keyframe_delta_time = pnh.param<double>("keyframe_delta_time", 1.0);

    // Registration validation by thresholding
    transform_thresholding = pnh.param<bool>("transform_thresholding", false);
    max_acceptable_trans = pnh.param<double>("max_acceptable_trans", 1.0);
    max_acceptable_angle = pnh.param<double>("max_acceptable_angle", 1.0);

    // select a downsample method (VOXELGRID, APPROX_VOXELGRID, NONE)
    std::string downsample_method = pnh.param<std::string>("downsample_method", "VOXELGRID");
    double downsample_resolution = pnh.param<double>("downsample_resolution", 0.1);
    if(downsample_method == "VOXELGRID") {
      std::cout << "downsample: VOXELGRID " << downsample_resolution << std::endl;
      boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter = voxelgrid;
    } else if(downsample_method == "APPROX_VOXELGRID") {
      std::cout << "downsample: APPROX_VOXELGRID " << downsample_resolution << std::endl;
      boost::shared_ptr<pcl::ApproximateVoxelGrid<PointT>> approx_voxelgrid(new pcl::ApproximateVoxelGrid<PointT>());
      approx_voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter = approx_voxelgrid;
    } else {
      if(downsample_method != "NONE") {
        std::cerr << "warning: unknown downsampling type (" << downsample_method << ")" << std::endl;
        std::cerr << "       : use passthrough filter" <<std::endl;
      }
      std::cout << "downsample: NONE" << std::endl;
      boost::shared_ptr<pcl::PassThrough<PointT>> passthrough(new pcl::PassThrough<PointT>());
      downsample_filter = passthrough;
    }

    //registration = select_registration_method(pnh);
    boost::shared_ptr<fast_gicp::FastVGICPCuda<PointT, PointT>> fast_gicp_cuda(new fast_gicp::FastVGICPCuda<PointT, PointT>());;
    


    fast_gicp_cuda->setTransformationEpsilon(pnh.param<double>("transformation_epsilon", 0.01));
    fast_gicp_cuda->setMaximumIterations(pnh.param<int>("maximum_iterations", 64));
    //fast_gicp_cuda->setUseReciprocalCorrespondences(pnh.param<bool>("use_reciprocal_correspondences", false));
    fast_gicp_cuda->setCorrespondenceRandomness(pnh.param<int>("gicp_correspondence_randomness", 20));
    //fast_gicp_cuda->setMaximumOptimizerIterations(pnh.param<int>("gicp_max_optimizer_iterations", 20));


    fast_gicp_cuda->setNearesetNeighborSearchMethod(fast_gicp::CPU_PARALLEL_KDTREE);
    fast_gicp_cuda->setResolution(pnh.param<double>("vgicp_resolution", 1.0));
    registration = fast_gicp_cuda;
    
    // init values for calc velocity by positions(for odometry)
    prev_odom.header.stamp = ros::Time(0.0);
    prev_odom.header.frame_id = odom_frame_id;
    prev_odom.pose.pose.position.x = 0.0;
    prev_odom.pose.pose.position.y = 0.0;
    prev_odom.pose.pose.position.z = 0.0;
    prev_odom.pose.pose.orientation.x = 0.0;
    prev_odom.pose.pose.orientation.y = 0.0;
    prev_odom.pose.pose.orientation.z = 0.0;



    // initialize pose estimator
    if(private_nh.param<bool>("specify_init_pose", true)) {
      NODELET_INFO("initialize pose estimator with specified parameters!!");
      pose_estimator.reset(new hdl_localization::PoseEstimator(registration,
        ros::Time::now(),
        Eigen::Vector3f(
          private_nh.param<double>("init_pos_x", 0.0),
          private_nh.param<double>("init_pos_y", 0.0),
          private_nh.param<double>("init_pos_z", 0.0)
        ),
        Eigen::Quaternionf(
          private_nh.param<double>("init_ori_w", 1.0),
          private_nh.param<double>("init_ori_x", 0.0),
          private_nh.param<double>("init_ori_y", 0.0),
          private_nh.param<double>("init_ori_z", 0.0)
        ),
        private_nh.param<double>("cool_time_duration", 0.5),

        private_nh.param<double>("process_pos_noise", 1.0),
        private_nh.param<double>("process_vel_noise", 1.0),
        private_nh.param<double>("process_ori_noise", 0.5),
        private_nh.param<double>("process_ang_vel_noise", 1e-6),
        private_nh.param<double>("process_acc_noise", 1e-6),
        private_nh.param<double>("measure_pos_noise", 0.01),
        private_nh.param<double>("measure_ori_noise", 0.001)
      ));
    }

  }

  /**
   * @brief callback for point clouds
   * @param cloud_msg  point cloud msg
   */
  void cloud_callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    if(!ros::ok()) {
      return;
    }

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);
    
    Eigen::Matrix4f pose = matching(cloud_msg->header.stamp, cloud);

    publish_odometry(cloud_msg->header.stamp, cloud_msg->header.frame_id, pose);

    // In offline estimation, point clouds until the published time will be supplied
    std_msgs::HeaderPtr read_until(new std_msgs::Header());
    read_until->frame_id = points_topic;
    read_until->stamp = cloud_msg->header.stamp + ros::Duration(1, 0);
    read_until_pub.publish(read_until);

    read_until->frame_id = "/filtered_points";
    read_until_pub.publish(read_until);

  }

  /**
   * @brief callback for imu data
   * @param imu_msg
   */
  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    std::lock_guard<std::mutex> lock(imu_data_mutex);
    imu_data.push_back(imu_msg);
  }

  /**
   * @brief callback for initial pose input ("2D Pose Estimate" on rviz)
   * @param pose_msg
   */
  void initialpose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg) {
    NODELET_INFO("initial pose received!!");
    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    const auto& p = pose_msg->pose.pose.position;
    const auto& q = pose_msg->pose.pose.orientation;
    pose_estimator.reset(
          new hdl_localization::PoseEstimator(
            registration,
            ros::Time::now(),
            Eigen::Vector3f(p.x, p.y, p.z),
            Eigen::Quaternionf(q.w, q.x, q.y, q.z),
            private_nh.param<double>("cool_time_duration", 0.5))
    );
  }

  /**
   * @brief downsample a point cloud
   * @param cloud  input cloud
   * @return downsampled point cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);

    return filtered;
  }

  /**
   * @brief estimate the relative pose between an input cloud and a keyframe cloud
   * @param stamp  the timestamp of the input cloud
   * @param cloud  the input cloud
   * @return the relative pose between the input cloud and the keyframe cloud
   */
  Eigen::Matrix4f matching(const ros::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    if(!keyframe) {
      prev_trans.setIdentity();
      keyframe_pose.setIdentity();
      keyframe_stamp = stamp;
      keyframe = downsample(cloud);
      registration->setInputTarget(keyframe);

      //imu_data[0] = imu_data[1];

      return Eigen::Matrix4f::Identity();
    }

    auto filtered = downsample(cloud);

    registration->setInputSource(filtered);



    // predict
    if(!use_imu) {
      pose_estimator->predict(stamp, Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero());
    } else {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      auto imu_iter = imu_data.begin();
      for(imu_iter; imu_iter != imu_data.end(); imu_iter++) {
        if(stamp < (*imu_iter)->header.stamp) {
          break;
        }
        const auto& acc = (*imu_iter)->linear_acceleration;
        const auto& gyro = (*imu_iter)->angular_velocity;
        double gyro_sign = invert_imu ? -1.0 : 1.0;
        pose_estimator->predict((*imu_iter)->header.stamp, Eigen::Vector3f(acc.x, acc.y, acc.z), gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
      }
      imu_data.erase(imu_data.begin(), imu_iter);
    }

    /*Eigen::Matrix4f init_guess = Eigen::Matrix4f(prev_trans);
    if (!use_imu) {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      const auto& ori0 = imu_data[0].orientation;
      Eigen::Quaternionf quat0 = Eigen::Quaternionf(ori0.w,ori0.x,ori0.y,ori0.z).inverse();
      const auto& ori1 = imu_data[1].orientation;
      Eigen::Quaternionf quat1 = (quat0*Eigen::Quaternionf(ori1.w,ori1.x,ori1.y,ori1.z)).normalized();

      init_guess.block<3, 3>(0, 0) = quat1.toRotationMatrix();

      geometry_msgs::Pose predict_trans_msg;
      //predict_trans_msg.position = imu_data[0].position;
      predict_trans_msg.orientation.x = quat1.x();
      predict_trans_msg.orientation.y = quat1.y();
      predict_trans_msg.orientation.z = quat1.z();
      predict_trans_msg.orientation.w = quat1.w();
      predict_trans_pub.publish(predict_trans_msg);

      imu_data[0] = imu_data[1];
    }*/




    geometry_msgs::Pose init_guess_msg;
    Eigen::Quaternionf predict_q = pose_estimator->quat();
    init_guess_msg.orientation.x = predict_q.x();
    init_guess_msg.orientation.y = predict_q.y();
    init_guess_msg.orientation.z = predict_q.z();
    init_guess_msg.orientation.w = predict_q.w();
    predict_trans_pub.publish(init_guess_msg);



    auto t1 = std::chrono::high_resolution_clock::now();
    auto aligned = pose_estimator->correct(filtered);
    //pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    //registration->align(*aligned, init_guess);
    auto t2 = std::chrono::high_resolution_clock::now();
    double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
    //NODELET_INFO_STREAM("processing_time: " << single << "[msec]");
    
  
    if(!registration->hasConverged()) {
      NODELET_INFO_STREAM("scan matching has not converged!!");
      NODELET_INFO_STREAM("ignore this frame(" << stamp << ")");
      return keyframe_pose * prev_trans;
    }

    Eigen::Matrix4f trans = registration->getFinalTransformation();

    geometry_msgs::Pose final_trans_msg;
    auto final_trans = matrix2transform(stamp, trans, "root_link", "keyframe");
    //final_trans_msg.position = final_trans.transform.translation;
    final_trans_msg.orientation = final_trans.transform.rotation;
    final_trans_pub.publish(final_trans_msg);

/*
    if (!use_imu) {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      const auto& ori0 = imu_data[0].orientation;
      Eigen::Quaternionf quat0 = Eigen::Quaternionf(ori0.w,ori0.x,ori0.y,ori0.z).inverse().normalized();
      const auto& ori1 = imu_data[1].orientation;
      Eigen::Quaternionf quat1 = (Eigen::Quaternionf(ori1.w,ori1.x,ori1.y,ori1.z)*quat0).normalized();

      //trans.block<3, 3>(0, 0) = quat1.toRotationMatrix();

      geometry_msgs::Pose predict_trans_msg;
      //predict_trans_msg.position = imu_data[0].position;
      predict_trans_msg.orientation.x = quat1.x();
      predict_trans_msg.orientation.y = quat1.y();
      predict_trans_msg.orientation.z = quat1.z();
      predict_trans_msg.orientation.w = quat1.w();
      predict_trans_pub.publish(predict_trans_msg);


      imu_data[0] = imu_data[1];
    }*/

    Eigen::Matrix4f odom = keyframe_pose * trans;

    if(transform_thresholding) {
      Eigen::Matrix4f delta = prev_trans.inverse() * trans;
      double dx = delta.block<3, 1>(0, 3).norm();
      double da = std::acos(Eigen::Quaternionf(delta.block<3, 3>(0, 0)).w());

      if(dx > max_acceptable_trans || da > max_acceptable_angle) {
        NODELET_INFO_STREAM("too large transform!!  " << dx << "[m] " << da << "[rad]");
        NODELET_INFO_STREAM("ignore this frame(" << stamp << ")");
        return keyframe_pose * prev_trans;
      }
    }

    prev_trans = trans;

    auto keyframe_trans = matrix2transform(stamp, keyframe_pose, odom_frame_id, "keyframe");
    keyframe_broadcaster.sendTransform(keyframe_trans);

    double delta_trans = trans.block<3, 1>(0, 3).norm();
    double delta_angle = std::acos(Eigen::Quaternionf(trans.block<3, 3>(0, 0)).w());
    double delta_time = (stamp - keyframe_stamp).toSec();
    if(delta_trans > keyframe_delta_trans || delta_angle > keyframe_delta_angle || delta_time > keyframe_delta_time) {
      keyframe = filtered;
      registration->setInputTarget(keyframe);

      keyframe_pose = odom;
      keyframe_stamp = stamp;
      prev_trans.setIdentity();
    }

    return odom;
  }

  /**
   * @brief publish odometry
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  void publish_odometry(const ros::Time& stamp, const std::string& cloud_frame_id, const Eigen::Matrix4f& pose) {
    // broadcast the transform over tf
    geometry_msgs::TransformStamped odom_trans = matrix2transform(stamp, pose, odom_frame_id, cloud_frame_id);

    // publish the transform
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_id;

    odom.pose.pose.position.x = pose(0, 3);
    odom.pose.pose.position.y = pose(1, 3);
    odom.pose.pose.position.z = pose(2, 3);
    odom.pose.pose.orientation = odom_trans.transform.rotation;

    odom.child_frame_id = cloud_frame_id;
    //odom.twist.twist.linear.x = 0.0;
    //odom.twist.twist.linear.y = 0.0;
    //odom.twist.twist.angular.z = 0.0;

    prev_odom.child_frame_id = cloud_frame_id;
    /*double x = odom.pose.pose.position.x;
    double px = prev_odom.pose.pose.position.x;
    double y = odom.pose.pose.position.y;
    double py = prev_odom.pose.pose.position.y;
    double ox = odom.pose.pose.orientation.x;
    double pox = prev_odom.pose.pose.orientation.x;
    double oy = odom.pose.pose.orientation.y;
    double poy = prev_odom.pose.pose.orientation.y;
    double oz = odom.pose.pose.orientation.z;
    double poz = prev_odom.pose.pose.orientation.z;
    double dt = (stamp-prev_odom.header.stamp).toSec();

    odom.twist.twist.linear.x = (x-px)/dt;
    odom.twist.twist.linear.y = (y-py)/dt;
    odom.twist.twist.angular.x = (ox-pox)/dt;
    odom.twist.twist.angular.y = (oy-poy)/dt;
    odom.twist.twist.angular.z = (oz-poz)/dt;

    odom.pose.covariance =
			{	.001,		.0,		.0,		.0,		.0,		.0,
				.0,		.001,		.0,		.0,		.0,		.0,
				.0,		.0,		.01, 		.0,		.0,		.0,
				.0,		.0,		.0,		.01,		.0,		.0,
				.0,		.0,		.0,		.0,		.01,		.0,
				.0,		.0,		.0,		.0,		.0,		.001		};
		odom.twist.covariance =
			{	.001,		.0,		.0,		.0,		.0,		.0,
				.0,		.001,		.0,		.0,		.0,		.0,
				.0,		.0,		.01, 		.0,		.0,		.0,
				.0,		.0,		.0,		.01,		.0,		.0,
				.0,		.0,		.0,		.0,		.01,		.0,
				.0,		.0,		.0,		.0,		.0,		.001		};*/


    prev_odom = odom;
    odom_pub.publish(odom);
    if (publish_tf) {
      odom_broadcaster.sendTransform(odom_trans);
    }
  }
  /**
   * @brief convert Eigen::Matrix to geometry_msgs::TransformStamped
   * @param stamp            timestamp
   * @param pose             Eigen::Matrix to be converted
   * @param frame_id         tf frame_id
   * @param child_frame_id   tf child frame_id
   * @return converted TransformStamped
   */
  static geometry_msgs::TransformStamped matrix2transform(const ros::Time& stamp, const Eigen::Matrix4f& pose, const std::string& frame_id, const std::string& child_frame_id) {
    Eigen::Quaternionf quat(pose.block<3, 3>(0, 0));
    quat.normalize();
    geometry_msgs::Quaternion odom_quat;
    odom_quat.w = quat.w();
    odom_quat.x = quat.x();
    odom_quat.y = quat.y();
    odom_quat.z = quat.z();

    geometry_msgs::TransformStamped odom_trans;
    odom_trans.header.stamp = stamp;
    odom_trans.header.frame_id = frame_id;
    odom_trans.child_frame_id = child_frame_id;

    odom_trans.transform.translation.x = pose(0, 3);
    odom_trans.transform.translation.y = pose(1, 3);
    odom_trans.transform.translation.z = pose(2, 3);
    odom_trans.transform.rotation = odom_quat;

    return odom_trans;
  }

private:
  // ROS topics
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;
  
  bool use_imu;
  bool invert_imu;
  ros::Subscriber imu_sub;
  ros::Subscriber points_sub;
  ros::Subscriber initialpose_sub;

  ros::Publisher odom_pub;
  tf::TransformBroadcaster odom_broadcaster;
  tf::TransformBroadcaster keyframe_broadcaster;

  std::string points_topic;
  std::string odom_frame_id;
  bool publish_tf;
  ros::Publisher read_until_pub;
  ros::Publisher predict_trans_pub;
  ros::Publisher final_trans_pub;

  // imu input buffer
  std::mutex imu_data_mutex;
  std::vector<sensor_msgs::ImuConstPtr> imu_data;

  // keyframe parameters
  double keyframe_delta_trans;  // minimum distance between keyframes
  double keyframe_delta_angle;  //
  double keyframe_delta_time;   //

  // registration validation by thresholding
  bool transform_thresholding;  //
  double max_acceptable_trans;  //
  double max_acceptable_angle;

  // pose estimator
  std::mutex pose_estimator_mutex;
  std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator;

  // processing time buffer
  boost::circular_buffer<double> processing_time;

  // odometry calculation
  Eigen::Matrix4f prev_trans;                  // previous estimated transform from keyframe
  Eigen::Matrix4f keyframe_pose;               // keyframe pose
  ros::Time keyframe_stamp;                    // keyframe time
  pcl::PointCloud<PointT>::ConstPtr keyframe;  // keyframe point cloud

  //
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration;

  nav_msgs::Odometry prev_odom;
};

}

PLUGINLIB_EXPORT_CLASS(fast_gicp::FastGICPOdometryNodelet, nodelet::Nodelet)
