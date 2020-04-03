#ifndef POSE_ESTIMATOR_HPP
#define POSE_ESTIMATOR_HPP

#include <memory>
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <hdl_localization/pose_system.hpp>
#include <kkl/alg/unscented_kalman_filter.hpp>

namespace hdl_localization {

/**
 * @brief scan matching-based pose estimator
 */
class PoseEstimator {
public:
  using PointT = pcl::PointXYZ;

  /**
   * @brief constructor
   * @param registration        registration method
   * @param stamp               timestamp
   * @param pos                 initial position
   * @param quat                initial orientation
   * @param cool_time_duration  during "cool time", prediction is not performed
   */
  PoseEstimator(pcl::Registration<PointT, PointT>::Ptr& registration,
                const ros::Time& stamp,
                const Eigen::Vector3f& pos,
                const Eigen::Quaternionf& quat,
                double cool_time_duration = 1.0,
                double process_pos_noise = 1.0,
                double process_vel_noise = 1.0,
                double process_ori_noise = 0.5,
                double process_ang_vel_noise = 1e-6,
                double process_acc_noise = 1e-6,
                double measure_pos_noise = 0.01,
                double measure_ori_noise = 0.001)
    : init_stamp(stamp),
      registration(registration),
      cool_time_duration(cool_time_duration)
  {
    process_noise = Eigen::MatrixXf::Identity(16, 16);
    process_noise.middleRows(0, 3) *= process_pos_noise; // process position noise
    process_noise.middleRows(3, 3) *= process_vel_noise; // process velocity noise
    process_noise.middleRows(6, 4) *= process_ori_noise; // process orientation noise
    process_noise.middleRows(10, 3) *= process_ang_vel_noise; // process angular vel noise ??
    process_noise.middleRows(13, 3) *= process_acc_noise; // process acceleration noise ??

    Eigen::MatrixXf measurement_noise = Eigen::MatrixXf::Identity(7, 7);
    measurement_noise.middleRows(0, 3) *= measure_pos_noise; // measurement position noise
    measurement_noise.middleRows(3, 4) *= measure_ori_noise; // measurement orientation noise

    Eigen::VectorXf mean(16); // init state
    mean.middleRows(0, 3) = pos;
    mean.middleRows(3, 3).setZero();
    mean.middleRows(6, 4) = Eigen::Vector4f(quat.w(), quat.x(), quat.y(), quat.z());
    mean.middleRows(10, 3).setZero();
    mean.middleRows(13, 3).setZero();

    Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(16, 16) * 0.01; // init covariance

    PoseSystem system;
    ukf.reset(new kkl::alg::UnscentedKalmanFilterX<float, PoseSystem>(system, 16, 6, 7, process_noise, measurement_noise, mean, cov));
  }

  /**
   * @brief predict
   * @param stamp    timestamp
   * @param acc      acceleration
   * @param gyro     angular velocity
   */
  void predict(const ros::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro) {
    if((stamp - init_stamp).toSec() < cool_time_duration || prev_stamp.is_zero() || prev_stamp == stamp) {
      prev_stamp = stamp;
      return;
    }

    double dt = (stamp - prev_stamp).toSec();
    prev_stamp = stamp;

    ukf->setProcessNoiseCov(process_noise * dt);
    ukf->system.dt = dt;

    Eigen::VectorXf control(6);
    control.head<3>() = acc;
    control.tail<3>() = gyro;

    ukf->predict(control);
  }

  /**
   * @brief get_predicted_trans
   * @return init_guess (predicted)
   */
  Eigen::Matrix4f get_predicted_trans() const {
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    init_guess.block<3, 3>(0, 0) = quat().toRotationMatrix();
    init_guess.block<3, 1>(0, 3) = pos();
    return init_guess;
  }


  /**
   * @brief correct
   * @param cloud   input cloud
   * @return cloud aligned to the globalmap
   */
  void correct(const Eigen::Matrix4f &trans) {
    Eigen::Vector3f p = trans.block<3, 1>(0, 3);
    Eigen::Quaternionf q(trans.block<3, 3>(0, 0));

    if(quat().coeffs().dot(q.coeffs()) < 0.0f) {
      q.coeffs() *= -1.0f;
    }

    Eigen::VectorXf observation(7);
    observation.middleRows(0, 3) = p;
    observation.middleRows(3, 4) = Eigen::Vector4f(q.w(), q.x(), q.y(), q.z());

    ukf->correct(observation);
  }

  /* getters */
  Eigen::Vector3f pos() const {
    return Eigen::Vector3f(ukf->mean[0], ukf->mean[1], ukf->mean[2]);
  }

  Eigen::Vector3f vel() const {
    return Eigen::Vector3f(ukf->mean[3], ukf->mean[4], ukf->mean[5]);
  }

  Eigen::Quaternionf quat() const {
    return Eigen::Quaternionf(ukf->mean[6], ukf->mean[7], ukf->mean[8], ukf->mean[9]).normalized();
  }

  Eigen::Matrix4f matrix() const {
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m.block<3, 3>(0, 0) = quat().toRotationMatrix();
    m.block<3, 1>(0, 3) = pos();
    return m;
  }

private:
  ros::Time init_stamp;         // when the estimator was initialized
  ros::Time prev_stamp;         // when the estimator was updated last time
  double cool_time_duration;    //

  Eigen::MatrixXf process_noise;
  std::unique_ptr<kkl::alg::UnscentedKalmanFilterX<float, PoseSystem>> ukf;

  pcl::Registration<PointT, PointT>::Ptr registration;
};

}

#endif // POSE_ESTIMATOR_HPP
