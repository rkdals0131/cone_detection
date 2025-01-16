#include "cone_detection/cone_detection_node.h"
#include <memory>

namespace LIDAR {

// OutlierFilter 클래스 생성자: ROS2 노드 초기화 및 설정
OutlierFilter::OutlierFilter()
    : Node("outlier_filter") {
        
    // ROS2 파라미터 선언
    this->declare_parameter("z_threshold_enable", params_.z_threshold_enable);
    this->declare_parameter("z_threshold_min", params_.z_threshold_min);
    this->declare_parameter("z_threshold_max", params_.z_threshold_max);
    this->declare_parameter("min_distance", params_.min_distance);
    this->declare_parameter("max_distance", params_.max_distance);
    this->declare_parameter("intensity_threshold", params_.intensity_threshold);
    this->declare_parameter("plane_distance_threshold", params_.plane_distance_threshold);
    this->declare_parameter("roi_angle_min", params_.roi_angle_min);
    this->declare_parameter("roi_angle_max", params_.roi_angle_max);
    this->declare_parameter("voxel_leaf_size", params_.voxel_leaf_size);
    this->declare_parameter("ec_cluster_tolerance", params_.ec_cluster_tolerance);
    this->declare_parameter("ec_min_cluster_size", params_.ec_min_cluster_size);
    this->declare_parameter("ec_max_cluster_size", params_.ec_max_cluster_size);

    // Load parameters from Config file
    this->get_parameter("z_threshold_enable", params_.z_threshold_enable);
    this->get_parameter("z_threshold_min", params_.z_threshold_min);
    this->get_parameter("z_threshold_max", params_.z_threshold_max);
    this->get_parameter("min_distance", params_.min_distance);
    this->get_parameter("max_distance", params_.max_distance);
    this->get_parameter("intensity_threshold", params_.intensity_threshold);
    this->get_parameter("plane_distance_threshold", params_.plane_distance_threshold);
    this->get_parameter("roi_angle_min", params_.roi_angle_min);
    this->get_parameter("roi_angle_max", params_.roi_angle_max);
    this->get_parameter("voxel_leaf_size", params_.voxel_leaf_size);
    this->get_parameter("ec_cluster_tolerance", params_.ec_cluster_tolerance);
    this->get_parameter("ec_min_cluster_size", params_.ec_min_cluster_size);
    this->get_parameter("ec_max_cluster_size", params_.ec_max_cluster_size);

    // Log loaded parameters for verification
    RCLCPP_INFO(this->get_logger(), "Loaded Parameters:");
    RCLCPP_INFO(this->get_logger(), "  z_threshold_enable: %s", params_.z_threshold_enable ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "  z_threshold_min: %.2f", params_.z_threshold_min);
    RCLCPP_INFO(this->get_logger(), "  z_threshold_max: %.2f", params_.z_threshold_max);
    RCLCPP_INFO(this->get_logger(), "  min_distance: %.2f", params_.min_distance);
    RCLCPP_INFO(this->get_logger(), "  max_distance: %.2f", params_.max_distance);
    RCLCPP_INFO(this->get_logger(), "  intensity_threshold: %.2f", params_.intensity_threshold);
    RCLCPP_INFO(this->get_logger(), "  plane_distance_threshold: %.2f", params_.plane_distance_threshold);
    RCLCPP_INFO(this->get_logger(), "  roi_angle_min: %.2f", params_.roi_angle_min);
    RCLCPP_INFO(this->get_logger(), "  roi_angle_max: %.2f", params_.roi_angle_max);
    RCLCPP_INFO(this->get_logger(), "  voxel_leaf_size: %.2f", params_.voxel_leaf_size);
    RCLCPP_INFO(this->get_logger(), "  ec_cluster_tolerance: %.2f", params_.ec_cluster_tolerance);
    RCLCPP_INFO(this->get_logger(), "  ec_min_cluster_size: %d", params_.ec_min_cluster_size);
    RCLCPP_INFO(this->get_logger(), "  ec_max_cluster_size: %d", params_.ec_max_cluster_size);


    // 퍼블리셔 초기화 (마커, 정렬된 콘, 처리된 포인트 클라우드)
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/visualization_marker", 10);
    cones_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/sorted_cones", 10);
    pub_cones_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/point_cones", 10);

    // 서브스크라이버 초기화 (포인트 클라우드 데이터 수신)
    point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/ouster/points", rclcpp::SensorDataQoS(), // <-- QoS '10' -> 'rclcpp::SensorDataQoS()'로 바꿈.
        std::bind(&OutlierFilter::callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Cone_detection_node has been started!!!!!!!!!!!!!!!!!!!");  // 노드 시작 로그 출력
}

// 콜백 함수: 수신된 포인트 클라우드 데이터를 처리
void OutlierFilter::callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    Cloud::Ptr cloud_in(new Cloud), cloud_out(new Cloud);

    // ROS 메시지를 PCL 포인트 클라우드로 변환
    pcl::fromROSMsg(*msg, *cloud_in);

    // 이상점 제거 및 필터링 수행
    filterPointCloud(cloud_in, cloud_out);

    // 필터링된 포인트 클라우드를 퍼블리싱
    publishCloud(pub_cones_cloud_, cloud_out);

    // 클러스터링 및 결과 퍼블리싱
    std::vector<ConeDescriptor> cones;
    clusterCones(cloud_out, cones);

    // 콘 정렬 및 결과 퍼블리싱
    std::vector<std::vector<double>> sorted_cones = sortCones(cones);
    publishArray(cones_pub_, sorted_cones);

    // 콘 데이터를 기반으로 MarkerArray 발행
    publishSortedConesMarkers(sorted_cones); // 추가된 부분

    // 콘 시각화
    visualizeCones(cones);
}


void OutlierFilter::voxelizeCloud(Cloud::Ptr &cloud_in, Cloud::Ptr &cloud_out, float leaf_size) {
    pcl::VoxelGrid<Point> voxel_filter;
    voxel_filter.setInputCloud(cloud_in);
    voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_filter.filter(*cloud_out);
}


// 포인트 클라우드 필터링 함수
void OutlierFilter::filterPointCloud(Cloud::Ptr &cloud_in, Cloud::Ptr &cloud_out) {
    Cloud::Ptr downsampled_cloud(new Cloud);

    // Voxelization (downsampling)
    voxelizeCloud(cloud_in, downsampled_cloud, params_.voxel_leaf_size);

    std::vector<Point> filtered_points;

    for (const auto &point : downsampled_cloud->points) {
        float angle = ROI_theta(point.y, point.x);
        float distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);

        // Config 파일에서 로드된 파라미터로 필터링
        if ((angle >= params_.roi_angle_min && angle <= params_.roi_angle_max) &&  // ROI 각도 범위
            point.x >= 0 &&
            distance >= params_.min_distance &&  // 최소 거리 조건 추가
            distance <= params_.max_distance &&
            point.z >= params_.z_threshold_min && point.z <= params_.z_threshold_max &&
            point.intensity >= params_.intensity_threshold) {
            filtered_points.push_back(point);
        }
    }

    // 필터링된 포인트 클라우드를 cloud_out에 복사
    cloud_out->points.clear();
    cloud_out->points.insert(cloud_out->points.end(), filtered_points.begin(), filtered_points.end());
    cloud_out->width = filtered_points.size();
    cloud_out->height = 1;
    cloud_out->is_dense = downsampled_cloud->is_dense;

    // 평면 제거를 위한 RANSAC 세그먼테이션
    pcl::ModelCoefficients::Ptr plane_coefs(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<Point> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(params_.plane_distance_threshold);
    seg.setInputCloud(cloud_out);
    seg.segment(*inliers, *plane_coefs);

    // 평면 포인트 제거
    if (!inliers->indices.empty()) {
        pcl::ExtractIndices<Point> extract;
        extract.setInputCloud(cloud_out);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud_out);
    }
}


// 클러스터링 수행 (콘 클러스터 식별)
void OutlierFilter::clusterCones(Cloud::Ptr &cloud_out, std::vector<ConeDescriptor> &cones) {
    pcl::EuclideanClusterExtraction<Point> ec;
    std::vector<pcl::PointIndices> cluster_indices;
    ec.setClusterTolerance(params_.ec_cluster_tolerance);
    ec.setMinClusterSize(params_.ec_min_cluster_size);
    ec.setMaxClusterSize(params_.ec_max_cluster_size);
    ec.setInputCloud(cloud_out);
    ec.extract(cluster_indices);

    pcl::ExtractIndices<Point> extract;
    extract.setInputCloud(cloud_out);

    // 각 클러스터를 ConeDescriptor로 변환
    cones.reserve(cluster_indices.size());
    for (const auto &indices : cluster_indices) {
        ConeDescriptor cone;
        pcl::PointIndices::Ptr indices_ptr(new pcl::PointIndices(indices));
        extract.setIndices(indices_ptr);
        extract.filter(*cone.cloud);
        cone.calculate();
        cones.push_back(cone);
    }
}

// 클러스터된 콘을 정렬
std::vector<std::vector<double>> OutlierFilter::sortCones(const std::vector<ConeDescriptor> &cones) {
    std::vector<std::vector<double>> sorted_cones;

    for (const auto &cone : cones) {
        sorted_cones.push_back({cone.mean.x, cone.mean.y});
    }

    // x축을 기준으로 정렬
    std::sort(sorted_cones.begin(), sorted_cones.end(),
              [](const std::vector<double> &a, const std::vector<double> &b) {
                  return a[0] < b[0];
              });

    return sorted_cones;
}

// 포인트 클라우드 퍼블리싱
void OutlierFilter::publishCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &publisher,
    Cloud::Ptr &cloud) {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = "os_sensor";
    cloud_msg.header.stamp = this->now();
    publisher->publish(cloud_msg);
}

// 정렬된 콘 데이터를 퍼블리싱
void OutlierFilter::publishArray(
    const rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr &publisher,
    const std::vector<std::vector<double>> &array) {
    std_msgs::msg::Float64MultiArray msg;

    msg.layout.dim.resize(2);
    if (!array.empty()) {
        msg.layout.dim[0].size = array.size();
        msg.layout.dim[1].size = array[0].size();
        msg.layout.dim[0].stride = array.size() * array[0].size();
        msg.layout.dim[1].stride = array[0].size();
    }

    for (const auto &row : array) {
        for (const auto &val : row) {
            msg.data.push_back(val);
        }
    }

    publisher->publish(msg);
}

// 콘 클러스터를 시각화
void OutlierFilter::visualizeCones(const std::vector<ConeDescriptor> &cones) {
    visualization_msgs::msg::MarkerArray markers;
    int id = 0;

    for (const auto &cone : cones) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "os_sensor";
        marker.header.stamp = this->now();
        marker.ns = "cones";
        marker.id = id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = cone.mean.x;
        marker.pose.position.y = cone.mean.y;
        marker.pose.position.z = 0.3;
        marker.scale.x = 0.3;
        marker.scale.y = 0.3;
        marker.scale.z = 0.3;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        markers.markers.push_back(marker);
    }

    marker_pub_->publish(markers);
}

void OutlierFilter::publishSortedConesMarkers(const std::vector<std::vector<double>> &sorted_cones) {
    visualization_msgs::msg::MarkerArray markers;
    int id = 0;

    for (const auto &cone : sorted_cones) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "os_sensor";
        marker.header.stamp = this->now();
        marker.ns = "sorted_cones";
        marker.id = id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        // Assign x, y from sorted_cones and set a fixed z value
        marker.pose.position.x = cone[0];
        marker.pose.position.y = cone[1];
        marker.pose.position.z = 0.3;  // Fixed height for visualization
        marker.scale.x = 0.3;
        marker.scale.y = 0.3;
        marker.scale.z = 0.3;

        // Color settings (red as an example)
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        markers.markers.push_back(marker);
    }

    marker_pub_->publish(markers);  // Publish the marker array
}


// ROI 영역의 각도를 계산
float OutlierFilter::ROI_theta(float x, float y) {
    return std::atan2(y, x) * 180 / M_PI;
}

}  // namespace LIDAR

// 프로그램 진입점 (main 함수)
int main(int argc, char **argv) {
    // ROS2 노드 초기화
    rclcpp::init(argc, argv);

    // OutlierFilter 노드 생성 및 실행
    auto node = std::make_shared<LIDAR::OutlierFilter>();
    rclcpp::spin(node);

    // ROS2 노드 종료
    rclcpp::shutdown();
    return 0;
}
