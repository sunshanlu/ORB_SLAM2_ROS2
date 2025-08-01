#include <chrono>

#include "ORB_SLAM2/Map.h"

int main()
{
  auto pMapStream = std::make_shared<ORB_SLAM2_ROS2::Map>();
  auto pMapProto = std::make_shared<ORB_SLAM2_ROS2::Map>();

  // Stream Load Test: 测试地图加载时间
  auto start = std::chrono::high_resolution_clock::now();
  ORB_SLAM2_ROS2::Map::loadFromTxtFile("/home/lucky-lu/Projects/ORB_SLAM2_ROS2/map", pMapStream);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Stream Map load time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

  double max_distance = 0;
  auto keyframes = pMapStream->getAllKeyFrames();
  for (auto &kf : keyframes){
    double distance = 0;
    cv::Mat trans = kf->getPose().rowRange(0, 3).col(3);
    for (int idx = 0; idx < 3; ++idx)
      distance += trans.at<float>(idx) * trans.at<float>(idx);
    
    distance = std::sqrt(distance);
    if (distance > max_distance)
      max_distance = distance;
  }

  // Stream Save Test: 测试地图保存时间
  start = std::chrono::high_resolution_clock::now();
  pMapStream->saveToTxtFile("/home/lucky-lu/Projects/ORB_SLAM2_ROS2/map");
  end = std::chrono::high_resolution_clock::now();
  std::cout << "Stream Map save time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

  // Proto Save Test: 测试Proto保存时间
  start = std::chrono::high_resolution_clock::now();
  pMapStream->saveToProtobuf("/home/lucky-lu/Projects/ORB_SLAM2_ROS2/map");
  end = std::chrono::high_resolution_clock::now();
  std::cout << "Proto Map save time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

  // Proto Load Test: 测试Proto加载时间
  start = std::chrono::high_resolution_clock::now();
  ORB_SLAM2_ROS2::Map::loadFromProtobuf("/home/lucky-lu/Projects/ORB_SLAM2_ROS2/map", pMapProto);
  end = std::chrono::high_resolution_clock::now();
  std::cout << "Proto Map load time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


  std::cout << "Max distance in Stream Map: " << max_distance << std::endl;
  return 0;
}