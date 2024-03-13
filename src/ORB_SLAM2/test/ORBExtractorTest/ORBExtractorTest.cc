#include <string>

#include <opencv2/opencv.hpp>

#include "ORB_SLAM2/ORBExtractor.h"

using namespace ORB_SLAM2_ROS2;
int main(int argc, char **argv) {
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Mat> descs;
    std::string imageFp = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_0/000000.png";
    std::string briefFp = "/home/rookie-lu/Project/ORB-SLAM2/ORB-SLAM2-ROS2/config/BRIEF_TEMPLATE.txt";
    cv::Mat image = cv::imread(imageFp, cv::IMREAD_GRAYSCALE);
    ORBExtractor extractor(image, 500, 8, 1.2, briefFp, 15, 8);
    extractor.extract(keypoints, descs);

    cv::drawKeypoints(image, keypoints, image);
    cv::imshow("image with keypoints", image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}