#include <string>

#include <opencv2/opencv.hpp>

#include "ORB_SLAM2/ORBExtractor.h"
#include "ORB_SLAM2/ORBMatcher.h"
#include "ORB_SLAM2_E/ORBextractor.h"

using namespace ORB_SLAM2_ROS2;
using namespace ORB_SLAM2;
int main(int argc, char **argv) {
    std::vector<cv::KeyPoint> keypoints, keypointsO;
    std::vector<cv::Mat> descs;
    cv::Mat descsO;
    std::string imageFp = "/media/rookie-lu/DATA1/Dataset/KITTI-ODOM/00/image_0/000634.png";
    std::string briefFp = "/home/rookie-lu/Project/ORB-SLAM2/ORB-SLAM2-ROS2/config/BRIEF_TEMPLATE.txt";
    cv::Mat image = cv::imread(imageFp, cv::IMREAD_GRAYSCALE);
    std::cout << std::endl;

    ORBExtractor extractor(image, 500, 1, 1.2, briefFp, 20, 8);
    extractor.extract(keypoints, descs);

    ORBextractor extractorO(2000, 1.2, 1, 20, 8);
    extractorO(image, cv::noArray(), keypointsO, descsO);

    cv::Mat image_self, image_o;
    cv::drawKeypoints(image, keypoints, image_self);
    cv::drawKeypoints(image, keypointsO, image_o);

    cv::imshow("self", image_self);
    cv::imshow("origin", image_o);
    cv::waitKey(0);
    cv::destroyAllWindows();

    int n = 0;
    for (std::size_t idx = 0; idx < keypoints.size(); ++idx) {
        const auto &kp = keypoints[idx];
        for (std::size_t jdx = 0; jdx < keypointsO.size(); ++jdx) {
            const auto &kpO = keypointsO[jdx];
            float dx = (kp.pt.x - kpO.pt.x);
            float dy = (kp.pt.y - kpO.pt.y);
            if (dx * dx + dy * dy == 0 && kp.octave == kpO.octave) {
                // std::cout << "self:   " << descs[idx] << kp.octave << std::endl;
                // std::cout << "origin: " << descsO.row(jdx) << kpO.octave << std::endl;
                // std::cout << kp.pt.x << "\t" << kp.pt.y << std::endl;
                int dis = ORBMatcher::descDistance(descs[idx], descsO.row(jdx));
                // std::cout << "self:   " << kp.angle << std::endl;
                // std::cout << "origin: " << kpO.angle << std::endl;
                std::cout << dis << std::endl;
                ++n;
            }
        }
    }

    return 0;
}