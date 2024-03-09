#pragma once
#include <memory>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "Error.h"
#include "MapPoint.h"
#include "ORBExtractor.h"
#include "ORBMatcher.h"

namespace ORB_SLAM2_ROS2 {
class Frame {
    friend class ORBMatcher;

public:
    typedef std::shared_ptr<Frame> SharedPtr;

    // 普通帧的工厂模式，用于普通帧的创建
    static Frame::SharedPtr create(cv::Mat leftImg, cv::Mat rightImg, int nFeatures, const std::string &briefFp,
                                   int maxThresh, int minThresh) {
        Frame* f = new Frame(leftImg, rightImg, nFeatures, briefFp, maxThresh, minThresh);
        Frame::SharedPtr pFrame(f);
        
        ++mnNextID;
        return pFrame;
    }

    Frame(const Frame &) = delete;
    Frame &operator=(const Frame &) = delete;

    // 获取左右帧图像api
    const cv::Mat &getLeftImage() const { return mLeftIm; }
    const cv::Mat &getRightImage() const { return mRightIm; }

    // 获取图像金字塔api
    const std::vector<cv::Mat> &getLeftPyramid() const { return mpExtractorLeft->getPyramid(); }
    const std::vector<cv::Mat> &getRightPyramid() const { return mpExtractorRight->getPyramid(); }

    // 获取左右帧ORB特征点api
    const std::vector<cv::KeyPoint> &getLeftKeyPoints() const { return mvFeatsLeft; }
    const std::vector<cv::KeyPoint> &getRightKeyPoints() const { return mvFeatsRight; }

    // 获取左右帧ORB描述子api
    const cv::Mat &getLeftDescriptor() const { return mLeftDescriptor; }
    const cv::Mat &getRightDescriptor() const { return mRightDescriptor; }

    // 显示双目匹配结果
    void showStereoMatches() const {
        cv::Mat showImage;
        std::vector<cv::Mat> imgs{mLeftIm, mRightIm};
        cv::hconcat(imgs, showImage);
        cv::cvtColor(showImage, showImage, cv::COLOR_GRAY2BGR);
        std::vector<cv::KeyPoint> rightKps, leftKps;
        for (std::size_t i = 0; i < mvFeatsLeft.size(); ++i) {
            const auto &rightU = mvFeatsRightU[i];
            if (rightU == -1) {
                continue;
            }
            const auto &lkp = mvFeatsLeft[i];
            cv::KeyPoint rkp;
            rkp.pt.x = rightU + mLeftIm.cols;
            rkp.pt.y = lkp.pt.y;
            cv::line(showImage, lkp.pt, rkp.pt, cv::Scalar(255, 0, 0));
            rightKps.push_back(rkp);
            leftKps.push_back(lkp);
        }
        cv::drawKeypoints(showImage, leftKps, showImage, cv::Scalar(0, 255, 0));
        cv::drawKeypoints(showImage, rightKps, showImage, cv::Scalar(0, 0, 255));
        cv::imshow("showImage", showImage);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

private:
    Frame(cv::Mat leftImg, cv::Mat rightImg, int nFeatures, const std::string &briefFp, int maxThresh, int minThresh)
        : mLeftIm(leftImg)
        , mRightIm(rightImg) {
        mpExtractorLeft = std::make_shared<ORBExtractor>(mLeftIm, nFeatures, 8, 1.2, briefFp, maxThresh, minThresh);
        mpExtractorRight = std::make_shared<ORBExtractor>(mRightIm, nFeatures, 8, 1.2, briefFp, maxThresh, minThresh);

        std::thread leftThread(
            std::bind(&ORBExtractor::extract, mpExtractorLeft.get(), std::ref(mvFeatsLeft), std::ref(mLeftDescriptor)));
        std::thread rightThread(std::bind(&ORBExtractor::extract, mpExtractorRight.get(), std::ref(mvFeatsRight),
                                          std::ref(mRightDescriptor)));
        if (!leftThread.joinable() || !rightThread.joinable())
            throw ThreadError("普通帧提取特征点时线程不可joinable");
        leftThread.join();
        rightThread.join();

        // mpExtractorLeft->extract(mvFeatsLeft, mLeftDescriptor);
        // mpExtractorRight->extract(mvFeatsRight, mRightDescriptor);
        ORBMatcher::SharedPtr mpMatcher = std::make_shared<ORBMatcher>();
        mpMatcher->searchByStereo(this);
        mnID = mnNextID;
    }

    static int mnNextID;                          ///< 下一帧ID
    int mnID;                                     ///< 帧ID
    std::vector<cv::KeyPoint> mvFeatsLeft;        ///< 左图特征点坐标
    std::vector<cv::KeyPoint> mvFeatsRight;       ///< 右图特征点坐标
    cv::Mat mLeftDescriptor;                      ///< 左图特征描述子
    cv::Mat mRightDescriptor;                     ///< 右图特征描述子
    std::vector<double> mvDepths;                 ///< 特征点对应的深度值
    std::vector<double> mvFeatsRightU;            ///< 右图特征点匹配的u坐标
    std::vector<MapPoint::SharedPtr> mvMapPoints; ///< 左图对应的地图点
    ORBExtractor::SharedPtr mpExtractorLeft;      ///< 左图特征提取器
    ORBExtractor::SharedPtr mpExtractorRight;     ///< 右图特征提取器
    cv::Mat mLeftIm;                              ///< 左图
    cv::Mat mRightIm;                             ///< 右图
};
} // namespace ORB_SLAM2_ROS2