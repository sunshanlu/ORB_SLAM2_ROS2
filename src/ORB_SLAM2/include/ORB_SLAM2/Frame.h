#pragma once
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <DBoW3/DBoW3.h>
#include <rclcpp/rclcpp.hpp>

#include "MapPoint.h"
#include "ORBExtractor.h"
#include "ORBMatcher.h"

namespace ORB_SLAM2_ROS2 {

/// 帧的抽象类（以析构函数作为纯虚函数）
class VirtualFrame {
public:
    typedef std::shared_ptr<DBoW3::Vocabulary> VocabPtr;

    VirtualFrame() = default;

    VirtualFrame(const VirtualFrame &other)
        : mvFeatsLeft(other.mvFeatsLeft)
        , mvFeatsRight(other.mvFeatsRight)
        , mvDepths(other.mvDepths)
        , mvFeatsRightU(other.mvFeatsRightU)
        , mvpMapPoints(other.mvpMapPoints)
        , mbBowComputed(other.mbBowComputed)
        , mBowVec(other.mBowVec)
        , mFeatVec(other.mFeatVec) {
    
        // 对cv::Mat类型的数据进行深拷贝
        std::vector<cv::Mat> leftDesc(other.mvLeftDescriptor.size()), rightDesc(other.mRightDescriptor.size());
        for (std::size_t idx = 0; idx < other.mvLeftDescriptor.size(); ++idx) {
            other.mvLeftDescriptor[idx].copyTo(leftDesc[idx]);
        }
        for (std::size_t idx = 0; idx < other.mRightDescriptor.size(); ++idx) {
            other.mRightDescriptor[idx].copyTo(rightDesc[idx]);
        }
        std::swap(leftDesc, mvLeftDescriptor);
        std::swap(rightDesc, mRightDescriptor);
        other.mtcw.copyTo(mtcw);
        other.mtwc.copyTo(mtwc);
        other.mRcw.copyTo(mRcw);
        other.mRwc.copyTo(mRwc);
        other.mTcw.copyTo(mTcw);
    }

    /**
     * @brief 设置位姿
     *
     * @param pose 输入的位姿信息
     */
    void setPose(cv::Mat pose) {
        std::unique_lock<std::mutex>(mPoseMutex);
        pose.copyTo(mTcw);
        pose.rowRange(0, 3).colRange(0, 3).copyTo(mRcw);
        pose.rowRange(0, 3).col(3).copyTo(mtcw);
        mRwc = mRcw.t();
        mtwc = -mRwc * mtcw;
    }

    void computeBow() {
        mpVoc->transform(mvLeftDescriptor, mBowVec, mFeatVec, 3);
        mbBowComputed = true;
    }

    bool isBowComputed() const { return mbBowComputed; }

    virtual ~VirtualFrame() = default;

protected:
    std::vector<cv::KeyPoint> mvFeatsLeft;         ///< 左图特征点坐标
    std::vector<cv::KeyPoint> mvFeatsRight;        ///< 右图特征点坐标
    std::vector<cv::Mat> mvLeftDescriptor;         ///< 左图特征描述子
    std::vector<cv::Mat> mRightDescriptor;         ///< 右图特征描述子
    std::vector<double> mvDepths;                  ///< 特征点对应的深度值
    std::vector<double> mvFeatsRightU;             ///< 右图特征点匹配的u坐标
    std::vector<MapPoint::SharedPtr> mvpMapPoints; ///< 左图对应的地图点
    ORBExtractor::SharedPtr mpExtractorLeft;       ///< 左图特征提取器
    ORBExtractor::SharedPtr mpExtractorRight;      ///< 右图特征提取器
    bool mbBowComputed = false;                    ///< 是否计算了BOW向量
    DBoW3::BowVector mBowVec;                      ///< 左图的BOW向量
    DBoW3::FeatureVector mFeatVec;                 ///< 左图的特征向量
    static std::string msVoc;                      ///< 字典词袋路径
    static VocabPtr mpVoc;                         ///< 字典词袋
    std::mutex mPoseMutex;                         ///< 位姿互斥锁
    cv::Mat mTcw;                                  ///< 帧位姿
    cv::Mat mRcw, mRwc;                            ///< 位姿的旋转矩阵
    cv::Mat mtcw, mtwc;                            ///< 位姿的平移向量
};

class Frame : public VirtualFrame {
    friend class ORBMatcher;

public:
    typedef std::shared_ptr<Frame> SharedPtr;

    // 普通帧的工厂模式，用于普通帧的创建
    static Frame::SharedPtr create(cv::Mat leftImg, cv::Mat rightImg, int nFeatures, const std::string &briefFp,
                                   int maxThresh, int minThresh) {

        Frame *f = new Frame(leftImg, rightImg, nFeatures, briefFp, maxThresh, minThresh);
        Frame::SharedPtr pFrame(f);
        ORBMatcher::SharedPtr mpMatcher = std::make_shared<ORBMatcher>();
        pFrame->mnN = mpMatcher->searchByStereo(pFrame);
        ++mnNextID;
        return pFrame;
    }

    Frame(const Frame &) = delete;
    Frame &operator=(const Frame &) = delete;
    ~Frame() = default;

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
    const std::vector<cv::Mat> &getLeftDescriptor() const { return mvLeftDescriptor; }
    const std::vector<cv::Mat> &getRightDescriptor() const { return mRightDescriptor; }

    // 显示双目匹配结果
    void showStereoMatches() const;

    /// 利用双目进行三角化得到地图点
    void unProject(std::vector<MapPoint::SharedPtr> &mapPoints);

    /// 获取某一个特征点的地图点
    cv::Mat unProject(std::size_t idx);

    int getN() { return mnN; }

private:
    // 帧的构造函数
    Frame(cv::Mat leftImg, cv::Mat rightImg, int nFeatures, const std::string &briefFp, int maxThresh, int minThresh);

    static std::size_t mnNextID; ///< 下一帧ID
    std::size_t mnID;            ///< 帧ID
    cv::Mat mLeftIm, mRightIm;   ///< 左右图
    int mnN;                     ///< 帧中有深度的地图点
};
} // namespace ORB_SLAM2_ROS2