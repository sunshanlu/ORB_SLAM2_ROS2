#pragma once
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <DBoW3/DBoW3.h>
#include <rclcpp/rclcpp.hpp>

#include "ORBExtractor.h"
#include "ORBMatcher.h"

namespace ORB_SLAM2_ROS2 {

class Optimizer;
class KeyFrame;
class MapPoint;

/// 帧的抽象类（以析构函数作为纯虚函数）
class VirtualFrame {
    friend class ORBMatcher;

public:
    typedef std::shared_ptr<DBoW3::Vocabulary> VocabPtr;
    typedef std::vector<std::vector<std::vector<std::size_t>>> GridsType;
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;
    typedef std::shared_ptr<MapPoint> MapPointPtr;
    typedef std::shared_ptr<VirtualFrame> SharedPtr;

    VirtualFrame(unsigned width, unsigned height)
        : mnMaxU(width)
        , mnMaxV(height) {
        mTcw = cv::Mat::eye(4, 4, CV_32F);
        mTwc = cv::Mat::eye(4, 4, CV_32F);
    }

    VirtualFrame(const VirtualFrame &other)
        : mvFeatsLeft(other.mvFeatsLeft)
        , mvFeatsRight(other.mvFeatsRight)
        , mvDepths(other.mvDepths)
        , mvFeatsRightU(other.mvFeatsRightU)
        , mvpMapPoints(other.mvpMapPoints)
        , mbBowComputed(other.mbBowComputed)
        , mBowVec(other.mBowVec)
        , mFeatVec(other.mFeatVec)
        , mGrids(other.mGrids)
        , mpRefKF(other.mpRefKF)
        , mnMaxU(other.mnMaxU)
        , mnMaxV(other.mnMaxV) {

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
        other.mTwc.copyTo(mTwc);
    }

    /// 获取某一个特征点的地图点
    cv::Mat unProject(std::size_t idx);

    /// 获取左右帧ORB描述子api
    const std::vector<cv::Mat> &getLeftDescriptor() const { return mvLeftDescriptor; }
    const std::vector<cv::Mat> &getRightDescriptor() const { return mRightDescriptor; }

    /// 获取与当前帧相连的关键帧，且共视权重大于等于th
    virtual std::vector<KeyFramePtr> getConnectedKfs(int th);

    /// 将帧中的匹配地图点置为空
    void setMapPointsNull() {
        for (auto &pMp : mvpMapPoints) {
            pMp = nullptr;
        }
    }

    /// 设置指定位置的地图点
    void setMapPoint(int idx, MapPointPtr pMp);

    /// 获取this的地图点信息
    virtual std::vector<MapPointPtr> getMapPoints();

    virtual MapPointPtr getMapPoint(std::size_t idx) {
        std::unique_lock<std::mutex> lock(mMutexMapPoints);
        return mvpMapPoints[idx];
    }

    /// 将世界坐标系下地图点投影到this的像素坐标系中
    cv::Point2f project2UV(const cv::Mat &p3dW, bool &isPositive);

    /// 在给定的区域里面快速符合要求的特征点
    std::vector<std::size_t> findFeaturesInArea(const cv::KeyPoint &kp, float radius, int minNLevel, int maxNLevel);

    const std::vector<double> &getDepth() const { return mvDepths; }

    const std::vector<double> &getRightU() const { return mvFeatsRightU; }
    const double &getRightU(const std::size_t &idx) const { return mvFeatsRightU[idx]; }

    /// 计算相似程度
    double computeSimilarity(const VirtualFrame &other) { return mpVoc->score(mBowVec, other.mBowVec); }

    /**
     * @brief 设置位姿
     *
     * @param pose 输入的位姿信息
     */
    void setPose(cv::Mat pose) {
        std::unique_lock<std::mutex> lock(mPoseMutex);
        pose.copyTo(mTcw);
        pose.rowRange(0, 3).colRange(0, 3).copyTo(mRcw);
        pose.rowRange(0, 3).col(3).copyTo(mtcw);
        mRwc = mRcw.t();
        mtwc = -mRwc * mtcw;
        mRwc.copyTo(mTwc(cv::Range(0, 3), cv::Range(0, 3)));
        mtwc.copyTo(mTwc(cv::Range(0, 3), cv::Range(3, 4)));
    }

    /**
     * @brief 获取位姿
     *
     */
    cv::Mat getPose() const {
        std::unique_lock<std::mutex> lock(mPoseMutex);
        return mTcw.clone();
    }

    /**
     * @brief 获取位姿
     *
     * @param Rcw 输出的旋转矩阵
     * @param tcw 输出的平移向量
     */
    void getPose(cv::Mat &Rcw, cv::Mat &tcw) const {
        std::unique_lock<std::mutex> lock(mPoseMutex);
        mRcw.copyTo(Rcw);
        mtcw.copyTo(tcw);
    }

    /**
     * @brief 获取位姿
     *
     * @param Rwc
     * @param twc
     */
    void getPoseInv(cv::Mat &Rwc, cv::Mat &twc) {
        std::unique_lock<std::mutex> lock(mPoseMutex);
        mRwc.copyTo(Rwc);
        mtwc.copyTo(twc);
    }

    /// 获取帧中图像金字塔的缩放层级
    static float getScaledFactor(const int &nLevel) { return mvfScaledFactors[nLevel]; }

    /// 获取金字塔缩放因子的平方
    static float getScaledFactor2(const int &nLevel) { return std::pow(getScaledFactor(nLevel), 2); }

    /// 获取金字塔缩放因子的倒数
    static float getScaledFactorInv(const int &nLevel) { return 1.0f / getScaledFactor(nLevel); }

    /// 获取金字塔缩放因子倒数的平方
    static float getScaledFactorInv2(const int &nLevel) { return std::pow(getScaledFactorInv(nLevel), 2); }

    /// 获取左右帧ORB特征点api
    const std::vector<cv::KeyPoint> &getLeftKeyPoints() const { return mvFeatsLeft; }
    const std::vector<cv::KeyPoint> &getRightKeyPoints() const { return mvFeatsRight; }

    /// 获取左右帧ORB特征点api
    const cv::KeyPoint &getLeftKeyPoint(const std::size_t &idx) const { return mvFeatsLeft[idx]; }
    const cv::KeyPoint &getRightKeyPoint(const std::size_t &idx) const { return mvFeatsRight[idx]; }

    /// 计算词袋
    void computeBow() {
        std::unique_lock<std::mutex> lock(mBowMutex);
        if (mbBowComputed)
            return;
        mpVoc->transform(mvLeftDescriptor, mBowVec, mFeatVec, 4);
        mbBowComputed = true;
    }

    /// 获取词袋信息
    const DBoW3::BowVector &getBowVec() const {
        std::unique_lock<std::mutex> lock(mBowMutex);
        return mBowVec;
    }

    /// 获取描述子
    std::vector<cv::Mat> &getDescriptor() { return mvLeftDescriptor; }
    const std::vector<cv::Mat> &getDescriptor() const { return mvLeftDescriptor; }

    /// 获取帧光心位置（世界坐标系下）
    cv::Mat getFrameCenter() const {
        std::unique_lock<std::mutex> lock(mPoseMutex);
        return mtwc.clone();
    }

    /// 获取Twc
    cv::Mat getPoseInv() {
        std::unique_lock<std::mutex> lock(mPoseMutex);
        return mTwc.clone();
    }

    virtual ~VirtualFrame() = default;

protected:
    std::vector<cv::KeyPoint> mvFeatsLeft;      ///< 左图特征点坐标
    std::vector<cv::KeyPoint> mvFeatsRight;     ///< 右图特征点坐标
    std::vector<cv::Mat> mvLeftDescriptor;      ///< 左图特征描述子
    std::vector<cv::Mat> mRightDescriptor;      ///< 右图特征描述子
    std::vector<double> mvDepths;               ///< 特征点对应的深度值
    std::vector<double> mvFeatsRightU;          ///< 右图特征点匹配的u坐标
    std::vector<MapPointPtr> mvpMapPoints;      ///< 左图对应的地图点
    mutable std::mutex mMutexMapPoints;         ///< 地图点对应的互斥锁
    mutable std::mutex mBowMutex;               ///< 对应词袋向量的互斥锁
    bool mbBowComputed = false;                 ///< 是否计算了BOW向量
    DBoW3::BowVector mBowVec;                   ///< 左图的BOW向量
    DBoW3::FeatureVector mFeatVec;              ///< 左图的特征向量
    static std::string msVoc;                   ///< 字典词袋路径
    mutable std::mutex mPoseMutex;              ///< 位姿互斥锁
    cv::Mat mTcw, mTwc;                         ///< 帧位姿
    cv::Mat mRcw, mRwc;                         ///< 位姿的旋转矩阵
    cv::Mat mtcw, mtwc;                         ///< 位姿的平移向量
    static std::vector<float> mvfScaledFactors; ///< 特征点缩放因子
    static bool mbScaled;                       ///< 金字塔缩放因子是否初始化
    static unsigned mnGridHeight;               ///< 网格高度
    static unsigned mnGridWidth;                ///< 网格宽度
    GridsType mGrids;                           ///< 网格（第一层为行，第二层为列）
    KeyFramePtr mpRefKF;                        ///< 普通帧的参考关键帧

public:
    static VocabPtr mpVoc; ///< 字典词袋
    unsigned mnMaxU;       ///< 图像宽度
    unsigned mnMaxV;       ///< 图像高度
};

/// 普通帧
class Frame : public VirtualFrame {
    friend class ORBMatcher;
    friend class Optimizer;

public:
    typedef std::shared_ptr<Frame> SharedPtr;
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;

    /// 普通帧的工厂模式，用于普通帧的创建
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

    std::size_t getID() { return mnID; }

    /// 获取左右帧图像api
    const cv::Mat &getLeftImage() const { return mLeftIm; }
    const cv::Mat &getRightImage() const { return mRightIm; }

    /// 获取图像金字塔api
    const std::vector<cv::Mat> &getLeftPyramid() const { return mpExtractorLeft->getPyramid(); }
    const std::vector<cv::Mat> &getRightPyramid() const { return mpExtractorRight->getPyramid(); }

    // 显示双目匹配结果
    void showStereoMatches() const;

    /// 利用双目进行三角化得到地图点
    void unProject(std::vector<MapPointPtr> &mapPoints);

    /// 获取帧中有深度特征点的数目
    int getN() { return mnN; }

    /// 获取帧中的参考关键帧
    KeyFramePtr getRefKF();

private:
    // 帧的构造函数
    Frame(cv::Mat leftImg, cv::Mat rightImg, int nFeatures, const std::string &briefFp, int maxThresh, int minThresh);

    static std::size_t mnNextID;              ///< 下一帧ID
    std::size_t mnID;                         ///< 帧ID
    cv::Mat mLeftIm, mRightIm;                ///< 左右图
    int mnN;                                  ///< 帧中有深度的地图点
    ORBExtractor::SharedPtr mpExtractorLeft;  ///< 左图特征提取器
    ORBExtractor::SharedPtr mpExtractorRight; ///< 右图特征提取器
};
} // namespace ORB_SLAM2_ROS2