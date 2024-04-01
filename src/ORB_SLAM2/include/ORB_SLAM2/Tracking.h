#pragma once

#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"

namespace ORB_SLAM2_ROS2 {
enum class TrackingState { NOT_IMAGE_YET, NOT_INITING, OK, LOST };

class KeyFrameDB;
class PnPSolver;

struct RelocBowParam {
    typedef std::shared_ptr<PnPSolver> PnPSolverPtr;

    RelocBowParam(int nCandidate);

    std::vector<PnPSolverPtr> mvpSolvers;
    std::vector<std::vector<cv::DMatch>> mvAllMatches;
    std::vector<std::map<std::size_t, std::size_t>> mvPnPId2MatchID;
};

class Tracking {
public:
    typedef std::vector<MapPoint::SharedPtr> TempMapPoints;
    typedef std::shared_ptr<KeyFrameDB> KFrameDBPtr;

    Tracking(Map::SharedPtr pMap, KFrameDBPtr pKfDB)
        : mnFeatures(2000)
        , mnInitFeatures(3000)
        , mnMaxThresh(20)
        , mnMinThresh(7)
        , mpMap(pMap)
        , mpKfDB(pKfDB) {
        msBriefTemFp = "/home/rookie-lu/Project/ORB-SLAM2/ORB-SLAM2-ROS2/config/BRIEF_TEMPLATE.txt";
    }

    cv::Mat grabFrame(cv::Mat leftImg, cv::Mat rightImg);

    /// 根据当前帧进行初始地图的初始化
    void initForStereo();

    /// 跟踪参考关键帧
    bool trackReference();

    /// 跟踪恒速运动模型
    bool trackMotionModel();

    /// 重定位跟踪
    bool trackReLocalize();

    /// 跟踪局部地图
    bool trackLocalMap();

    /// 处理上一帧
    void processLastFrame();

    /// 构建局部地图
    void buildLocalMap();

    /// 插入局部关键帧
    void insertLocalKFrame(KeyFrame::SharedPtr pKf);

    /// 插入局部地图点
    void insertLocalMPoint(MapPoint::SharedPtr pMp);

    /// 构建局部关键帧
    void buildLocalKfs();

    /// 构建局部地图点
    void buildLocalMps();

    /// 更新速度信息mVelocity
    void updateVelocity();

    /// 更新mTlr（上一帧到参考关键帧的位姿）
    void updateTlr();

    /// 将当前帧升级为关键帧
    KeyFrame::SharedPtr updateCurrFrame();

private:
    /// 使用关键帧数据库，寻找初步关键帧
    bool findInitialKF(std::vector<KeyFrame::SharedPtr> &vpCandidateKFs, int &candidateNum);

    /// 使用词袋匹配进一步筛选候选关键帧
    int filterKFByBow(RelocBowParam &relocBowParam, std::vector<bool> &vbDiscard, const int &candidateNum,
                      std::vector<KeyFrame::SharedPtr> &vpCandidateKFs);

    /// 设置当前关键帧的地图点为指定候选关键帧的匹配
    void setCurrFrameAttrib(const std::vector<std::size_t> &vInliers, const RelocBowParam &relocBowParam,
                            const std::size_t &idx, std::vector<KeyFrame::SharedPtr> &vpCandidateKFs,
                            const cv::Mat &Rcw, const cv::Mat &tcw);

    /// 使用重投影匹配进行精确匹配
    bool addMatchByProject(KeyFrame::SharedPtr pKFrame, int &nInliers);

    TrackingState mStatus = TrackingState::NOT_IMAGE_YET; ///< 跟踪状态

    Frame::SharedPtr mpCurrFrame; ///< 当前帧
    Frame::SharedPtr mpLastFrame; ///< 上一帧
    Map::SharedPtr mpMap;         ///< 地图
    KeyFrame::SharedPtr mpRefKf;  ///< 参考关键帧

    unsigned mnFeatures;                          ///< 正常跟踪时关键点数目
    unsigned mnInitFeatures;                      ///< 初始化地图时关键点数目
    std::string msBriefTemFp;                     ///< BRIEF描述子模版路径
    int mnMaxThresh;                              ///< FAST最大阈值
    int mnMinThresh;                              ///< FAST最小阈值
    cv::Mat mVelocity;                            ///< 速度Tcl
    bool mbUseMotionModel;                        ///< 是否使用运动模型
    cv::Mat mTlr;                                 ///< 上一帧到上一帧参考关键帧的位姿差
    TempMapPoints mvpTempMappoints;               ///< 临时地图点
    std::vector<KeyFrame::SharedPtr> mvpLocalKfs; ///< 局部地图关键帧
    std::vector<MapPoint::SharedPtr> mvpLocalMps; ///< 局部地图点
    KFrameDBPtr mpKfDB;                           ///< 关键帧数据库
};
} // namespace ORB_SLAM2_ROS2