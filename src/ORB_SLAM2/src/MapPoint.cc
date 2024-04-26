#include <cmath>

#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/KeyFrame.h"
#include "ORB_SLAM2/Map.h"
#include "ORB_SLAM2/MapPoint.h"

namespace ORB_SLAM2_ROS2 {

/**
 * @brief 添加地图向关键帧的观测信息
 * @details
 *      1. 当地图点没有针对pkf的观测时，直接添加观测
 *      2. 当地图点存在针对pkf的观测时
 *          1. 拿出两个2d特征点对应的描述子来，与当前关键帧描述子进行距离比较
 *          2. 将描述子距离小的替换距离大的
 * @param pKf       观测的关键帧
 * @param featId    关键帧对应特征点的id
 */
void MapPoint::addObservation(KeyFrame::SharedPtr pKf, std::size_t featId) {
    std::unique_lock<std::mutex> lock(mObsMutex);
    auto iter = mObs.find(pKf);
    if (iter == mObs.end())
        mObs.insert(std::make_pair(pKf, featId));
    else {
        if (!pKf || pKf->isBad()) {
            mObs.erase(iter);
            return;
        } else {
            auto desc1 = pKf->getDescriptor()[iter->second];
            auto desc2 = pKf->getDescriptor()[featId];
            int dis1 = ORBMatcher::descDistance(desc1, mDescriptor);
            int dis2 = ORBMatcher::descDistance(desc2, mDescriptor);
            std::size_t destID = dis1 < dis2 ? iter->second : featId;
            mObs[pKf] = destID;
        }
    }
}

/**
 * @brief 获取观测信息中可靠的数目
 *
 * @return int
 */
int MapPoint::getObsNum() {
    std::unique_lock<std::mutex> lock(mObsMutex);
    int cnt = 0;
    for (const auto &item : mObs) {
        auto pkf = item.first.lock();
        if (pkf && !pkf->isBad())
            ++cnt;
    }
    return cnt;
}

/**
 * @brief 设置当前地图点的最大距离和最小距离
 *
 * @param pRefKf    参考关键帧
 * @param nFeatId   特征点id（二维关键帧）
 */
void MapPoint::setDistance(KeyFramePtr pRefKf, std::size_t nFeatId) {
    auto kp = pRefKf->getLeftKeyPoints()[nFeatId];
    cv::Mat center2Mp;
    {
        std::unique_lock<std::mutex> lock(mPosMutex);
        center2Mp = mPoint3d - pRefKf->getFrameCenter();
    }
    center2Mp.copyTo(mViewDirection);
    float x = center2Mp.at<float>(0, 0);
    float y = center2Mp.at<float>(1, 0);
    float z = center2Mp.at<float>(2, 0);
    float dist = std::sqrt(x * x + y * y + z * z);
    mnMaxDistance = dist * std::pow(ORBExtractor::mfScaledFactor, kp.octave - 0);
    mnMinDistance = dist * std::pow(ORBExtractor::mfScaledFactor, kp.octave - ORBExtractor::mnLevels + 1);
}

/**
 * @brief 跟踪线程初始化地图专用函数，用于添加地图点的属性信息
 *
 * @param pRefKf    参考关键帧
 * @param nFeatId   特征点id（二维关键帧）
 */
void MapPoint::addAttriInit(KeyFramePtr pRefKf, std::size_t nFeatId) {
    mpRefKf = pRefKf;
    mnRefFaatID = nFeatId;
    if (!pRefKf || pRefKf->isBad()) {
        mbIsBad = true;
        return;
    }
    addObservation(pRefKf, nFeatId);
    pRefKf->getDescriptor()[nFeatId].copyTo(mDescriptor);
    setDistance(pRefKf, nFeatId);
}

MapPoint::MapPoint(cv::Mat mP3d, KeyFrame::SharedPtr pRefKf, std::size_t nObs)
    : mPoint3d(std::move(mP3d))
    , mObs(KeyFrame::weakCompare) {
    mId = mnNextId;
    addAttriInit(pRefKf, nObs);
}

MapPoint::MapPoint(cv::Mat mp3d)
    : mPoint3d(mp3d)
    , mObs(KeyFrame::weakCompare) {
    mId = mnNextId;
}

/**
 * @brief 判断某一位姿是否能观测到该地图点
 * @details
 *      1. 转换到相机坐标系下，z值大于0
 *      2. 投影到像素坐标系下，x和y值在图像上
 *      3. 长度在最大值和最小值之间
 *      4. 地图点平均观测角度和当前观测角度要小于60°
 *      5. 考虑到关键帧有可能失效，这里不做考虑，传入指针前应保证关键帧是有效的
 * @param pFrame        输入的帧
 * @param vecDistance   输出的计算pFrame光心到地图点的距离
 * @param uv            输出的投影产生的像素坐标系下的地图点
 * @return true     在可视范围内
 * @return false    不在可视范围内
 */
bool MapPoint::isInVision(VirtualFramePtr pFrame, float &vecDistance, cv::Point2f &uv, float &cosTheta) {
    cv::Mat Rcw, tcw;
    pFrame->getPose(Rcw, tcw);
    cv::Mat p3dC;
    {
        std::unique_lock<std::mutex> lock(mPosMutex);
        p3dC = Rcw * mPoint3d + tcw;
    }
    if (p3dC.at<float>(2, 0) < 0)
        return false;
    float x = p3dC.at<float>(0, 0);
    float y = p3dC.at<float>(1, 0);
    float z = p3dC.at<float>(2, 0);
    float distance = std::sqrt(x * x + y * y + z * z);
    vecDistance = distance;
    if (distance > mnMaxDistance || distance < mnMinDistance)
        return false;
    float u = x / z * Camera::mfFx + Camera::mfCx;
    float v = y / z * Camera::mfFy + Camera::mfCy;
    uv.x = u;
    uv.y = v;
    if (u > pFrame->mnMaxU || u < 0 || v > pFrame->mnMaxV || v < 0)
        return false;
    cv::Mat viewDirection = Rcw * mViewDirection;
    float viewDirectionAbs = cv::norm(viewDirection, cv::NORM_L2);
    cosTheta = viewDirection.dot(p3dC) / (distance * viewDirectionAbs);
    if (cosTheta < 0.5)
        return false;
    return true;
}

/**
 * @brief 判断地图点是否是bad（即将要删除的点）
 *
 * @return true     判断外点，即将要删除
 * @return false    判断内点，不会删除
 */
bool MapPoint::isBad() const {
    std::unique_lock<std::mutex> lock(mBadMutex);
    return mbIsBad;
}

/**
 * @brief 金字塔层级预测
 * 在帧位姿相对稳定的时候使用，不稳定的时候，无意义，因为距离不准确
 * @param distance 输入的距离（待匹配光心到地图点的距离）
 * @return int 金字塔层级
 */
int MapPoint::predictLevel(float distance) const {
    int level = cvRound(std::log(mnMaxDistance / distance) / std::log(ORBExtractor::mfScaledFactor));
    if (level < 0)
        level = 0;
    else if (level > 7)
        level = 7;
    return level;
}

/**
 * @brief pMp1替换pMp2
 * @details
 *      1. 更新观测数据mObs
 *      2. 更新描述子和观测方向
 *      3. 更新mnMatchesInTrack和mnInliersInTrack
 * @param pMp1 输入的替换地图点
 * @param pMp1 输入的被替换地图点
 * @param pMap 输入输出的地图（处理地图点）
 */
void MapPoint::replace(SharedPtr pMp1, SharedPtr pMp2, MapPtr pMap) {
    pMp2->setBad();
    auto obs = pMp2->clearObversition();
    for (auto &item : obs) {
        auto pkf = item.first.lock();
        if (!pkf || pkf->isBad())
            continue;
        if (pMp1->isInKeyFrame(pkf)) {
            continue;
        }
        pkf->setMapPoint(item.second, pMp1);
        pMp1->addObservation(pkf, item.second);
    }
    pMp1->updateDescriptor();
    pMp1->updateNormalAndDepth();
    pMp1->updateTrackParam(pMp2);
    pMap->eraseMapPoint(pMp2);
}

/**
 * @brief 判断this地图点是否在关键帧pkf中
 *
 * @param pkf   输入的关键帧pkf
 * @return true     在关键帧pkf中
 * @return false    不在关键帧pkf中
 */
bool MapPoint::isInKeyFrame(KeyFramePtr pkf) {
    bool isIn = false;
    std::unique_lock<std::mutex> lock(mObsMutex);
    for (auto &item : mObs) {
        auto pkfi = item.first.lock();
        if (pkfi == pkf) {
            isIn = true;
            break;
        }
    }
    return isIn;
}

/**
 * @brief 由关键帧构造的地图点
 * @details
 *      1. 确定参考关键帧
 *      2. 最大和最小深度
 *      3. 被观测方向
 *      4. 地图点的代表描述子
 * @param mP3dW 世界坐标系下的地图点位置
 * @param pKf   参考关键帧（基于哪个关键帧构建的）
 * @param nObs  该地图点对应的2D关键点的索引
 * @return MapPoint::SharedPtr 返回的地图点
 */
MapPoint::SharedPtr MapPoint::create(cv::Mat mP3dW, KeyFrame::SharedPtr pKf, std::size_t nObs) {
    MapPoint::SharedPtr pMpoint(new MapPoint(mP3dW, pKf, nObs));
    ++mnNextId;
    return pMpoint;
}

/**
 * @brief 由普通帧构造的地图点
 * @details
 *      没有复杂的和关键帧之间的属性信息，只有位置信息
 * @param mP3dW 世界坐标系下的坐标
 * @return MapPoint::SharedPtr 返回构造成功的地图点共享指针
 */
MapPoint::SharedPtr MapPoint::create(cv::Mat mP3dW) {
    MapPoint::SharedPtr pMpoint(new MapPoint(mP3dW));
    ++mnNextId;
    return pMpoint;
}

/**
 * @brief 获取观测中有效的关键帧和索引pair
 *
 */
std::vector<std::pair<KeyFrame::SharedPtr, std::size_t>> MapPoint::getPostiveObs() {
    std::vector<std::pair<KeyFrame::SharedPtr, std::size_t>> vObs;
    auto obsTmp = getObservation();
    for (auto &obs : obsTmp) {
        KeyFrame::SharedPtr pKf = obs.first.lock();
        if (pKf && !pKf->isBad())
            vObs.push_back(std::make_pair(pKf, obs.second));
    }
    return std::move(vObs);
}

/**
 * @brief 计算容器的行向量
 *
 * @param vec       输入的行向量
 * @return float    输出的中位数数值
 */
float MapPoint::computeMedian(const cv::Mat &vec) {
    std::vector<float> vecTmp;
    for (int col = 0; col < vec.cols; ++col)
        vecTmp.push_back(vec.at<float>(col));
    std::sort(vecTmp.begin(), vecTmp.end(), [](const float &a, const float &b) { return a < b; });
    std::size_t nNum = vecTmp.size();
    bool isOdd = !(nNum % 2);
    float idx = (nNum - 1) / 2.0f;
    if (isOdd)
        return vecTmp[(int)idx];
    else
        return (vecTmp[(int)idx] + vecTmp[(int)idx + 1]) / 2.0f;
}

/**
 * @brief 更新地图点代表性描述子
 * @details
 *      1. 获取观测中的距离矩阵
 *      2. 对矩阵的每一行取中位值
 *      3. 将最小中值对应的描述子作为地图点的新代表描述子
 */
void MapPoint::updateDescriptor() {
    auto vObs = getPostiveObs();
    if (vObs.empty()) {
        setBad();
        return;
    }
    int nNum = vObs.size();
    cv::Mat disMat = cv::Mat::zeros(nNum, nNum, CV_32F);
    for (int idx = 0; idx < nNum; ++idx) {
        const auto &iDesc = vObs[idx].first->getDescriptor()[vObs[idx].second];
        for (int jdx = idx + 1; jdx < idx; ++jdx) {
            const auto &jDesc = vObs[jdx].first->getDescriptor()[vObs[jdx].second];
            float distance = (float)ORBMatcher::descDistance(iDesc, jDesc);
            disMat.at<float>(idx, jdx) = distance;
            disMat.at<float>(jdx, idx) = distance;
        }
    }
    float minMedian = 300;
    int bestRow = 0;
    for (int row = 0; row < nNum; ++row) {
        float val = computeMedian(disMat.row(row));
        if (val < minMedian) {
            minMedian = val;
            bestRow = row;
        }
    }
    mDescriptor = vObs[bestRow].first->getDescriptor()[vObs[bestRow].second];
}

/**
 * @brief 判断地图点是否合格
 * @details
 *      1. 三维点应该在相机前方
 *      2. 重投影误差要小于5.991 * sigma^2
 *      3. 应具有尺度连续性(这里有1.5倍的让步)
 * @param pkf1      输入的关键帧1
 * @param pkf2      输入的关键帧2
 * @param nFeatId1  输入的关键帧1对应的特征点id
 * @param nFeatId2  输入的关键帧2对应的特征点id
 * @return true     地图点合格
 * @return false    地图点不合格
 */
bool MapPoint::checkMapPoint(KeyFrame::SharedPtr pkf1, KeyFrame::SharedPtr pkf2, const std::size_t &nFeatId1,
                             const std::size_t &nFeatId2) {
    cv::Mat R1w, R2w, t1w, t2w;
    pkf1->getPose(R1w, t1w);
    pkf2->getPose(R2w, t2w);
    const auto &kp1 = pkf1->getLeftKeyPoints()[nFeatId1];
    const auto &kp2 = pkf2->getLeftKeyPoints()[nFeatId2];
    int nLayer1 = kp1.octave;
    int nLayer2 = kp2.octave;
    float lsacle1 = Frame::getScaledFactor(nLayer1);
    float lsacle2 = Frame::getScaledFactor(nLayer2);
    float l2sacle1 = Frame::getScaledFactor2(nLayer1);
    float l2sacle2 = Frame::getScaledFactor2(nLayer2);
    /// 1. 三维点在相机前方
    cv::Mat p3dC1 = R1w * mPoint3d + t1w;
    cv::Mat p3dC2 = R2w * mPoint3d + t2w;
    if (p3dC1.at<float>(2) <= 0 || p3dC2.at<float>(2) <= 0)
        return false;

    /// 2. 三维重投影误差小于5.991 * sigma^2
    float u1 = p3dC1.at<float>(0) / p3dC1.at<float>(2) * Camera::mfFx + Camera::mfCx;
    float v1 = p3dC1.at<float>(1) / p3dC1.at<float>(2) * Camera::mfFy + Camera::mfCy;
    float u2 = p3dC2.at<float>(0) / p3dC2.at<float>(2) * Camera::mfFx + Camera::mfCx;
    float v2 = p3dC2.at<float>(1) / p3dC2.at<float>(2) * Camera::mfFy + Camera::mfCy;
    float error1 = std::pow(kp1.pt.x - u1, 2) + std::pow(kp1.pt.y - v1, 2);
    float error2 = std::pow(kp2.pt.x - u2, 2) + std::pow(kp1.pt.y - v2, 2);
    if (error1 > 5.991 * l2sacle1 || error2 > 5.991 * l2sacle2)
        return false;

    /// 3. 检查尺度连续性
    float disRatio = cv::norm(p3dC1) / cv::norm(p3dC2);
    float pyRatio = lsacle1 / lsacle2;
    if (disRatio > pyRatio * 1.5 || disRatio < pyRatio / 1.5)
        return false;
    return true;
}

/**
 * @brief 更新地图点的观测方向和深度信息（可能会更新参考关键帧的信息）
 * @details
 *      1. 更新平均观测方向，观测向量相加+向量归一化
 *      2. 判断地图点的参考关键帧是否有效，若无效则更新
 *      3. 根据更新的参考关键帧，进行地图点深度的更新
 */
void MapPoint::updateNormalAndDepth() {
    auto obsTmp = getObservation();
    if (obsTmp.empty()) {
        setBad();
        return;
    }
    int nNum = 0;
    cv::Mat viewDirection = cv::Mat::zeros(3, 1, CV_32F);
    std::vector<cv::Mat> viewDirections;
    std::vector<KeyFrame::SharedPtr> positiveKFs;
    std::vector<std::size_t> featIDs;
    for (auto &obs : obsTmp) {
        auto pKF = obs.first.lock();
        if (pKF && !pKF->isBad()) {
            positiveKFs.push_back(pKF);
            featIDs.push_back(obs.second);
            cv::Mat direction = getPos() - pKF->getFrameCenter();
            viewDirections.push_back(direction);
            viewDirection += direction;
            ++nNum;
        }
    }
    if (!nNum) {
        setBad();
        return;
    }
    cv::normalize(viewDirection, mViewDirection);

    KeyFrame::SharedPtr pRefKF = mpRefKf.lock();
    bool flag = (pRefKF && !pRefKF->isBad());
    if (!flag) {
        float minCosTheta = 1;
        std::size_t bestIdx = 0;
        for (std::size_t idx = 0; idx < nNum; ++idx) {
            const auto &viewDirection = viewDirections[idx];
            float cosTheta = viewDirection.dot(mViewDirection) / cv::norm(viewDirection);
            if (cosTheta < minCosTheta) {
                minCosTheta = cosTheta;
                bestIdx = idx;
            }
        }
        mpRefKf = positiveKFs[bestIdx];
        pRefKF = positiveKFs[bestIdx];
        mnRefFaatID = featIDs[bestIdx];
        setDistance(pRefKF, mnRefFaatID);
    }
}

/**
 * @brief 删除指定的观测信息
 * @details
 *      1. 进行地图点观测信息的删除
 *      2. 如何删除观测信息是地图点的参考关键帧，需要进行参考关键帧的更新
 * @param pkf 输入的待删除的关键帧信息
 */
void MapPoint::eraseObservetion(KeyFramePtr pkf, bool checkRef) {
    {
        std::unique_lock<std::mutex> lock(mObsMutex);
        auto iter = mObs.find(pkf);
        if (iter != mObs.end())
            mObs.erase(pkf);
    }
    if (checkRef) {
        KeyFrame::SharedPtr pRefKF = mpRefKf.lock();
        if (pkf != pRefKF)
            return;
        else {
            updateNormalAndDepth();
        }
    }
}

/// 获取参考关键帧
KeyFrame::SharedPtr MapPoint::getRefKF() { return mpRefKf.lock(); }

unsigned int MapPoint::mnNextId = 0;

} // namespace ORB_SLAM2_ROS2
