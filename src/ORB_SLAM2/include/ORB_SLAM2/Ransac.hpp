#pragma once

#include <functional>
#include <memory>
#include <vector>

namespace ORB_SLAM2_ROS2 {

template<typename T> class Ransac {
public:
    typedef std::shared_ptr<Ransac> SharedPtr;

    virtual ~Ransac() = default;

    /// 模型函数，需要子类的重写
    virtual void modelFunc(const std::vector<std::size_t> &vIndices, T &modelRet) {}

    Ransac(const Ransac &) = delete;
    Ransac &operator=(const Ransac &) = delete;

    /// 设置RANSAC算法的参数
    void setRansacParams(int nMaxIterations = 300, float fRatio = 0.4, int nMinSet = 4, float fProb = 0.99) {
        static std::default_random_engine generator;
        std::uniform_int_distribution<std::size_t> distribution(0, mnN - 1);
        std::vector<size_t> vChoseIndices;
        int nNum = 0;
        while (nNum != mnMinSet) {
            std::size_t rdVal = distribution(generator);
            if (std::find(vChoseIndices.begin(), vChoseIndices.end(), rdVal) == vChoseIndices.end()) {
                vChoseIndices.push_back(rdVal);
                nNum++;
            }
        }
        return vChoseIndices;
    }

    /// 进行RANSAC迭代（指定迭代次数）
    bool iterate(int nIterations, T &modelRet, bool &bNoMore, std::vector<std::size_t> &vnInlierIndices) {
        int iter = 0;
        if (mnN < mnMinSet) {
            bNoMore = true;
            return false;
        }
        while (iter < nIterations && mnCurrentIteration < mnMaxIterations) {
            auto vnChoseIndices = randomSample();
            modelFunc(vnChoseIndices, modelRet);
            if (!modelRet.error()) {
                int nInliers = checkInliers(vnInlierIndices, modelRet);
                if (nInliers > mnMinInlier) {
                    if (nInliers > mnBestInliers) {
                        mnBestInliers = nInliers;
                        modelRet.copyTo(mBestModelRet);
                    }
                    if (refine(vnInlierIndices, modelRet) > mnMinInlier)
                        return true;
                }
            }
            ++iter;
            ++mnCurrentIteration;
        }
        if (mnCurrentIteration >= mnMaxIterations)
            bNoMore = true;
        if (mnBestInliers == 0)
            return false;
        else {
            modelRet.copyTo(mBestModelRet);
            return true;
        }
    }

private:
    /// 随机无放回采样
    std::vector<std::size_t> randomSample();

    /// 判断内外点分布
    int checkInliers(std::vector<std::size_t> &vnInlierIndices, const T &modelRet) { return 0; }

    /// 根据内外点分布进行回代
    virtual int refine(std::vector<std::size_t> &vnInlierIndices, T &modelRet) { return 0; }

    int mnN;                               ///< 样本点的数目
    std::vector<std::size_t> mvAllIndices; ///< 样本的所有的索引
    std::vector<float> mvfErrors;          ///< 判断内外点阈值t
    int mnMaxIterations;                   ///< RANSAC最大迭代次数
    float mfMinInlierRatio;                ///< RANSAC内点比例
    int mnMinSet;                          ///< 每一次RANSAC取的样本点数
    int mnMinInlier;                       ///< RANSAC内点数（用于判断是否是合格迭代）
    int mnCurrentIteration = 0;            ///< 当前RANSAC的迭代次数
    T mBestModelRet;                       ///< 迭代中最优的位姿（全部refine失败后返回）
    int mnBestInliers = 0;                 ///< 迭代过程中最多内点数目
};

} // namespace ORB_SLAM2_ROS2