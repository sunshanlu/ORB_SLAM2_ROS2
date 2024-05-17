#pragma once

#include <functional>
#include <memory>
#include <random>
#include <vector>

namespace ORB_SLAM2_ROS2 {

template <typename T> class Ransac {
public:
    typedef std::shared_ptr<Ransac> SharedPtr;

    virtual ~Ransac() = default;

    /// 模型函数，需要子类的重写
    virtual void modelFunc(const std::vector<std::size_t> &vIndices, T &modelRet) {}

    Ransac(const Ransac &) = delete;
    Ransac &operator=(const Ransac &) = delete;

    Ransac() = default;

    /**
     * @brief Ransac算法的构造函数
     * @details
     *      1. 首先固定采样次数
     *      2. 固定（假定）样本中的最少内点数
     *      3. 固定样本中内点比例
     *      4. 根据内点比例和输入的迭代次数进行RANSAC迭代次数计算
     * @param nMinSet           输入的样本采样个数
     * @param nMaxIterations    输入的最大迭代次数
     * @param fRatio            输入的期望内点比例
     * @param fProb             输入的至少一次采样全是内点的概率
     */
    void setRansacParams(int nMinSet = 4, int nMaxIterations = 100, float fRatio = 0.4, float fProb = 0.99) {
        mnMinSet = nMinSet;
        mnMinInlier = std::max((float)nMinSet, mnN * fRatio);
        mfMinInlierRatio = (float)mnMinInlier / mnN;
        if (mfMinInlierRatio >= 1) {
            mnMaxIterations = 0;
            return;
        }
        int nIters = cvRound(std::log(1 - fProb) / std::log(1 - std::pow(mfMinInlierRatio, mnMinSet)));
        mnMaxIterations = std::min(nMaxIterations, nIters);
    }

    /**
     * @brief 进行RANSAC迭代（指定迭代次数）
     *
     * @param nIterations     指定的迭代次数
     * @param modelRet        输出的RANSAC算法的结果
     * @param bNoMore         输出的是否没有剩余迭代次数了
     * @param vnInlierIndices 内点分布索引
     * @return true     迭代成功，输出了位姿信息
     * @return false    迭代失败，没有输出位姿信息
     */
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
                        mvnInlierIndices = vnInlierIndices;
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
            vnInlierIndices = mvnInlierIndices;
            return true;
        }
    }

protected:
    /**
     * @brief 随机采样s次（mnMinSet）
     *
     * @return std::vector<std::size_t> 返回采样的索引
     */
    std::vector<std::size_t> randomSample() {
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

    /// 判断内外点分布
    virtual int checkInliers(std::vector<std::size_t> &vnInlierIndices, const T &modelRet) { return 0; }

    /**
     * @brief 根据内点分布的位置，进行内点回代
     *
     * @param vnInlierIndices   输入输出的内点索引
     * @param modelRet          输出的回代得到的RANSAC结果
     * @return int 输出的回代后的内点数目
     */
    int refine(std::vector<std::size_t> &vnInlierIndices, T &modelRet) {
        modelFunc(vnInlierIndices, modelRet);
        vnInlierIndices.clear();
        return checkInliers(vnInlierIndices, modelRet);
    }

    int mnN;                                   ///< 样本点的数目
    std::vector<std::size_t> mvAllIndices;     ///< 样本的所有的索引
    int mnMaxIterations;                       ///< RANSAC最大迭代次数
    float mfMinInlierRatio;                    ///< RANSAC内点比例
    int mnMinSet;                              ///< 每一次RANSAC取的样本点数
    int mnMinInlier;                           ///< RANSAC内点数（用于判断是否是合格迭代）
    int mnCurrentIteration = 0;                ///< 当前RANSAC的迭代次数
    T mBestModelRet;                           ///< 迭代中最优的位姿（全部refine失败后返回）
    int mnBestInliers = 0;                     ///< 迭代过程中最多内点数目
    std::vector<std::size_t> mvnInlierIndices; ///< 迭代过程中最优结果的内点数目
};

} // namespace ORB_SLAM2_ROS2