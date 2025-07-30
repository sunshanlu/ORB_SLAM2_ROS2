#include <float.h>
#include <random>

#include "ORB_SLAM2/Camera.h"
#include "ORB_SLAM2/Error.h"
#include "ORB_SLAM2/Frame.h"
#include "ORB_SLAM2/PnPSolver.h"

namespace ORB_SLAM2_ROS2
{

/**
 * @brief PnPSolver的构造函数
 *
 * @param vMapPoints 世界坐标系下对应的3D点
 * @param vORBPoints ORB特征点（带金字塔信息）
 */
PnPSolver::PnPSolver(std::vector<cv::Mat> &vMapPoints, std::vector<cv::KeyPoint> &vORBPoints)
{
  int n = vMapPoints.size();
  for (int idx = 0; idx < n; ++idx)
  {
    const cv::Mat &mpPos = vMapPoints[idx];
    const cv::KeyPoint &kp = vORBPoints[idx];
    mvMapPoints.emplace_back(mpPos.at<float>(0), mpPos.at<float>(1), mpPos.at<float>(2));
    mvORBPoints.emplace_back(kp.pt);
    mvfErrors.emplace_back(5.991 * Frame::getScaledFactor2(kp.octave));
    mvAllIndices.push_back(idx);
  }
  mnN = n;
  setRansacParams();
}

/**
 * @brief 使用EPnP初步求解位姿
 * @details
 *      1. 确定控制点坐标（质心 + PCA归一化法）
 *      2. 确定参数矩阵alpha
 *      3. 根据投影约束，获取Mx = 0
 *      4. 根据刚体约束，获取Lβ = rho，根据取参数技巧获取β的初步值
 *      5. 使用高斯牛顿法和刚体约束，对β进行进一步求解
 *      6. 根据四个控制点的坐标，使用ICP方法求解相机位姿
 * @param vIndices
 */
void PnPSolver::modelFunc(const std::vector<std::size_t> &vIndices, PnPRet &modelRet)
{
  if (vIndices.size() < 4)
    throw EPnPError("进行EPnP时，传入的索引数目不合法！");
  std::vector<cv::Point3f> vMapPoints;
  std::vector<cv::Point2f> vORBPoints;
  for (const auto &idx : vIndices)
  {
    vMapPoints.push_back(mvMapPoints[idx]);
    vORBPoints.push_back(mvORBPoints[idx]);
  }
  bool isPositive = true;
  std::vector<cv::Point3f> ctlPointsW = computeCtlPoint(vMapPoints, isPositive);
  if (!isPositive)
    return;
  cv::Mat Alpha = computeAlpha(vMapPoints, ctlPointsW);
  cv::Mat M = computeMMat(vORBPoints, Alpha);
  auto MREVec = computeMREVec(M);
  cv::Mat L = computeLMat(MREVec);
  cv::Mat rho = computeRho(ctlPointsW);
  cv::Mat beta = computeBetaUnOpt(L, rho);
  GLOptimize(L, beta, rho, 5);
  std::vector<cv::Point3f> ctlPointsC(4, cv::Point3f(0, 0, 0));
  for (int idx = 0; idx < 4; ++idx)
  {
    auto eigenVec = MREVec[idx];
    float betaVal = beta.at<float>(idx, 0);
    for (int jdx = 0; jdx < 4; ++jdx)
      ctlPointsC[jdx] += betaVal * eigenVec[jdx];
  }
  ICP(ctlPointsW, ctlPointsC, modelRet.mRcw, modelRet.mtcw);
}

/**
 * @brief 根据四个控制点坐标，使用ICP方法求解相机位姿
 * @details
 *      1. 计算相机坐标系和世界世界坐标系下的去质心坐标矩阵
 *      2. 计算中间矩阵H
 *      3. 对矩阵H进行SVD分解，获取旋转矩阵和位移向量
 * @param ctlPointsW    输入的世界坐标系下的控制点坐标
 * @param ctlPointsC    输入的相机坐标系下的控制点坐标
 * @param Rcw           输出的相机旋转矩阵
 * @param tcw           输出的相机位移向量
 */
void PnPSolver::ICP(const std::vector<cv::Point3f> &ctlPointsW, const std::vector<cv::Point3f> &ctlPointsC, cv::Mat &Rcw, cv::Mat &tcw)
{
  assert(ctlPointsC.size() == ctlPointsW.size() && "世界坐标系下的点和相机坐标系下的点不对应");
  int nNum = ctlPointsC.size();
  cv::Point3f centroidW, centroidC;
  for (int jdx = 0; jdx < nNum; ++jdx)
  {
    centroidW += ctlPointsW[jdx];
    centroidC += ctlPointsC[jdx];
  }
  centroidC /= nNum;
  centroidW /= nNum;
  cv::Mat A(nNum, 3, CV_32F), B(nNum, 3, CV_32F);
  for (int idx = 0; idx < nNum; ++idx)
  {
    A.at<float>(idx, 0) = ctlPointsW[idx].x - centroidW.x;
    A.at<float>(idx, 1) = ctlPointsW[idx].y - centroidW.y;
    A.at<float>(idx, 2) = ctlPointsW[idx].z - centroidW.z;
    B.at<float>(idx, 0) = ctlPointsC[idx].x - centroidC.x;
    B.at<float>(idx, 1) = ctlPointsC[idx].y - centroidC.y;
    B.at<float>(idx, 2) = ctlPointsC[idx].z - centroidC.z;
  }
  cv::Mat H = B.t() * A;
  cv::Mat U, W, Vt;
  cv::SVD::compute(H, W, U, Vt);
  Rcw = U * Vt;
  float det = computeDet3(Rcw);
  if (det < 0)
  {
    Rcw.at<float>(2, 0) = -Rcw.at<float>(2, 0);
    Rcw.at<float>(2, 1) = -Rcw.at<float>(2, 1);
    Rcw.at<float>(2, 2) = -Rcw.at<float>(2, 2);
  }

  cv::Mat centroidCMat(3, 1, CV_32F), centroidWMat(3, 1, CV_32F);
  centroidCMat.at<float>(0, 0) = centroidC.x;
  centroidCMat.at<float>(1, 0) = centroidC.y;
  centroidCMat.at<float>(2, 0) = centroidC.z;
  centroidWMat.at<float>(0, 0) = centroidW.x;
  centroidWMat.at<float>(1, 0) = centroidW.y;
  centroidWMat.at<float>(2, 0) = centroidW.z;
  tcw = centroidCMat - Rcw * centroidWMat;
}

/**
 * @brief 使用PCA归一化的方法，求解控制点坐标
 *
 * @param vMapPoints 输入的所有地图点坐标
 * @return std::vector<cv::Point3f> 输出的世界坐标系下的控制点坐标
 */
std::vector<cv::Point3f> PnPSolver::computeCtlPoint(const std::vector<cv::Point3f> &vMapPoints, bool &bIsPositive)
{
  std::vector<cv::Point3f> vCtlPoints;
  int n = vMapPoints.size();
  cv::Point3f centroid(0, 0, 0);
  for (const auto &point : vMapPoints)
    centroid += point;
  centroid /= n;
  vCtlPoints.push_back(centroid);
  cv::Mat A(n, 3, CV_32F);
  for (int row = 0; row < n; ++row)
  {
    cv::Point3f pointDiff = vMapPoints[row] - centroid;
    A.at<float>(row, 0) = pointDiff.x;
    A.at<float>(row, 1) = pointDiff.y;
    A.at<float>(row, 2) = pointDiff.z;
  }
  A = A.t() * A;
  cv::Mat eigenVal, eigenVec;
  bool ret = cv::eigen(A, eigenVal, eigenVec);
  assert(ret && "获取矩阵特征值和特征向量函数出错！");
  for (int idx = 0; idx < 3; ++idx)
  {
    cv::Point3f point;
    float eigenVali = eigenVal.at<float>(idx, 0);
    if (eigenVali < 1e-3)
    {
      bIsPositive = false;
      break;
    }
    cv::Mat temp = std::sqrt(eigenVal.at<float>(idx, 0) / n) * eigenVec.row(idx);
    point.x = centroid.x + temp.at<float>(0, 0);
    point.y = centroid.y + temp.at<float>(0, 1);
    point.z = centroid.z + temp.at<float>(0, 2);
    vCtlPoints.push_back(point);
  }
  return vCtlPoints;
}

/**
 * @brief 计算alpha参数矩阵
 *
 * @param vMapPoints 地图点
 * @param vCtlPoints 控制点
 * @return cv::Mat 输出的alpha参数矩阵
 */
cv::Mat PnPSolver::computeAlpha(const std::vector<cv::Point3f> &vMapPoints, const std::vector<cv::Point3f> &vCtlPoints)
{
  int n = vMapPoints.size();
  cv::Mat alpha(n, 4, CV_32F);
  const cv::Point3f &centroid = vCtlPoints[0];
  for (int row = 0; row < n; ++row)
  {
    cv::Mat A(3, 3, CV_32F), b(3, 1, CV_32F);
    for (std::size_t idx = 1; idx < 4; ++idx)
    {
      cv::Point3f temp = vCtlPoints[idx] - centroid;
      A.at<float>(0, idx - 1) = temp.x;
      A.at<float>(1, idx - 1) = temp.y;
      A.at<float>(2, idx - 1) = temp.z;
    }
    cv::Point3f temp = vMapPoints[row] - centroid;
    b.at<float>(0, 0) = temp.x;
    b.at<float>(1, 0) = temp.y;
    b.at<float>(2, 0) = temp.z;
    cv::Mat alphaI;
    cv::solve(A, b, alphaI);
    float alpha1 = alpha.at<float>(row, 1) = alphaI.at<float>(0, 0);
    float alpha2 = alpha.at<float>(row, 2) = alphaI.at<float>(1, 0);
    float alpha3 = alpha.at<float>(row, 3) = alphaI.at<float>(2, 0);
    alpha.at<float>(row, 0) = 1.0f - alpha1 - alpha2 - alpha3;
  }
  return alpha;
}

/**
 * @brief 利用重投影约束，获取矩阵M。
 *
 * @param vORBPoints    输入的2D特征点
 * @param alpha         输入的alpha参数矩阵
 * @return cv::Mat      输出的M矩阵
 */
cv::Mat PnPSolver::computeMMat(const std::vector<cv::Point2f> &vORBPoints, const cv::Mat &alpha)
{
  int n = vORBPoints.size();
  cv::Mat M(2 * n, 12, CV_32F);
  for (int idx = 0; idx < n; ++idx)
  {
    const cv::Point2f &point = vORBPoints[idx];
    for (int jdx = 0; jdx < 4; ++jdx)
    {
      float alphaV = alpha.at<float>(idx, jdx);
      M.at<float>(idx * 2, jdx * 3 + 0) = alphaV * Camera::mfFx;
      M.at<float>(idx * 2, jdx * 3 + 1) = 0;
      M.at<float>(idx * 2, jdx * 3 + 2) = alphaV * (Camera::mfCx - point.x);

      M.at<float>(idx * 2 + 1, jdx * 3 + 0) = 0;
      M.at<float>(idx * 2 + 1, jdx * 3 + 1) = alphaV * Camera::mfFy;
      M.at<float>(idx * 2 + 1, jdx * 3 + 2) = alphaV * (Camera::mfCy - point.y);
    }
  }
  return M;
}

/**
 * @brief 计算矩阵M的右奇异值特征向量，并把他们分开（一个向量对应四个点）
 *
 * @param M 输入的M矩阵
 * @return std::vector<std::vector<cv::Point3f>> 输出的特征向量对应的3D点
 */
std::vector<std::vector<cv::Point3f>> PnPSolver::computeMREVec(const cv::Mat &M)
{
  assert(M.cols == 12 && "M矩阵不符合形状");
  cv::Mat temp = M.t() * M;
  cv::Mat _, eigenVec;
  bool ret = cv::eigen(temp, _, eigenVec);
  assert(ret && "获取矩阵特征值和特征向量函数出错！");
  std::vector<std::vector<cv::Point3f>> vMREVec;
  for (std::size_t idx = 11; idx > 7; --idx)
  {
    cv::Mat eigenVeci = eigenVec.row(idx);
    std::vector<cv::Point3f> vContent;
    for (std::size_t idx = 0; idx < 4; ++idx)
    {
      cv::Point3f point;
      point.x = eigenVeci.at<float>(0, 3 * idx + 0);
      point.y = eigenVeci.at<float>(0, 3 * idx + 1);
      point.z = eigenVeci.at<float>(0, 3 * idx + 2);
      vContent.push_back(point);
    }
    vMREVec.push_back(vContent);
  }
  return vMREVec;
}

/**
 * @brief 获取矩阵L
 *
 * @param vMREVec   输入的M矩阵的右奇异特征向量
 * @return cv::Mat  输出的L矩阵
 */
cv::Mat PnPSolver::computeLMat(const std::vector<std::vector<cv::Point3f>> &vMREVec)
{
  cv::Mat L(6, 10, CV_32F);
  int row = 0;
  for (int idx = 0; idx < 3; ++idx)
  {
    for (int jdx = idx + 1; jdx < 4; ++jdx)
    {
      cv::Point3f x1 = vMREVec[0][idx] - vMREVec[0][jdx];
      cv::Point3f x2 = vMREVec[1][idx] - vMREVec[1][jdx];
      cv::Point3f x3 = vMREVec[2][idx] - vMREVec[2][jdx];
      cv::Point3f x4 = vMREVec[3][idx] - vMREVec[3][jdx];
      L.at<float>(row, 0) = x1.dot(x1);
      L.at<float>(row, 1) = x2.dot(x2);
      L.at<float>(row, 2) = x3.dot(x3);
      L.at<float>(row, 3) = x4.dot(x4);
      L.at<float>(row, 4) = 2 * x1.dot(x2);
      L.at<float>(row, 5) = 2 * x1.dot(x3);
      L.at<float>(row, 6) = 2 * x1.dot(x4);
      L.at<float>(row, 7) = 2 * x2.dot(x3);
      L.at<float>(row, 8) = 2 * x2.dot(x4);
      L.at<float>(row, 9) = 2 * x3.dot(x4);
      ++row;
    }
  }
  return L;
}

/// 计算Rho矩阵（世界坐标系控制点的距离）
cv::Mat PnPSolver::computeRho(const std::vector<cv::Point3f> &vCtlPoints)
{
  cv::Mat rho(6, 1, CV_32F);
  int row = 0;
  for (int idx = 0; idx < 3; ++idx)
  {
    for (int jdx = idx + 1; jdx < 4; ++jdx)
    {
      rho.at<float>(row, 0) = dist2(vCtlPoints[idx], vCtlPoints[jdx]);
      ++row;
    }
  }
  return rho;
}

/**
 * @brief 计算Beta参数[beta1, beta2, beta3, beta4]
 * @details
 *      1. 取L矩阵的[0, 4, 5, 6]列，作为新矩阵
 *      2. 计算rho矩阵（世界坐标系控制点的距离）
 * @param L      输入的L矩阵
 * @param rho    输入的控制点距离矩阵（世界坐标系下）
 * @return cv::Mat 输出的beta的参数值
 */
cv::Mat PnPSolver::computeBetaUnOpt(const cv::Mat &L, const cv::Mat &rho)
{
  std::vector<cv::Mat> toMerge;
  std::vector<std::size_t> vIndices{0, 4, 5, 6};
  cv::Mat newL, beta;
  for (const auto &col : vIndices)
    toMerge.push_back(L.col(col));
  cv::hconcat(toMerge, newL);

  cv::solve(newL, rho, beta, cv::DECOMP_SVD);
  float betaPow2 = beta.at<float>(0, 0);
  if (betaPow2 < 0)
  {
    beta = -beta;
    betaPow2 = -betaPow2;
  }
  float beta1 = std::sqrt(betaPow2);
  beta.at<float>(0, 0) = beta1;
  beta.at<float>(1, 0) /= beta1;
  beta.at<float>(2, 0) /= beta1;
  beta.at<float>(3, 0) /= beta1;
  return beta;
}

/**
 * @brief 使用高斯牛顿优化器，进行beta矩阵的优化
 * @details
 *      1. f(beta) = Lβ-rho
 *      2. 计算雅可比矩阵J
 *      3. Hδx = b ==> J * J^T = -J * (Lβ-rho)
 * @param L         输入的L矩阵
 * @param vbeta     输入的beta向量
 * @param rho       输入的控制点距离矩阵（世界坐标系下）
 */
void PnPSolver::GLOptimize(const cv::Mat &L, cv::Mat &vbeta, const cv::Mat &rho, int maxIteration)
{
  float currError = 0, oldError = FLT_MAX;
  for (int idx = 0; idx < maxIteration; ++idx)
  {
    const float &beta1 = vbeta.at<float>(0, 0);
    const float &beta2 = vbeta.at<float>(1, 0);
    const float &beta3 = vbeta.at<float>(2, 0);
    const float &beta4 = vbeta.at<float>(3, 0);
    cv::Mat fullBeta = (cv::Mat_<float>(10, 1, CV_32F) << std::pow(beta1, 2), std::pow(beta2, 2), std::pow(beta3, 2), std::pow(beta4, 2), beta1 * beta2,
                        beta1 * beta3, beta1 * beta4, beta2 * beta3, beta2 * beta4, beta3 * beta4);
    cv::Mat deltaBeta;
    cv::Mat jacobi = computeJacobi(L, vbeta);
    cv::solve(jacobi * jacobi.t(), -jacobi * (L * fullBeta - rho), deltaBeta, cv::DECOMP_SVD);
    if (cv::norm(deltaBeta) < 1e-4)
      return;
    vbeta += deltaBeta;

    fullBeta = (cv::Mat_<float>(10, 1, CV_32F) << std::pow(beta1, 2), std::pow(beta2, 2), std::pow(beta3, 2), std::pow(beta4, 2), beta1 * beta2, beta1 * beta3,
                beta1 * beta4, beta2 * beta3, beta2 * beta4, beta3 * beta4);
    currError = cv::norm(L * fullBeta - rho);
    if (currError > oldError)
    {
      vbeta -= deltaBeta;
      return;
    }
    oldError = currError;
  }
}

/**
 * @brief 根据L(6 * 10)矩阵和beta向量，计算雅可比矩阵jcaobi
 *
 * @param L     输入的L矩阵6*10
 * @param vbeta 输入的beta向量4*1
 * @return cv::Mat 输出的jacobi矩阵4 * 6
 */
cv::Mat PnPSolver::computeJacobi(const cv::Mat &L, const cv::Mat &vbeta)
{
  cv::Mat J(4, 6, CV_32F); ///< 分母布局
  static std::vector<std::vector<int>> vJIndices{{1, 5, 6, 7}, {5, 2, 8, 9}, {6, 8, 3, 10}, {7, 9, 10, 4}};
  for (int col = 0; col < 6; ++col)
  {
    for (int row = 0; row < 4; ++row)
    {
      float Jval = 0;
      const auto &vJIndex = vJIndices[row];
      for (int idx = 0; idx < 4; ++idx)
      {
        const float &lVal = L.at<float>(col, vJIndex[idx] - 1);
        const float &betaVal = vbeta.at<float>(idx, 0);
        if (idx == row)
          Jval += 2 * lVal * betaVal;
        else
          Jval += lVal * betaVal;
      }
      J.at<float>(row, col) = Jval;
    }
  }
  return J;
}

/**
 * @brief 计算3*3矩阵的秩
 *
 * @param mat   输入的矩阵
 * @return float 输出的矩阵的秩
 */
float PnPSolver::computeDet3(const cv::Mat &mat)
{
  return mat.at<float>(0, 0) * mat.at<float>(1, 1) * mat.at<float>(2, 2) + mat.at<float>(0, 1) * mat.at<float>(1, 2) * mat.at<float>(2, 0) +
         mat.at<float>(0, 2) * mat.at<float>(1, 0) * mat.at<float>(2, 1) - mat.at<float>(0, 2) * mat.at<float>(1, 1) * mat.at<float>(2, 0) -
         mat.at<float>(0, 1) * mat.at<float>(1, 0) * mat.at<float>(2, 2) - mat.at<float>(0, 0) * mat.at<float>(1, 2) * mat.at<float>(2, 1);
}

/**
 * @brief 判断样本中内点和外点的数目
 *
 * @param vbInlierFlags 输出的样本中内点标记
 * @param Rcw           输入的Rcw（某一次RANSAC计算的旋转矩阵）
 * @param tcw           输入的tcw（某一次RANSAC计算的平移向量）
 * @return int 返回样本中内点的个数
 */
int PnPSolver::checkInliers(std::vector<std::size_t> &vnInlierIndices, const PnPRet &modelRet)
{
  int nInliers = 0;
  for (std::size_t idx = 0; idx < mnN; ++idx)
  {
    const cv::Point2f &point2d = mvORBPoints[idx];
    const cv::Point3f &point3d = mvMapPoints[idx];
    const float &error = mvfErrors[idx];
    cv::Mat p3dW = (cv::Mat_<float>(3, 1) << point3d.x, point3d.y, point3d.z);
    cv::Mat p3dC = modelRet.mRcw * p3dW + modelRet.mtcw;
    const float &z = p3dC.at<float>(2, 0);
    float u = p3dC.at<float>(0, 0) / z * Camera::mfFx + Camera::mfCx;
    float v = p3dC.at<float>(1, 0) / z * Camera::mfFy + Camera::mfCy;
    float err = (point2d.x - u) * (point2d.x - u) + (point2d.y - v) * (point2d.y - v);
    if (err < error)
    {
      vnInlierIndices.push_back(idx);
      ++nInliers;
    }
  }
  return nInliers;
}
} // namespace ORB_SLAM2_ROS2
