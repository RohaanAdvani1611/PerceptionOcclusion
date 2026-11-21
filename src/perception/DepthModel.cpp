#include "perception/DepthModel.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <vector>

cv::Mat normalizeDepth(const cv::Mat& raw, cv::Size outSize) {
    cv::Mat depth;
    cv::resize(raw, depth, outSize, 0, 0, cv::INTER_LINEAR);

    std::vector<float> vals(depth.begin<float>(), depth.end<float>());
    int n     = static_cast<int>(vals.size());
    int loIdx = std::max(0, static_cast<int>(0.01f * n));
    int hiIdx = std::min(n - 1, static_cast<int>(0.99f * n));
    std::nth_element(vals.begin(), vals.begin() + loIdx, vals.end());
    float pLow = vals[loIdx];
    std::nth_element(vals.begin() + loIdx + 1, vals.begin() + hiIdx, vals.end());
    float pHigh = vals[hiIdx];

    if (pHigh > pLow) {
        float range = pHigh - pLow;
        depth.forEach<float>([pLow, range](float& v, const int*) {
            v = std::max(0.0f, std::min(1.0f, (v - pLow) / range));
        });
    }
    return depth;
}
