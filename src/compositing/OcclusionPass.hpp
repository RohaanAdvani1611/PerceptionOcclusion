#pragma once
#include <opencv2/opencv.hpp>

class OcclusionPass {
public:
    // rendered  : BGRA CV_8UC4 — sphere pixels with alpha=255, background alpha=0
    // realDepth : CV_32F [0,1] — MiDaS smoothed depth, 1=closest to camera
    // threshold : pixels where realDepth > threshold are treated as foreground
    //             and will occlude the sphere (alpha zeroed out)
    // Returns a new BGRA mat with the sphere masked where real geometry is in front.
    cv::Mat apply(const cv::Mat& rendered,
                  const cv::Mat& realDepth,
                  float          threshold) const;
};
