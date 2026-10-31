#include "compositing/OcclusionPass.hpp"
#include <opencv2/imgproc.hpp>

void OcclusionPass::apply(const cv::Mat& realDepth, cv::Mat out) const {
    // Morphological close: fills small holes caused by depth noise near object edges,
    // preventing flickery punch-through on occlusion boundaries. Closing the depth map
    // gives the same result as closing a thresholded mask, at every sphere distance.
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

    if (realDepth.size() == out.size()) {
        cv::morphologyEx(realDepth, out, cv::MORPH_CLOSE, kernel);
    } else {
        cv::Mat closed;
        cv::morphologyEx(realDepth, closed, cv::MORPH_CLOSE, kernel);
        cv::resize(closed, out, out.size(), 0, 0, cv::INTER_LINEAR);
    }
}
