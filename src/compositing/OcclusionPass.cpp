#include "compositing/OcclusionPass.hpp"
#include <opencv2/imgproc.hpp>

cv::Mat OcclusionPass::apply(const cv::Mat& rendered,
                              const cv::Mat& realDepth,
                              float          threshold) const {
    if (rendered.empty() || realDepth.empty()) return rendered;

    // Foreground mask: 255 where a real object is close enough to occlude the sphere
    cv::Mat foreground;
    cv::threshold(realDepth, foreground, threshold, 255.0, cv::THRESH_BINARY);
    foreground.convertTo(foreground, CV_8U);

    // Morphological close: fills small holes caused by depth noise near object edges,
    // preventing flickery punch-through on occlusion boundaries
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(foreground, foreground, cv::MORPH_CLOSE, kernel);

    // Resize mask to match rendered dimensions if they differ (they shouldn't, but be safe)
    if (foreground.size() != rendered.size())
        cv::resize(foreground, foreground, rendered.size(), 0, 0, cv::INTER_NEAREST);

    // Zero out sphere alpha wherever a real object is in front
    cv::Mat result = rendered.clone();
    std::vector<cv::Mat> ch(4);
    cv::split(result, ch); // [B, G, R, A]
    ch[3].setTo(0, foreground);
    cv::merge(ch, result);
    return result;
}
