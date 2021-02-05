#include <vector>
#include <algorithm>

#include <nms.h>

static bool sort_score(Bbox box1, Bbox box2) {
    return box1.score > box2.score ? true : false;
}

float iou(Bbox box1, Bbox box2) {
    int x1 = std::max(box1.x1, box2.x1);
    int y1 = std::max(box1.y1, box2.y1);
    int x2 = std::min(box1.x2, box2.x2);
    int y2 = std::min(box1.y2, box2.y2);
    int w = std::max(0, x2 - x1 + 1);
    int h = std::max(0, y2 - y1 + 1);
    float over_area = w * h;
    return over_area / ((box1.x2 - box1.x1) * (box1.y2 - box1.y1) + (box2.x2 - box2.x1) * (box2.y2 - box2.y1) - over_area);
}

void calnms(std::vector<Bbox>& vec_boxs, std::vector<Bbox>& results, float threshold, float minscore)
{
    std::vector<Bbox> vec_boxs_;
    for (int i = 0; i < vec_boxs.size(); i++) {
        if (vec_boxs[i].score > minscore) {
            vec_boxs_.push_back(vec_boxs[i]);
        }
    }

    std::sort(vec_boxs_.begin(), vec_boxs_.end(), sort_score);
    while (vec_boxs_.size() > 0) {
        results.push_back(vec_boxs_[0]);
        int index = 1;
        while (index < vec_boxs_.size()) {
            float iou_value = iou(vec_boxs_[0], vec_boxs_[index]);
            //std::cout << "iou:" << iou_value << std::endl;
            if (iou_value > threshold) {
                vec_boxs_.erase(vec_boxs_.begin() + index);
            }
            else {
                index++;
            }
        }
        vec_boxs_.erase(vec_boxs_.begin());
    }
}