#pragma once

typedef struct Bbox {
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    int cls;
}Bbox;

//---------------------------------------------------------------------------------------------------//
static bool sort_score(Bbox box1, Bbox box2);
float iou(Bbox box1, Bbox box2);
void calnms(std::vector<Bbox>& vec_boxs, std::vector<Bbox>& results, float threshold, float minscore);
