#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <string>
using namespace std;
using namespace cv;
int main2()
{    
    /* 这一段不要放进循环里面去，*/
    bool isColor = true;//(show.type() == CV_8UC3);
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');  // select desired codec (must be available at runtime)
    double fps = 25.0;                          // framerate of the created video stream
    std::string filename = "Z:/ee/ee/live.avi"; //"E:/code/hand_collection/hand_collection/resource/video.avi";             // name of the output video file
    cv::VideoWriter writer;
    writer.open(filename, codec, fps, cv::Size(340 * 6, 340 * 3), isColor);
    if (!writer.isOpened()) {
        std::cout << "写入视频失败" << std::endl;
        system("pause");
        return -1;
    }
    
    int iter = 0;
    while (1)
    {
        
        cv::Mat frame = cv::imread("E:\\data\\0\\18517833\\" + std::to_string(iter) + "_18517833.bmp");
        if (frame.empty())
        {
            cout << "Video process finished!" << endl;
            return 0;
        }
        imshow("video", frame);
        if (waitKey(10) == 'q') break;
        writer << frame;
        iter++;
    }
    return 0;
}