
#include <iostream>
#include "vcpkg_installed/x86-windows/x86-windows/include/opencv2/opencv.hpp"
#include "vcpkg_installed/x86-windows/x86-windows/include/opencv2/highgui.hpp"

#include "PlayArea.h"

struct Mouse
{
    bool click;
    float x, y;
    Mouse() { click = false; x = 0; y = 0; }

};
static void click(int event, int x, int y, int flags, void* param)
{
    ((Mouse*)param)->x = x;
    ((Mouse*)param)->y = y;
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        ((Mouse*)param)->click = true;

    }
    if (event == cv::EVENT_LBUTTONUP)
    {
        ((Mouse*)param)->click = false;
    }


    return;
}

int main()
{
    Mouse mouse;
    cv::Mat frame(600, 800,CV_8UC3);
    PlayArea area(800,600);
    std::cout << "Hello World!\n";
    cv::namedWindow("AATPTPT");
    cv::setMouseCallback("AATPTPT", click, &mouse);
    while (cv::waitKey(1) != 27)
    {
    
        if (mouse.click)
        {
            area.AddSandToCursorPosition(mouse.x, mouse.y);
        }
    
        area.Calc();
        area.Render(frame);
        cv::imshow("AATPTPT", frame);
    }


    
    cv::destroyWindow("AATPTPT");
    return 0;
}

