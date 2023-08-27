
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

    int w = 1600;
    int h = 900;
    Mouse mouse;


    int maxGPUs = 2; // some systems have duplicated drivers that become an issue for pcie-bandwidth utilization. To limit maximum number of gpus to use, use this value.
    PlayArea area(w,h,maxGPUs);
    

    std::cout << "Hello World!\n";
    
    cv::setMouseCallback("AATPTPT", click, &mouse);
    while (cv::waitKey(1) != 27)
    {
    
        if (mouse.click)
        {
            area.AddSandToCursorPosition(mouse.x, mouse.y);
        }
    
        area.Calc();
        area.Render();
        
    }


    
    cv::destroyWindow("AATPTPT");
    return 0;
}

