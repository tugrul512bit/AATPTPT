
#include <iostream>
#include "vcpkg_installed/x86-windows/x86-windows/include/opencv2/opencv.hpp"
#include "vcpkg_installed/x86-windows/x86-windows/include/opencv2/highgui.hpp"

#include "PlayArea.h"

struct Mouse
{
    bool click;
    bool clickr;
    bool shift;
    float x, y;
    Mouse() { clickr = false;  click = false; x = 0; y = 0; }

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

    if (event == cv::EVENT_RBUTTONDOWN)
    {
        ((Mouse*)param)->clickr = true;

    }
    if (event == cv::EVENT_RBUTTONUP)
    {
        ((Mouse*)param)->clickr = false;
    }

    return;
}



int main()
{

    int w = 1600;
    int h = 900;
    Mouse mouse;


    int maxGPUs = 2; // some systems have duplicated drivers that become an issue for pcie-bandwidth utilization. To limit maximum number of gpus to use, use this value.
    int stepsPerFrame = 10;
    int quantumStrength = 1;
    PlayArea area(w,h,maxGPUs,stepsPerFrame,quantumStrength);
    

    std::cout << "Hello World!\n";
    
    cv::setMouseCallback("AATPTPT", click, &mouse);
    int key = 0;
    while ((key = cv::waitKey(1)) != 27)
    {

        if (mouse.click)
            area.AddSandToCursorPosition(mouse.x, mouse.y);



        if(mouse.clickr)
            area.RemoveSandFromCursorPosition(mouse.x, mouse.y);

        if (key == 'r')
        {
            area.Reset();
        }

        area.Calc();
        area.Render();
        
    }


    
    cv::destroyWindow("AATPTPT");
    return 0;
}

