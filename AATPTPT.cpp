
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

    // currently only 1 GPU supported
    int maxGPUs = 1; 

    // change this if you have an iGPU and a dGPU. discrete GPUs are generally faster because of high bandwidth memory
    // if there are 2 gpus in system, index can be 0 or 1 but if a greater index is given then it picks last device
    int indexGPU = 2; 


    // rtx-4070 can do 20000 steps per second, this makes 100 updates per second (sand falls at 100 pixels per update speed)
    // iGPU of Ryzen 7000 series CPU can do 500 steps per second
    // Ryzen 7900 CPU cores can do 1200 steps per second
    int stepsPerFrame = 200; 

    // doesn't work yet
    int quantumStrength = 1;

    PlayArea area(w,h,maxGPUs, indexGPU,stepsPerFrame,quantumStrength);
    

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

