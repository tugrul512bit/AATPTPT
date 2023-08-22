
#include <iostream>
#include "vcpkg_installed/x86-windows/x86-windows/include/opencv2/opencv.hpp"
#include "vcpkg_installed/x86-windows/x86-windows/include/opencv2/highgui.hpp"

int main()
{
    std::cout << "Hello World!\n";
    cv::namedWindow("AATPTPT");
    cv::waitKey(0);
    return 0;
}

