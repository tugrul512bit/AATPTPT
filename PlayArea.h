#pragma once
#include "gpgpu/gpgpu.hpp"
#include<memory>
#include<string>
struct PlayArea
{
private:
    int _width;
    int _height;
    int _totalCells;
    std::shared_ptr<GPGPU::Computer> _computer;
    std::shared_ptr<GPGPU::HostParameter> _areaIn;
    std::shared_ptr<GPGPU::HostParameter> _areaOut;
    std::shared_ptr<GPGPU::HostParameter> _temperatureIn;
    std::shared_ptr<GPGPU::HostParameter> _temperatureOut;
    std::shared_ptr<GPGPU::HostParameter> _parameters;



    std::string _defineMacros;
    size_t _frameTime;
public:
    // width and height must be multiple of 16
    PlayArea(int width, int height)
    {
        _width = width;
        _height = height;
        while (_width % 16 != 0)
            _width++;
        while (_height % 16 != 0)
            _height++;
        _totalCells = _width * _height;
        _computer = std::make_shared<GPGPU::Computer>(GPGPU::Computer::DEVICE_GPUS); // allocate all devices for computations

        // broadcast type input (duplicated on all gpus from ram)
        // load-balanced output
        _areaIn = std::make_shared<GPGPU::HostParameter>(_computer->createArrayInput<int>("areaIn", _totalCells));
        _areaOut = std::make_shared<GPGPU::HostParameter>(_computer->createArrayOutput<int>("areaOut", _totalCells));
        _temperatureIn = std::make_shared<GPGPU::HostParameter>(_computer->createArrayInput<float>("temperatureIn", _totalCells));
        _temperatureOut = std::make_shared<GPGPU::HostParameter>(_computer->createArrayOutput<float>("temperatureOut", _totalCells));
        _parameters = std::make_shared<GPGPU::HostParameter>(_areaIn->next(*_areaOut).next(*_temperatureIn).next(*_temperatureOut));
       
        _defineMacros = std::string("#define PLAY_AREA_WIDTH ")+std::to_string(_width)+R"(
        )";
        _defineMacros += std::string("#define PLAY_AREA_HEIGHT ") + std::to_string(_height) + R"(
        )";
        _defineMacros += std::string("#define PLAY_AREA_TOTAL_CELLS ") + std::to_string(_totalCells) + R"(
        )";

        // kernel for matter: falling sand
        _computer->compile(_defineMacros+R"(
            // todo: compute 5x5 neighborhood
            kernel void computeSand(
                const global int * __restrict__ areaIn, global int * __restrict__ areaOut,
                const global float * __restrict__ temperatureIn, global float * __restrict__ temperatureOut) 
            { 
                const int id=get_global_id(0); 
                const int x = id % PLAY_AREA_WIDTH;
                const int y = id / PLAY_AREA_WIDTH;
                int idTop = (y - 1) * PLAY_AREA_WIDTH + x;
                int top = (y>0 ? areaIn[idTop] : 0);
                int center = areaIn[id];
                int idBot = (y + 1) * PLAY_AREA_WIDTH + x;
                int bot = (y<PLAY_AREA_HEIGHT-1 ? areaIn[idBot] : 0);
                if(top == 1 && center == 0)
                    areaOut[id] = 1;
                else if(top == 0 && center == 1 && bot == 0)
                    areaOut[id] = 0;
                else if(top == 1 && center == 1 && bot == 0)
                    areaOut[id] = 1;
                else if(top == 0 && center == 1 && bot == 1)
                    areaOut[id] = 0;
                else
                    areaOut[id] = areaIn[id];
             })", "computeSand");


     
    }
    
    void Reset()
    {
        for (int i = 0; i < _width * _height; i++)
        {
            _areaIn->access<int>(i) = 0;
            _temperatureIn->access<float>(i) = 0.0f;
        }
    }

    void Calc()
    {
        {
            GPGPU::Bench bench(&_frameTime);
            CalcFallingSand();
            
        }
    } 

    void CalcFallingSand()
    {
        _computer->compute(*_parameters, "computeSand", 0, _totalCells, 256);
        _areaOut->copyDataToPtr(_areaIn->accessPtr<int>(0));
        _temperatureOut->copyDataToPtr(_temperatureIn->accessPtr<float>(0));
    }



    void AddSandToCursorPosition(int x, int y)
    {
        for (int j = -5; j <= 5; j++)
            for (int i = -5; i <= 5; i++)
                if(x+i>=0 && x+i<_width && y+j>=0 && y+j<_height)
                    _areaIn->access<int>(x + i + (y + j) * _width) = 1;
    }

    void Render(cv::Mat& frame)
    {

        for (int j = 0; j < frame.rows; j++)
            for (int i = 0; i < frame.cols; i++)
            {
                int matter = _areaOut->access<int>(i + j * _width);
                if (matter == 1)
                {
                    frame.at<cv::Vec3b>(i + j * _width).val[0] = 100;
                    frame.at<cv::Vec3b>(i + j * _width).val[1] = 50;
                    frame.at<cv::Vec3b>(i + j * _width).val[2] = 25;
                }
                else
                {
                    frame.at<cv::Vec3b>(i + j * _width).val[0] = 0;
                    frame.at<cv::Vec3b>(i + j * _width).val[1] = 0;
                    frame.at<cv::Vec3b>(i + j * _width).val[2] = 0;
                }
            }

        cv::putText(frame, std::to_string(_frameTime / 1000000000.0) + std::string(" seconds"), cv::Point2f(76, 76), 1, 5, cv::Scalar(50, 59, 69));
    }

};