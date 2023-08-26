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
    std::shared_ptr<GPGPU::HostParameter> _randomSeedIn;
    std::shared_ptr<GPGPU::HostParameter> _randomSeedOut;
    std::shared_ptr<GPGPU::HostParameter> _targetPositionIn;
    std::shared_ptr<GPGPU::HostParameter> _targetPositionOut;
    std::shared_ptr<GPGPU::HostParameter> _parametersComputeTargetPosition;
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
        _randomSeedIn = std::make_shared<GPGPU::HostParameter>(_computer->createArrayInput<unsigned int>("randomSeedIn", _totalCells));
        _randomSeedOut = std::make_shared<GPGPU::HostParameter>(_computer->createArrayOutput<unsigned int>("randomSeedOut", _totalCells));
        _targetPositionIn = std::make_shared<GPGPU::HostParameter>(_computer->createArrayInput<unsigned char>("targetPositionIn", _totalCells));
        _targetPositionOut = std::make_shared<GPGPU::HostParameter>(_computer->createArrayOutput<unsigned char>("targetPositionOut", _totalCells));

        _parametersComputeTargetPosition = std::make_shared<GPGPU::HostParameter>(
            _areaIn->next(*_temperatureIn).next(*_randomSeedIn).next(*_randomSeedOut).next(*_targetPositionOut)
        );


        _parameters = std::make_shared<GPGPU::HostParameter>(
            _areaIn->next(*_areaOut).next(*_temperatureIn).next(*_temperatureOut).next(*_randomSeedIn).next(*_randomSeedOut).next(*_targetPositionIn)
        );

        _defineMacros = std::string("#define PLAY_AREA_WIDTH ") + std::to_string(_width) + R"(
        )";
        _defineMacros += std::string("#define PLAY_AREA_HEIGHT ") + std::to_string(_height) + R"(
        )";
        _defineMacros += std::string("#define PLAY_AREA_TOTAL_CELLS ") + std::to_string(_totalCells) + R"(
        )";

        _defineMacros += R"(

            #define UIMAXFLOATINV (2.32830644e-10f)

   		    const unsigned int rnd(unsigned int seed)
		    {			
			    seed = (seed ^ 61) ^ (seed >> 16);
			    seed *= 9;
			    seed = seed ^ (seed >> 4);
			    seed *= 0x27d4eb2d;
			    seed = seed ^ (seed >> 15);
			    return seed;
		    }

            const float randomFloat(unsigned int * seed)
            {
                unsigned int newSeed = rnd(*seed);
                *seed = newSeed;                
                return newSeed * UIMAXFLOATINV;
            }

        )";

        _computer->compile(_defineMacros + R"(
           kernel void computePositionTarget(
                const global int * __restrict__ areaIn,
                const global float * __restrict__ temperatureIn,
                const global unsigned int * __restrict__ randomSeedIn, global unsigned int * __restrict__ randomSeedOut,
                global unsigned char * __restrict__ targetPositionOut
            )
            {
                const int id=get_global_id(0);                                 
                unsigned int randomSeed = randomSeedIn[id];
                float motionPossibility = randomFloat(&randomSeed);
                unsigned char newPos = -1;                
                if(temperatureIn[id] > motionPossibility)
                {                    
                    float positionPossibility = randomFloat(&randomSeed);
                    newPos = positionPossibility/0.25f; // 0 = top, 1 = right, 2 = bot, 3 = left
                }
                randomSeedOut[id]=randomSeed;
                targetPositionOut[id]=newPos;
            }
        )","computePositionTarget");
        
        _computer->compile(_defineMacros + R"(
    

            // todo: compute 5x5 neighborhood
            kernel void computeSand(
                const global int * __restrict__ areaIn, global int * __restrict__ areaOut,
                const global float * __restrict__ temperatureIn, global float * __restrict__ temperatureOut,
                const global unsigned int * __restrict__ randomSeedIn, global unsigned int * __restrict__ randomSeedOut,
                const global unsigned char * __restrict__ targetPositionIn
            ) 
            { 
                const int id=get_global_id(0); 
                const int x = id % PLAY_AREA_WIDTH;
                const int y = id / PLAY_AREA_WIDTH;
                unsigned int randomSeed = randomSeedIn[id];
                int idTop = (y - 1) * PLAY_AREA_WIDTH + x;
                int top = (y>0 ? areaIn[idTop] : 0);
                unsigned char targetTop = (y>0 ? targetPositionIn[idTop] : -1);

                int idLeft = y * PLAY_AREA_WIDTH + (x-1);
                int left =(x>0 ? areaIn[idLeft] : 0);                                                
                unsigned char targetLeft =(x>0 ? targetPositionIn[idLeft] : -1);                                                

                int idRight = y * PLAY_AREA_WIDTH + (x+1);
                int right =(x<PLAY_AREA_WIDTH-1 ? areaIn[idRight] : 0);             
                unsigned char targetRight =(x<PLAY_AREA_WIDTH-1 ? targetPositionIn[idRight] : -1);             

                int center = areaIn[id];

                int idBot = (y + 1) * PLAY_AREA_WIDTH + x;
                int bot = (y<PLAY_AREA_HEIGHT-1 ? areaIn[idBot] : 0);
                unsigned char targetBot = (y<PLAY_AREA_HEIGHT-1 ? targetPositionIn[idBot] : -1);

                if (center == 1)
                {
                    unsigned char targetPos = targetPositionIn[id];
                    if(targetPos == 0 && top == 0)
                    {   
                        areaOut[id]=0;
                        temperatureOut[id]=0;
                    }
                    else if(targetPos == 1 && right == 0)
                    {
                        areaOut[id]=0;
                        temperatureOut[id]=0;
                    }
                    else if(targetPos == 2 && bot == 0)
                    {
                        areaOut[id]=0;
                        temperatureOut[id]=0;
                    }
                    else if(targetPos == 3 && left == 0)
                    {
                        areaOut[id]=0;
                        temperatureOut[id]=0;
                    }
                    else // no movement
                    {
                        areaOut[id]=center;
                        temperatureOut[id]=temperatureIn[id];
                    }

                }
                else
                {
                    // if top cell is filled and its going down
                    int sourceId = -1;                    
                    if(top == 1 && targetTop == 2)
                    {
                        sourceId = idTop;
                    }
                    else if(right == 1 && targetRight == 3)
                    {
                        sourceId = idRight;
                    }
                    else if(bot == 1 && targetBot == 0)
                    {
                        sourceId = idBot;
                    }
                    else if(left == 1 && targetLeft == 1)
                    {
                        sourceId = idLeft;
                    }
            
                    if(sourceId != -1)
                    {
                     
                        areaOut[id] = areaIn[sourceId];
                        temperatureOut[id] = temperatureIn[sourceId];
                    }
                    else
                    {
                        areaOut[id]=0;
                        temperatureOut[id]=0.0f;
                    }
                }
                randomSeedOut[id]=randomSeed;
             })", "computeSand");


        Reset();
    }
    
    void Reset()
    {
        for (int i = 0; i < _width * _height; i++)
        {
            _areaIn->access<int>(i) = 0;
            _temperatureIn->access<float>(i) = 0.0f;
            _randomSeedIn->access<int>(i) = i;
            _targetPositionIn->access<unsigned char>(i) = -1;
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
        _computer->compute(*_parametersComputeTargetPosition, "computePositionTarget", 0, _totalCells, 256);
        _randomSeedOut->copyDataToPtr(_randomSeedIn->accessPtr<unsigned int>(0));
        _targetPositionOut->copyDataToPtr(_targetPositionIn->accessPtr<unsigned char>(0));

        _computer->compute(*_parameters, "computeSand", 0, _totalCells, 256);
        _areaOut->copyDataToPtr(_areaIn->accessPtr<int>(0));
        _temperatureOut->copyDataToPtr(_temperatureIn->accessPtr<float>(0));
        _randomSeedOut->copyDataToPtr(_randomSeedIn->accessPtr<unsigned int>(0));
    }



    void AddSandToCursorPosition(int x, int y)
    {
        for (int j = -5; j <= 5; j++)
            for (int i = -5; i <= 5; i++)
                if (x + i >= 0 && x + i < _width && y + j >= 0 && y + j < _height)
                {
                    auto id = x + i + (y + j) * _width;
                    _areaIn->access<int>(id) = 1;
                    _temperatureIn->access<float>(id) = 0.5f;
                }
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