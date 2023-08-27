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
    std::shared_ptr<GPGPU::HostParameter> _randomSeedState;
    std::shared_ptr<GPGPU::HostParameter> _targetPositionIn;
    std::shared_ptr<GPGPU::HostParameter> _targetPositionOut;

    std::shared_ptr<GPGPU::HostParameter> _parametersRandomInit;
    std::shared_ptr<GPGPU::HostParameter> _parametersComputeTargetPosition;
    std::shared_ptr<GPGPU::HostParameter> _parameters;



    std::string _defineMacros;
    size_t _frameTime;
public:
    // width and height must be multiple of 16
    PlayArea(int width, int height, int maximumGPUsToUse = 10)
    {
        _width = width;
        _height = height;
        while (_width % 16 != 0)
            _width++;
        while (_height % 16 != 0)
            _height++;
        _totalCells = _width * _height;
        _computer = std::make_shared<GPGPU::Computer>(GPGPU::Computer::DEVICE_GPUS,-1,1,true, maximumGPUsToUse); // allocate all devices for computations

        // broadcast type input (duplicated on all gpus from ram)
        // load-balanced output                                                
        _areaIn = std::make_shared<GPGPU::HostParameter>(_computer->createArrayInput<unsigned char>("areaIn", _totalCells));
        _areaOut = std::make_shared<GPGPU::HostParameter>(_computer->createArrayOutput<unsigned char>("areaOut", _totalCells));
        _temperatureIn = std::make_shared<GPGPU::HostParameter>(_computer->createArrayInput<unsigned char>("temperatureIn", _totalCells));
        _temperatureOut = std::make_shared<GPGPU::HostParameter>(_computer->createArrayOutput<unsigned char>("temperatureOut", _totalCells));
        _randomSeedIn = std::make_shared<GPGPU::HostParameter>(_computer->createArrayInput<unsigned int>("randomSeedIn", _totalCells));
        _randomSeedState = std::make_shared<GPGPU::HostParameter>(_computer->createArrayState<unsigned int>("randomSeedState", _totalCells));
        _targetPositionIn = std::make_shared<GPGPU::HostParameter>(_computer->createArrayInput<unsigned char>("targetPositionIn", _totalCells));
        _targetPositionOut = std::make_shared<GPGPU::HostParameter>(_computer->createArrayOutput<unsigned char>("targetPositionOut", _totalCells));

        _parametersRandomInit = std::make_shared<GPGPU::HostParameter>(
            _randomSeedIn->next(*_randomSeedState)
        );

        _parametersComputeTargetPosition = std::make_shared<GPGPU::HostParameter>(
            _areaIn->next(*_temperatureIn).next(*_randomSeedState).next(*_targetPositionOut)
        );


        _parameters = std::make_shared<GPGPU::HostParameter>(
            _areaIn->next(*_areaOut).next(*_temperatureIn).next(*_temperatureOut).next(*_randomSeedState).next(*_targetPositionIn)
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
            kernel void initRandomSeed(
                const global unsigned int * __restrict__ randomSeedIn,
                global unsigned int * __restrict__ randomSeedState
            )
            {
                const int id=get_global_id(0);  
                const int N = get_global_size(0);
                const int nLoop = 1 + ((PLAY_AREA_WIDTH * PLAY_AREA_HEIGHT)  / N);
                for(int i=0;i<nLoop;i++)
                {
                    int idLoop = i * N + id%N;
                    if(idLoop < PLAY_AREA_WIDTH * PLAY_AREA_HEIGHT)
                        randomSeedState[idLoop]=randomSeedIn[idLoop];
                }
            }
        )", "initRandomSeed");

        _computer->compile(_defineMacros + R"(
           kernel void computePositionTarget(
                const global unsigned char * __restrict__ areaIn,
                const global unsigned char * __restrict__ temperatureIn,
                global unsigned int * __restrict__ randomSeedState,
                global unsigned char * __restrict__ targetPositionOut
            )
            {
                const int id=get_global_id(0);                                 
                unsigned int randomSeed = randomSeedState[id];
                float motionPossibility = randomFloat(&randomSeed) * 255;
                unsigned char newPos = 0;   
                float positionPossibility = randomFloat(&randomSeed); 
                if(areaIn[id]>0 && temperatureIn[id] > motionPossibility)
                {                                        
                    newPos = 1 + (positionPossibility/0.25f); // 1 = top, 2 = right, 3 = bot, 4 = left
                }
                else if(areaIn[id]>0)
                {
                    // gravity
                    newPos = 3;
                }
                else if(areaIn[id] == 0)
                {
                    // 100+ means selecting which source to accept (when it is empty)
                    newPos = 101 + (positionPossibility/0.25f); //  101= top, 102 = right, 103 = bot, 104 = left
                }
                randomSeedState[id]=randomSeed;
                targetPositionOut[id]=newPos;
            }
        )","computePositionTarget");
        
        _computer->compile(_defineMacros + R"(
    

            // todo: compute 5x5 neighborhood
            kernel void computeSand(
                const global unsigned char * __restrict__ areaIn, global unsigned char * __restrict__ areaOut,
                const global unsigned char * __restrict__ temperatureIn, global unsigned char * __restrict__ temperatureOut,
                global unsigned int * __restrict__ randomSeedState,
                const global unsigned char * __restrict__ targetPositionIn
            ) 
            { 
                const int id=get_global_id(0); 
                const int x = id % PLAY_AREA_WIDTH;
                const int y = id / PLAY_AREA_WIDTH;
                unsigned int randomSeed = randomSeedState[id];
                int idTop = (y - 1) * PLAY_AREA_WIDTH + x;
                unsigned char top = (y>0 ? areaIn[idTop] : 0);
                unsigned char targetTop = (y>0 ? targetPositionIn[idTop] : 0);

                int idLeft = y * PLAY_AREA_WIDTH + (x-1);
                unsigned char left =(x>0 ? areaIn[idLeft] : 0);                                                
                unsigned char targetLeft =(x>0 ? targetPositionIn[idLeft] : 0);                                                

                int idRight = y * PLAY_AREA_WIDTH + (x+1);
                unsigned char right =(x<PLAY_AREA_WIDTH-1 ? areaIn[idRight] : 0);             
                unsigned char targetRight =(x<PLAY_AREA_WIDTH-1 ? targetPositionIn[idRight] : 0);

                unsigned char center = areaIn[id];
                unsigned char targetAccepted = targetPositionIn[id];

                int idBot = (y + 1) * PLAY_AREA_WIDTH + x;
                unsigned char bot = (y<PLAY_AREA_HEIGHT-1 ? areaIn[idBot] : 0);
                unsigned char targetBot = (y<PLAY_AREA_HEIGHT-1 ? targetPositionIn[idBot] : 0);

                if (center == 1)
                {
                    unsigned char targetPos = targetPositionIn[id];
                    if(targetPos == 1 && top == 0 && targetTop == 103)
                    {   
                        areaOut[id]=0;
                        temperatureOut[id]=0;
                    }
                    else if(targetPos == 2 && right == 0  && targetRight == 104)
                    {
                        areaOut[id]=0;
                        temperatureOut[id]=0;
                    }
                    else if(targetPos == 3 && bot == 0 && targetBot == 101)
                    {
                        areaOut[id]=0;
                        temperatureOut[id]=0;
                    }
                    else if(targetPos == 4 && left == 0 && targetLeft == 102)
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
                    if(top == 1 && targetTop == 3 && targetAccepted == 101)
                    {
                        sourceId = idTop;
                    }
                    else if(right == 1 && targetRight == 4  && targetAccepted == 102)
                    {
                        sourceId = idRight;
                    }
                    else if(bot == 1 && targetBot == 1 && targetAccepted == 103)
                    {
                        sourceId = idBot;
                    }
                    else if(left == 1 && targetLeft == 2 && targetAccepted == 104)
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
                        temperatureOut[id]=0;
                    }
                }
                randomSeedState[id]=randomSeed;
             })", "computeSand");


        Reset();
    }
    
    void Reset()
    {
        
        for (int i = 0; i < _width * _height; i++)
        {
            _areaIn->access<unsigned char>(i) = 0;
            _temperatureIn->access<unsigned char>(i) = 0.0f;
            _randomSeedIn->access<unsigned int>(i) = i;
            _targetPositionIn->access<unsigned char>(i) = 0;
        }
        _computer->compute(*_parametersRandomInit, "initRandomSeed", 0, _totalCells, 256);
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
        _targetPositionOut->copyDataToPtr(_targetPositionIn->accessPtr<unsigned char>(0));

        _computer->compute(*_parameters, "computeSand", 0, _totalCells, 256);
        _areaOut->copyDataToPtr(_areaIn->accessPtr<unsigned char>(0));
        _temperatureOut->copyDataToPtr(_temperatureIn->accessPtr<unsigned char>(0));
    }



    void AddSandToCursorPosition(int x, int y)
    {
        for (int j = -5; j <= 5; j++)
            for (int i = -5; i <= 5; i++)
                if (x + i >= 0 && x + i < _width && y + j >= 0 && y + j < _height)
                {
                    auto id = x + i + (y + j) * _width;
                    _areaIn->access<unsigned char>(id) = 1;
                    _temperatureIn->access<unsigned char>(id) = 0.35f * 255;
                }
    }

    void Render(cv::Mat& frame)
    {

        for (int j = 0; j < frame.rows; j++)
            for (int i = 0; i < frame.cols; i++)
            {
                unsigned char matter = _areaOut->access<unsigned char>(i + j * _width);
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