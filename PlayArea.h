#pragma once
#include "gpgpu/gpgpu.hpp"
#include<memory>
#include<string>
#include<atomic>

struct PlayArea
{
private:
    int _width;
    int _height;
    int _totalCells;
    std::shared_ptr<GPGPU::Computer> _computer;
    std::shared_ptr<GPGPU::HostParameter> _areaIn;
    std::shared_ptr<GPGPU::HostParameter> _areaOut;
    std::shared_ptr<GPGPU::HostParameter> _areaTargetSourceIn;
    std::shared_ptr<GPGPU::HostParameter> _areaTargetSourceOut;

    std::shared_ptr<GPGPU::HostParameter> _randomSeedIn;
    std::shared_ptr<GPGPU::HostParameter> _randomSeedState;

    std::shared_ptr<GPGPU::HostParameter> _parametersRandomInit;
    std::shared_ptr<GPGPU::HostParameter> _parameterSand;
    std::shared_ptr<GPGPU::HostParameter> _parameterSandMove;


    std::string _defineMacros;
    size_t _frameTime;
    int _numComputePerFrame;
    int _quantumStrength;
public:
    // width and height must be multiple of 16
    PlayArea(int & width, int & height, int maximumGPUsToUse = 10, int numStepsPerFrame=10, int quantumStrength=1)
    {
        cv::namedWindow("AATPTPT");
        _numComputePerFrame = numStepsPerFrame;
        _width = width;
        _height = height;
        while (_width % 16 != 0)
            _width++;
        while (_height % 16 != 0)
            _height++;
        height = _height;
        width = _width;
        _totalCells = _width * _height;
        _quantumStrength = quantumStrength;
        _computer = std::make_shared<GPGPU::Computer>(GPGPU::Computer::DEVICE_GPUS,-1,1,false, maximumGPUsToUse); // allocate all devices for computations

        // broadcast type input (duplicated on all gpus from ram)
        // load-balanced output                                                
        _areaIn = std::make_shared<GPGPU::HostParameter>(_computer->createArrayInput<unsigned char>("areaIn", _totalCells));
        _areaOut = std::make_shared<GPGPU::HostParameter>(_computer->createArrayOutput<unsigned char>("areaOut", _totalCells));
        _areaTargetSourceIn = std::make_shared<GPGPU::HostParameter>(_computer->createArrayInput<unsigned char>("areaTargetSourceIn", _totalCells));
        _areaTargetSourceOut = std::make_shared<GPGPU::HostParameter>(_computer->createArrayOutput<unsigned char>("areaTargetSourceOut", _totalCells));
     
        _randomSeedIn = std::make_shared<GPGPU::HostParameter>(_computer->createArrayInput<unsigned int>("randomSeedIn", _totalCells));
        _randomSeedState = std::make_shared<GPGPU::HostParameter>(_computer->createArrayState<unsigned int>("randomSeedState", _totalCells));

        _parametersRandomInit = std::make_shared<GPGPU::HostParameter>(
            _randomSeedIn->next(*_randomSeedState)
        );

        _parameterSand = std::make_shared<GPGPU::HostParameter>(
            _areaIn->next(*_randomSeedState).next(*_areaTargetSourceOut)
        );


        _parameterSandMove = std::make_shared<GPGPU::HostParameter>(
            _areaTargetSourceIn->next(*_areaIn).next(*_areaOut)
        );
      

        _defineMacros = std::string("#define PLAY_AREA_WIDTH ") + std::to_string(_width) + R"(
        )";
        _defineMacros += std::string("#define PLAY_AREA_HEIGHT ") + std::to_string(_height) + R"(
        )";
        _defineMacros += std::string("#define PLAY_AREA_TOTAL_CELLS ") + std::to_string(_totalCells) + R"(
        )";

        _defineMacros += std::string("#define PLAY_AREA_QUANTUM_STRENGTH ") + std::to_string(_quantumStrength) + R"(
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
    

            // todo: compute 5x5 neighborhood
            kernel void computeSand(
                const global unsigned char * __restrict__ areaIn, 
                global unsigned int * __restrict__ randomSeedState,
                global unsigned char * __restrict__ areaTargetSource
            ) 
            { 
                const int id=get_global_id(0); 
                const int x = id % PLAY_AREA_WIDTH;
                const int y = id / PLAY_AREA_WIDTH;
                unsigned int randomSeed = randomSeedState[id];
                
                const int matter = areaIn[id];

                const int topIdY = (y==0?y:y-1);
                const int topIdX = x;
                const int rightIdY = y;
                const int rightIdX = (x==PLAY_AREA_WIDTH - 1 ? x:x+1);
                const int botIdY = (y==PLAY_AREA_HEIGHT-1?y:y+1);
                const int botIdX = x;
                const int leftIdY = y;
                const int leftIdX = (x==0 ? x:x-1);

                const unsigned char top = areaIn[topIdX + topIdY * PLAY_AREA_WIDTH];
                const unsigned char right = areaIn[rightIdX + rightIdY * PLAY_AREA_WIDTH];
                const unsigned char bot = areaIn[botIdX + botIdY * PLAY_AREA_WIDTH];
                const unsigned char left = areaIn[leftIdX + leftIdY * PLAY_AREA_WIDTH];

                // quantum-mechenical pressure solver
                // a particle can go only 4 places
                // having more particles on target pos lowers the probability for there
                // then particle picks a pos to go, marks its id (does not move yet because parallel update will be done)
                // todo: weighted probabilities
           
               
                int options[8]={-1,-1,-1,-1,-1,-1,-1,-1};
                int optIndex = 0;
                // check where can 1 matter go
                // gravity
                if(bot<255-PLAY_AREA_QUANTUM_STRENGTH*4 && matter > PLAY_AREA_QUANTUM_STRENGTH*4 && botIdY != y)
                    options[optIndex++]=4;

                // matter leaving
                if(matter > 0)
                {
                    if(matter > top && top<255-PLAY_AREA_QUANTUM_STRENGTH*4)
                        options[optIndex++]=1;
                    if(matter > right && right<255-PLAY_AREA_QUANTUM_STRENGTH*4)
                        options[optIndex++]=2;
                    if(matter > bot && bot<255-PLAY_AREA_QUANTUM_STRENGTH*4)
                        options[optIndex++]=4;
                    if(matter > left && left<255-PLAY_AREA_QUANTUM_STRENGTH*4)
                        options[optIndex++]=8;
                }

                // matter arriving
                if(matter<255)
                {
                    if(matter < top && top > PLAY_AREA_QUANTUM_STRENGTH*4)
                        options[optIndex++]=16;
                    if(matter < right && right > PLAY_AREA_QUANTUM_STRENGTH*4)
                        options[optIndex++]=32;
                    if(matter < bot && bot > PLAY_AREA_QUANTUM_STRENGTH*4)
                        options[optIndex++]=64;
                    if(matter < left && left > PLAY_AREA_QUANTUM_STRENGTH*4)
                        options[optIndex++]=128;
                }

                const int selected = randomFloat(&randomSeed) * optIndex;

                // 1 = goes top, 2 = goes right, 4 = goes bottom, 8 = goes left, 
                // 16 = comes from top, 32 = comes from right, 64 = comes from bottom, 128 = comes from left
                areaTargetSource[id] = options[selected];

                randomSeedState[id]=randomSeed;
             })", "computeSand");


        _computer->compile(_defineMacros + R"(
            kernel void moveSand(
                const global unsigned char * __restrict__ areaTargetSource,
                global unsigned char * __restrict__ areaIn,
                global unsigned char * __restrict__ areaOut
            )
            {
                const int id=get_global_id(0);  
                const int x = id%PLAY_AREA_WIDTH;
                const int y = id/PLAY_AREA_WIDTH;


                const int topIdY = (y==0?y:y-1);
                const int topIdX = x;
                const int rightIdY = y;
                const int rightIdX = (x==PLAY_AREA_WIDTH - 1 ? x:x+1);
                const int botIdY = (y==PLAY_AREA_HEIGHT-1?y:y+1);
                const int botIdX = x;
                const int leftIdY = y;
                const int leftIdX = (x==0 ? x:x-1);
                

                const unsigned char tsCenter = areaTargetSource[id];

                const unsigned char top = areaTargetSource[topIdX + topIdY * PLAY_AREA_WIDTH];
                const unsigned char right = areaTargetSource[rightIdX + rightIdY * PLAY_AREA_WIDTH];
                const unsigned char bot = areaTargetSource[botIdX + botIdY * PLAY_AREA_WIDTH];
                const unsigned char left = areaTargetSource[leftIdX + leftIdY * PLAY_AREA_WIDTH];



                int matter = 0;
                if(tsCenter == 1)
                    matter-=PLAY_AREA_QUANTUM_STRENGTH;

                if(tsCenter == 2)
                    matter-=PLAY_AREA_QUANTUM_STRENGTH;

                if(tsCenter == 4)
                    matter-=PLAY_AREA_QUANTUM_STRENGTH;

                if(tsCenter == 8)
                    matter-=PLAY_AREA_QUANTUM_STRENGTH;

                if(tsCenter == 16)
                    matter+=PLAY_AREA_QUANTUM_STRENGTH;

                if(tsCenter == 32)
                    matter+=PLAY_AREA_QUANTUM_STRENGTH;

                if(tsCenter == 64)
                    matter+=PLAY_AREA_QUANTUM_STRENGTH;

                if(tsCenter == 128)
                    matter+=PLAY_AREA_QUANTUM_STRENGTH;

                if(topIdY != y)
                {
                    // if top cell sending 1 matter to its bottom (that is current cell)
                    if(top == 4)
                        matter+=PLAY_AREA_QUANTUM_STRENGTH;

                    if(top == 64)
                        matter-=PLAY_AREA_QUANTUM_STRENGTH;
                }

                if(rightIdX != x)
                {
                    if(right == 8)
                        matter+=PLAY_AREA_QUANTUM_STRENGTH;

                    if(right == 128)
                        matter-=PLAY_AREA_QUANTUM_STRENGTH;
                }

                if(botIdY != y)
                {
                    if(bot == 1)
                        matter+=PLAY_AREA_QUANTUM_STRENGTH;

                    if(bot == 16)
                        matter-=PLAY_AREA_QUANTUM_STRENGTH;
                }

                if(leftIdX != x)
                {
                    if(left == 2)
                        matter+=PLAY_AREA_QUANTUM_STRENGTH;

                    if(left == 32)
                        matter-=PLAY_AREA_QUANTUM_STRENGTH;
                }

                areaOut[id] = areaIn[id] + matter;

            }
        )", "moveSand");

        Reset();

    }
    
    void Reset()
    {
        
        for (int i = 0; i < _width * _height; i++)
        {
            _areaIn->access<unsigned char>(i) = 0;
            _randomSeedIn->access<unsigned int>(i) = i;
        }
        _computer->compute(*_parametersRandomInit, "initRandomSeed", 0, _totalCells, 256);
    }

    void Calc()
    {
        {
            GPGPU::Bench bench(&_frameTime);
            for(int i=0;i< _numComputePerFrame;i++)
                CalcFallingSand();            
        }
    } 

    void CalcFallingSand()
    {
        _computer->compute(*_parameterSand, "computeSand", 0, _totalCells, 256);
        _areaTargetSourceOut->copyDataToPtr(_areaTargetSourceIn->accessPtr<unsigned char>(0));
        _computer->compute(*_parameterSandMove, "moveSand", 0, _totalCells, 256);
        _areaOut->copyDataToPtr(_areaIn->accessPtr<unsigned char>(0));

    }



    void AddSandToCursorPosition(int x, int y)
    {
        for (int j = -15; j <= 15; j++)
            for (int i = -15; i <= 15; i++)
                if (x + i >= 0 && x + i < _width && y + j >= 0 && y + j < _height)
                {
                    auto id = x + i + (y + j) * _width;
                    if (_areaIn->access<unsigned char>(id) < 200)
                        _areaIn->access<unsigned char>(id)+=5;                    
                }
    }

    void Render()
    {
        size_t ti = 0;
        static cv::Mat frame(_height, _width, CV_8UC3);

        {


            GPGPU::Bench bench(&ti);
            std::vector<std::thread> thr;
            std::atomic<int> _j = 0,total=0;
            
            for (int k = 0; k < std::thread::hardware_concurrency(); k++)
            {
                thr.emplace_back([&]() {
                    int j = 0;
                    int tot = 0;
                        
                    while ((j = _j++) < frame.rows)
                    {
                        for (int i = 0; i < frame.cols; i++)
                        {
                            unsigned char matter = _areaOut->access<unsigned char>(i + j * _width);
                            frame.at<cv::Vec3b>(i + j * _width).val[0] = matter;
                            frame.at<cv::Vec3b>(i + j * _width).val[1] = matter;
                            frame.at<cv::Vec3b>(i + j * _width).val[2] = matter;
                            tot += matter;
                        }
                    }

                    total += tot;
                    });
            }
            for (auto& e : thr)
                e.join();
            cv::putText(frame, std::string("compute(")+std::to_string(_numComputePerFrame) + std::string(" steps): ") + std::to_string(_frameTime / 1000000000.0) + std::string(" seconds"), cv::Point2f(46, 76), 1, 5, cv::Scalar(50, 59, 69));
            cv::putText(frame, std::string("matter: ") + std::to_string(total.load()), cv::Point2f(46, 126), 1, 5, cv::Scalar(50, 59, 69));
            cv::imshow("AATPTPT", frame);

        }

     

    }

    void Stop()
    {
        cv::destroyWindow("AATPTPT");
    }
};