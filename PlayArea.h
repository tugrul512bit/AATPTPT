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
    std::shared_ptr<GPGPU::HostParameter> _areaState;
    std::shared_ptr<GPGPU::HostParameter> _areaState2;
    std::shared_ptr<GPGPU::HostParameter> _areaOut;
    std::shared_ptr<GPGPU::HostParameter> _areaTargetSource;
    std::shared_ptr<GPGPU::HostParameter> _areaTargetSource2;

    std::shared_ptr<GPGPU::HostParameter> _areaPressureIn;
    std::shared_ptr<GPGPU::HostParameter> _areaPressureOut;

    std::shared_ptr<GPGPU::HostParameter> _randomSeedIn;
    std::shared_ptr<GPGPU::HostParameter> _randomSeedState;

    std::shared_ptr<GPGPU::HostParameter> _parametersRandomInit;
    std::shared_ptr<GPGPU::HostParameter> _parameterAreaInput;
    std::shared_ptr<GPGPU::HostParameter> _parameterAreaOutput;
    std::shared_ptr<GPGPU::HostParameter> _parameterAreaState;
    std::shared_ptr<GPGPU::HostParameter> _parameterGuess1;
    std::shared_ptr<GPGPU::HostParameter> _parameterGuess2;
    std::shared_ptr<GPGPU::HostParameter> _parameterSandMove;


    std::string _defineMacros;
    size_t _frameTime;
    int _numComputePerFrame;
    int _quantumStrength;

    std::vector<GPGPU::HostParameter> _listPrm;
    std::vector<std::string> _listKernel;
public:
    // width and height must be multiple of 16
    PlayArea(int & width, int & height, int maximumGPUsToUse = 10, int indexGPU=0,  int numStepsPerFrame=10, int quantumStrength=1)
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
        _computer = std::make_shared<GPGPU::Computer>(GPGPU::Computer::DEVICE_GPUS, indexGPU,1,false, maximumGPUsToUse); // allocate all devices for computations

        // broadcast type input (duplicated on all gpus from ram)
        // load-balanced output                                                
        _areaIn = std::make_shared<GPGPU::HostParameter>(_computer->createArrayInput<unsigned char>("areaIn", _totalCells));
        _areaState = std::make_shared<GPGPU::HostParameter>(_computer->createArrayState<unsigned char>("areaState", _totalCells));
        _areaState2 = std::make_shared<GPGPU::HostParameter>(_computer->createArrayState<unsigned char>("areaState2", _totalCells));
        _areaOut = std::make_shared<GPGPU::HostParameter>(_computer->createArrayOutputAll<unsigned char>("areaOut", _totalCells));
        _areaTargetSource = std::make_shared<GPGPU::HostParameter>(_computer->createArrayState<unsigned char>("areaTargetSource", _totalCells));
        _areaTargetSource2 = std::make_shared<GPGPU::HostParameter>(_computer->createArrayState<unsigned char>("areaTargetSource2", _totalCells));
        
     
        _randomSeedIn = std::make_shared<GPGPU::HostParameter>(_computer->createArrayInput<unsigned int>("randomSeedIn", _totalCells));
        _randomSeedState = std::make_shared<GPGPU::HostParameter>(_computer->createArrayState<unsigned int>("randomSeedState", _totalCells));

        _areaPressureIn = std::make_shared<GPGPU::HostParameter>(_computer->createArrayState<unsigned char>("areaPressureIn", _totalCells));
        _areaPressureOut = std::make_shared<GPGPU::HostParameter>(_computer->createArrayState<unsigned char>("areaPressureOut", _totalCells));


        _parametersRandomInit = std::make_shared<GPGPU::HostParameter>(
            _randomSeedIn->next(*_randomSeedState)
        );

        _parameterGuess1 = std::make_shared<GPGPU::HostParameter>(
            _areaState->next(*_randomSeedState).next(*_areaTargetSource).next(*_areaPressureIn)
        );
        _parameterGuess2 = std::make_shared<GPGPU::HostParameter>(
            _areaTargetSource->next(*_areaTargetSource2).next(*_randomSeedState)
        );

        _parameterSandMove = std::make_shared<GPGPU::HostParameter>(
            _areaTargetSource->next(*_areaTargetSource2).next(*_areaState).next(*_areaState2)
        );
      
        _parameterAreaInput = std::make_shared<GPGPU::HostParameter>(
            _areaIn->next(*_areaState)
        );
        _parameterAreaOutput = std::make_shared<GPGPU::HostParameter>(
            _areaOut->next(*_areaState2)
        );
        _parameterAreaState = std::make_shared<GPGPU::HostParameter>(
            _areaState->next(*_areaState2)
        );


        _defineMacros = std::string("#define PLAY_AREA_WIDTH ") + std::to_string(_width) + R"(
        )";
        _defineMacros += std::string("#define PLAY_AREA_HEIGHT ") + std::to_string(_height) + R"(
        )";
        _defineMacros += std::string("#define PLAY_AREA_TOTAL_CELLS ") + std::to_string(_totalCells) + R"(
        )";

        _defineMacros += std::string("#define PLAY_AREA_QUANTUM_STRENGTH ") + std::to_string(_quantumStrength) + R"(
        )";

        _defineMacros += std::string("#define PLAY_AREA_TEST_UP_PROB ") + std::to_string(1) + R"(
        )";
        _defineMacros += std::string("#define PLAY_AREA_TEST_RIGHT_PROB ") + std::to_string(14) + R"(
        )";
        _defineMacros += std::string("#define PLAY_AREA_TEST_BOT_PROB ") + std::to_string(15) + R"(
        )";
        _defineMacros += std::string("#define PLAY_AREA_TEST_LEFT_PROB ") + std::to_string(14) + R"(
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
            kernel void areaBufInput(
                const global unsigned char * __restrict__ areaIn,
                global unsigned char * __restrict__ areaState
            )
            {
                const int id=get_global_id(0);  
                areaState[id]=areaIn[id];
            }
        )", "areaBufInput");

        _computer->compile(_defineMacros + R"(
            kernel void areaBufOutput(
                global unsigned char * __restrict__ areaOut,
                const global unsigned char * __restrict__ areaState2
            )
            {
                const int id=get_global_id(0);  
                areaOut[id]=areaState2[id];
            }
        )", "areaBufOutput");

        _computer->compile(_defineMacros + R"(
            kernel void areaBufState(
                global unsigned char * __restrict__ areaState,
                const global unsigned char * __restrict__ areaState2
            )
            {
                const int id=get_global_id(0);  
                areaState[id]=areaState2[id];
            }
        )", "areaBufState");
        
        _computer->compile(_defineMacros + R"(
            

            // todo: weighted probabilities
            // a side with empty cell will have more probability to be filled
            kernel void guessParticleTarget(
                const global unsigned char * __restrict__ areaState, 
                global unsigned int * __restrict__ randomSeedState,
                global unsigned char * __restrict__ areaTargetSource,
                global unsigned char * __restrict__ areaPressureIn
            ) 
            { 
                const int id=get_global_id(0); 
                const int x = id % PLAY_AREA_WIDTH;
                const int y = id / PLAY_AREA_WIDTH;
                unsigned int randomSeed = randomSeedState[id];
                
                const int matter = areaState[id];

                const int topIdY = (y==0?y:y-1);
                const int topIdX = x;
                const int rightIdY = y;
                const int rightIdX = (x==PLAY_AREA_WIDTH - 1 ? x:x+1);
                const int botIdY = (y==PLAY_AREA_HEIGHT-1?y:y+1);
                const int botIdX = x;
                const int leftIdY = y;
                const int leftIdX = (x==0 ? x:x-1);

                const int top = areaState[topIdX + topIdY * PLAY_AREA_WIDTH];
                const int right = areaState[rightIdX + rightIdY * PLAY_AREA_WIDTH];
                const int bot = areaState[botIdX + botIdY * PLAY_AREA_WIDTH];
                const int left = areaState[leftIdX + leftIdY * PLAY_AREA_WIDTH];

                // quantum-mechenical pressure solver
                // a particle can go only 4 places
                // having more particles on target pos lowers the probability for there
                // then particle picks a pos to go, marks its id (does not move yet because parallel update will be done)
                // todo: weighted probabilities
           
              
                int totProb = 0;
                int tot = 0;
                // check where can 1 matter go

             
                if(matter == 1 && top == 0 && topIdY != y)
                {
                    totProb+=PLAY_AREA_TEST_UP_PROB;
                }
                if(matter == 1 && right == 0 && rightIdX != x)
                {
                    totProb+=PLAY_AREA_TEST_RIGHT_PROB;
                }
                if(matter == 1 && bot == 0 && botIdY != y)
                {
                    totProb+=PLAY_AREA_TEST_BOT_PROB;
                }
                if(matter == 1 && left == 0 && leftIdX != x)
                {
                    totProb+=PLAY_AREA_TEST_LEFT_PROB;
                }                

                

                const int selected = floor(randomFloat(&randomSeed) * totProb);


                // check where can 1 matter go               
                if(matter == 1 && top == 0 && topIdY != y)
                {
                    tot+=PLAY_AREA_TEST_UP_PROB;
                    if(selected<tot)
                    {
                        areaTargetSource[id]=1;
                        randomSeedState[id]=randomSeed;
                        return;
                    }
                }

                if(matter == 1 && right == 0 && rightIdX != x)
                {
                    tot+=PLAY_AREA_TEST_RIGHT_PROB;
                    if(selected<tot)
                    {
                        areaTargetSource[id]=2;
                        randomSeedState[id]=randomSeed;
                        return;
                    }
                }

                if(matter == 1 && bot == 0 && botIdY != y)
                {
                    tot+=PLAY_AREA_TEST_BOT_PROB;
                    if(selected<tot)
                    {
                        areaTargetSource[id]=4;
                        randomSeedState[id]=randomSeed;
                        return;
                    }
                }

                if(matter == 1 && left == 0 && leftIdX != x)
                {
                    tot+=PLAY_AREA_TEST_LEFT_PROB;
                    if(selected<tot)
                    {
                        areaTargetSource[id]=8;
                        randomSeedState[id]=randomSeed;
                        return;
                    }
                }

              

                // 1 = goes top, 2 = goes right, 4 = goes bottom, 8 = goes left, 
                         
                areaTargetSource[id]=0;
            })", "guessParticleTarget");


        // picks 1 of multiple cells that want to send matter
        _computer->compile(_defineMacros + R"(
            kernel void pickOneTargetGuess(
                const global unsigned char * __restrict__ areaTargetSource,
                global unsigned char * __restrict__ areaTargetSource2,
                global unsigned int * __restrict__ randomSeedState
            )
            {
                const int id=get_global_id(0);  
                const int x = id%PLAY_AREA_WIDTH;
                const int y = id/PLAY_AREA_WIDTH;
                unsigned int randomSeed = randomSeedState[id];

                const int topIdY = (y==0?y:y-1);
                const int topIdX = x;
                const int rightIdY = y;
                const int rightIdX = (x==PLAY_AREA_WIDTH - 1 ? x:x+1);
                const int botIdY = (y==PLAY_AREA_HEIGHT-1?y:y+1);
                const int botIdX = x;
                const int leftIdY = y;
                const int leftIdX = (x==0 ? x:x-1);
                

                const unsigned char tsCenter = areaTargetSource[id];

                const int top = areaTargetSource[topIdX + topIdY * PLAY_AREA_WIDTH];
                const int right = areaTargetSource[rightIdX + rightIdY * PLAY_AREA_WIDTH];
                const int bot = areaTargetSource[botIdX + botIdY * PLAY_AREA_WIDTH];
                const int left = areaTargetSource[leftIdX + leftIdY * PLAY_AREA_WIDTH];

                // picks one of neighbors that send matter to this cell
                // by probability
                // gas: all neighbors
                // liquid: left right bottom bottom-left bottom-right
                // solid: bottom-left bottom bottom-right
                int totProb = 0;
                int tot = 0;

                // more probability for top to down because of gravity
                if(topIdY != y)
                {
                    if(top == 4)
                        totProb +=PLAY_AREA_TEST_BOT_PROB;
                }

                if(rightIdX != x)
                {
                    if(right == 8)
                        totProb +=PLAY_AREA_TEST_LEFT_PROB;
                }

                if(botIdY != y)
                {
                    if(bot == 1)
                        totProb +=PLAY_AREA_TEST_UP_PROB;
                }

                if(leftIdX != x)
                {
                    if(left == 2)
                        totProb +=PLAY_AREA_TEST_RIGHT_PROB;
                }

                const int selected = floor(randomFloat(&randomSeed) * totProb);
                randomSeedState[id]=randomSeed;

                if(topIdY != y)
                {
                    if(top == 4)
                        tot +=PLAY_AREA_TEST_BOT_PROB;

                    if(selected<tot)
                    {
                        areaTargetSource2[id]=1;
                        randomSeedState[id]=randomSeed;
                        return;
                    }
                }

                if(rightIdX != x)
                {
                    if(right == 8)
                        tot +=PLAY_AREA_TEST_LEFT_PROB;

                    if(selected<tot)
                    {
                        areaTargetSource2[id]=2;
                        randomSeedState[id]=randomSeed;
                        return;
                    }
                }

                if(botIdY != y)
                {
                    if(bot == 1)
                        tot +=PLAY_AREA_TEST_UP_PROB;

                    if(selected<tot)
                    {
                        areaTargetSource2[id]=4;
                        randomSeedState[id]=randomSeed;
                        return;
                    }
                }

                if(leftIdX != x)
                {
                    if(left == 2)
                        tot +=PLAY_AREA_TEST_RIGHT_PROB;

                    if(selected<tot)
                    {
                        areaTargetSource2[id]=8;
                        randomSeedState[id]=randomSeed;
                        return;
                    }
                }             
                areaTargetSource2[id]=0;
            }
        )", "pickOneTargetGuess");




        // picks 1 of multiple cells that want to send matter
        _computer->compile(_defineMacros + R"(
            kernel void moveSand(
                const global unsigned char * __restrict__ areaTargetSource,
                const global unsigned char * __restrict__ areaTargetSource2,
                const global unsigned char * __restrict__ areaState,
                global unsigned char * __restrict__ areaState2
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
                

         
                const int center = areaState[id];
                // if empty, check the accepted movement data
                if(center == 0)
                {                    
                    const int tsCenter = areaTargetSource2[id];

                    const int top = areaTargetSource[topIdX + topIdY * PLAY_AREA_WIDTH];
                    const int right = areaTargetSource[rightIdX + rightIdY * PLAY_AREA_WIDTH];
                    const int bot = areaTargetSource[botIdX + botIdY * PLAY_AREA_WIDTH];
                    const int left = areaTargetSource[leftIdX + leftIdY * PLAY_AREA_WIDTH];

                    // if top cell wants to send down 1 matter and current cell want to receive 1 matter, it is ok
                    if(top == 4 && tsCenter == 1 && topIdY != y)
                    {
                        areaState2[id] = 1;
                        return;
                    }

                    // if right cell sends to left, current cell receives from right
                    if(right == 8 &&  tsCenter == 2 && rightIdX != x)
                    {
                        areaState2[id] = 1;
                        return;
                    }

                    // if bottom cell sends up, current cell receives
                    if(bot == 1 &&  tsCenter == 4 && botIdY != y)
                    {
                        areaState2[id] = 1;
                        return;
                    }

                    // if bottom cell sends up, current cell receives
                    if(left == 2 &&  tsCenter == 8 && leftIdX != x)
                    {
                        areaState2[id] = 1;
                        return;
                    }
                }
                else    // if not empty, check if can send (areaTargetSourceIn2 is for knowing who takes)
                {
                    const int tsCenter = areaTargetSource[id];

                    const int top = areaTargetSource2[topIdX + topIdY * PLAY_AREA_WIDTH];
                    const int right = areaTargetSource2[rightIdX + rightIdY * PLAY_AREA_WIDTH];
                    const int bot = areaTargetSource2[botIdX + botIdY * PLAY_AREA_WIDTH];
                    const int left = areaTargetSource2[leftIdX + leftIdY * PLAY_AREA_WIDTH];

                    // if top cell wants to receive
                    if(top == 4 && tsCenter == 1 && topIdY != y)
                    {
                        areaState2[id] = 0;
                        return;
                    }

                    // if right cell receives
                    if(right == 8 && tsCenter == 2 && rightIdX != x)
                    {
                        areaState2[id] = 0;
                        return;
                    }

                    // if bottom cell receives
                    if(bot == 1 && tsCenter == 4 && botIdY != y)
                    {
                        areaState2[id] = 0;
                        return;
                    }

                    // if left cell receives
                    if(left == 2 && tsCenter == 8 && leftIdX != x)
                    {
                        areaState2[id] = 0;
                        return;
                    }    
                }
                areaState2[id]=center;
            }
        )", "moveSand");
        Reset();
        PrepareGpuParameterList();

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
            CalcFallingSand();            
        }
    } 


    void PrepareGpuParameterList()
    {
        for (int i = 0; i < _numComputePerFrame; i++)
        {
            _listPrm.push_back(*_parameterGuess1);
            _listPrm.push_back(*_parameterGuess2);
            _listPrm.push_back(*_parameterSandMove);
            _listPrm.push_back(*_parameterAreaState);
            _listKernel.push_back("guessParticleTarget");
            _listKernel.push_back("pickOneTargetGuess");
            _listKernel.push_back("moveSand");
            _listKernel.push_back("areaBufState");
        }
    }
    void CalcFallingSand()
    {
        _computer->compute(*_parameterAreaInput, "areaBufInput", 0, _totalCells, 256);

         // runs many repeatations of a kernel sequence
         _computer->computeMultiple(_listPrm, _listKernel, 0, _totalCells, 256);
        

        _computer->compute(*_parameterAreaOutput, "areaBufOutput", 0, _totalCells, 256);
        _areaIn->copyDataFromPtr(_areaOut->accessPtr<unsigned char>(0));
    }



    void AddSandToCursorPosition(int x, int y)
    {
        for (int j = -15; j <= 15; j++)
            for (int i = -15; i <= 15; i++)
                if (x + i >= 0 && x + i < _width && y + j >= 0 && y + j < _height)
                {
                    auto id = x + i + (y + j) * _width;
                    _areaIn->access<unsigned char>(id)=1;                    
                }
    }

    void RemoveSandFromCursorPosition(int x, int y)
    {
        for (int j = -15; j <= 15; j++)
            for (int i = -15; i <= 15; i++)
                if (x + i >= 0 && x + i < _width && y + j >= 0 && y + j < _height)
                {
                    auto id = x + i + (y + j) * _width;
                   _areaIn->access<unsigned char>(id) = 0;
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
                            frame.at<cv::Vec3b>(i + j * _width).val[0] = 0;
                            frame.at<cv::Vec3b>(i + j * _width).val[1] = matter*200;
                            frame.at<cv::Vec3b>(i + j * _width).val[2] = 0;
                            tot += matter;
                        }
                    }

                    total += tot;
                    });
            }
            for (auto& e : thr)
                e.join();
            cv::putText(frame, std::string("compute(")+std::to_string(_numComputePerFrame) + std::string(" steps): ") + std::to_string(_frameTime / 1000000000.0) + std::string(" seconds"), cv::Point2f(46, 76), 1, 4, cv::Scalar(50, 59, 69));
            cv::putText(frame, std::string("steps per second: ") + std::to_string(_numComputePerFrame/(_frameTime / 1000000000.0)), cv::Point2f(46, 126), 1, 4, cv::Scalar(50, 59, 69));
            cv::putText(frame, std::string("matter: ") + std::to_string(total.load()), cv::Point2f(46, 176), 1, 4, cv::Scalar(50, 59, 69));
            cv::imshow("AATPTPT", frame);

        }

     

    }

    void Stop()
    {
        cv::destroyWindow("AATPTPT");
    }
};