
#include <CL/OpenCL.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include<time.h>
#include<stdio.h>
#include<stdlib.h>

using namespace std;

const int t = 1000;


const int heightA = t;
const int widthB = t;
const int midle = t;

//const int heightB = 3;

//一、 选择OpenCL平台并创建一个上下文
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    //选择可用的平台中的第一个
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    //创建一个OpenCL上下文环境
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
        NULL, NULL, &errNum);

    return context;
}


//二、 创建设备并创建命令队列
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id* device)
{
    cl_int errNum;
    cl_device_id* devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // 获取设备缓冲区大小
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // 为设备分配缓存空间
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);

    //选取可用设备中的第一个
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

    *device = devices[0];
    delete[] devices;
    return commandQueue;
}


// 三、创建和构建程序对象
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char* srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
        (const char**)&srcStr,
        NULL, NULL);

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    return program;
}

//创建和构建程序对象
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
    int* a, int* b)
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * midle * heightA, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * widthB * midle, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(int) * widthB * heightA, NULL, NULL);
    return true;
}


// 释放OpenCL资源
void Cleanup(cl_context context, cl_command_queue commandQueue,
    cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);
}


int main(int argc, char** argv)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem memObjects[3] = { 0, 0, 0 };
    cl_int errNum;
    cl_event events[1];
    clock_t t1, t2, t3;


    const char* filename = "./a.cl";
    // 一、选择OpenCL平台并创建一个上下文
    context = CreateContext();

    // 二、 创建设备并创建命令队列
    commandQueue = CreateCommandQueue(context, &device);

    //三、创建和构建程序对象
    program = CreateProgram(context, device, filename);

    // 四、 创建OpenCL内核并分配内存空间
    kernel = clCreateKernel(program, "hello_kernel", NULL);

    //创建要处理的数据
    int* a = NULL; // 输入数组
    int* b = NULL; // 输入数组
    int* result = NULL; // 输出数组
    // 数组的大小
    const int  elementsA = heightA * midle;
    const int  elementsB = midle * widthB;
    const int  elementsC = heightA * widthB;

    // 计算内存大小
    size_t datasizeA = sizeof(float) * elementsA;
    size_t datasizeB = sizeof(float) * elementsB;
    size_t datasizeC = sizeof(float) * elementsC;
    // 分配内存空间
    a = (int*)malloc(datasizeA);
    b = (int*)malloc(datasizeB);
    result = (int*)malloc(datasizeC);

    for (int i = 0; i < heightA; i++)
    {
        for (int j = 0; j < midle; j++)
        {
            a[i * midle + j] = 2;//10.0f * ((int) rand() / (int) RAND_MAX);
        }

    }


    for (int k = 0; k < midle; k++)
    {
        for (int m = 0; m < widthB; m++)
        {
            b[k * widthB + m] = 3;//10.0f * ((int) rand() / (int) RAND_MAX);
        }

    }

    t1 = clock();  //mach_absolute_time();
    //cpu串行处理代码
    for (int l = 0; l < heightA; l++) {
        for (int n = 0; n < widthB; n++) {
            for (int q = 0; q < midle; q++) {
                result[l * widthB + n] += a[l * midle + q] * b[q * widthB + n];

            }
            //std::cout<<"r = "<<result[l*widthB+n]<<std::endl;
        }
    }
    t2 = clock(); //mach_absolute_time();

    //创建内存对象
    if (!CreateMemObjects(context, memObjects, a, b))
    {
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // 五、 设置内核数据并执行内核
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
    errNum = clSetKernelArg(kernel, 3, sizeof(int), &heightA);
    errNum = clSetKernelArg(kernel, 4, sizeof(int), &widthB);
    errNum = clSetKernelArg(kernel, 5, sizeof(int), &midle);

    size_t globalWorkSize[2];
    globalWorkSize[0] = heightA;
    globalWorkSize[1] = widthB;
    // size_t localWorkSize[2] = { 1,1 };

    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
        globalWorkSize, NULL,
        0, NULL, &events[0]);


    // 六、 读取执行结果并释放OpenCL资源
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
        0, widthB * heightA * sizeof(int), result,
        0, NULL, NULL);
    //    for(int p=0;p<20;p++){
    //        cout<<"new ="<<result[p];
    //    }
    errNum = clWaitForEvents(1, &events[0]);

    t3 = clock();  //mach_absolute_time();

    errNum = clReleaseEvent(events[0]);




    printf("cpu t = %.8f\n", ((double)t2 - (double)t1) / CLOCKS_PER_SEC);
    printf("gpu t = %.8f \n", ((double)t3 - (double)t2)/ CLOCKS_PER_SEC);

    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);

    return 0;
}