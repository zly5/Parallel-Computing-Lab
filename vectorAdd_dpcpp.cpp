#include <CL/sycl.hpp>
using namespace sycl;
static const size_t numElements = 50000;
void work(queue &q) {
  std::cout << "Device : "
            << q.get_device().get_info<info::device::name>()
            << std::endl;
  float vector1[numElements] , vector2[numElements] , vector3[numElements];
  auto R = range(numElements);
   for (int i = 0; i < numElements; ++i) {
        vector1[i] = rand()/(float)RAND_MAX;
        vector2[i] = rand()/(float)RAND_MAX;
    }
//2.创建vector1、vector2、vector3向量的SYCL缓冲区；
  buffer vector1_buffer(vector1,R);
  buffer vector2_buffer(vector2,R);
  buffer vector3_buffer(vector3,R);

//3.向Device提交工作（定义了访问缓冲区内存的accessor；）
  q.submit([&](handler &h) {
    accessor v1_accessor (vector1_buffer,h,read_only);
    accessor v2_accessor (vector2_buffer,h,read_only);
    accessor v3_accessor (vector3_buffer,h);
   //4. 调用oneAPI的核函数在Device上完成指定的运算；
        h.parallel_for (range<1>(numElements), [=](id<1> index) {
  //核函数部分，若单独写一个函数，直接使用函数名（参数表）调用即可
      if (index < numElements)
        v3_accessor [index] = v1_accessor [index] + v2_accessor [index];
    });
  }).wait(); //排队等待
   // 5. 将SYCL缓冲区的数据读到Host端，检查误差 
    host_accessor h_c(vector3_buffer,read_only);
    for (int i = 0; i < numElements; ++i) {
     if (fabs(vector1[0] + vector2[0] - vector3[0] ) > 1e-8 ) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
}
int main() {  
  try {
    queue q;
    work(q);
  } catch (exception e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    std::terminate();
  } catch (...) {
    std::cerr << "Unknown exception" << std::endl;
    std::terminate();
  }
}