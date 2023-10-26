%%writefile lab/gpu_sample.cpp
#include <chrono>
#include <cmath>
#include <iostream>
#include "CL/sycl.hpp"
//#include "device_selector.hpp"
// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

// stb/*.h files can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/stb/*.h
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

using namespace std;
using namespace sycl;

static void ReportTime(const string &msg, event e) {
  cl_ulong time_start =
      e.get_profiling_info<info::event_profiling::command_start>();

  cl_ulong time_end =
      e.get_profiling_info<info::event_profiling::command_end>();

  double elapsed = (time_end - time_start) / 1e6;
  cout << msg << elapsed << " milliseconds\n";
}


// SYCL does not need any special mark-up for functions which are called from
// SYCL kernel and defined in the same compilation unit. SYCL compiler must be
// able to find the full call graph automatically.
// always_inline as calls are expensive on Gen GPU.
// Notes:
// - coeffs can be declared outside of the function, but still must be constant
// - SYCL compiler will automatically deduce the address space for the two
//   pointers; sycl::multi_ptr specialization for particular address space
//   can used for more control
__attribute__((always_inline)) static void ApplyFilter(uint8_t *I,
                                                       uint8_t *T,
                                                       float *result,
                                                       int i,
                                                       int j,
                                                       int Iw,
                                                       int Ih, 
                                                       int Tw,
                                                       int Th)  {
    
  if (i >= Ih - Th + 1 || j >= Iw - Tw + 1) {
      return;
  }
  float sum = 0.0;
  for (int k = 0; k < Th; k++) {
    for (int s = 0; s < Tw; s++) {
        float diff = I[(i + k) * Iw + j + s] - T[k * Tw + s];
        sum += diff * diff;
    }
  }
  result[i * Iw + j] = sum;  
}


int main(int argc, char **argv) {
  // loading the src image
  int src_img_width, src_img_height, src_channels;
  // 使用灰度图像
  // 加载图片 源图片
  uint8_t *src_image = stbi_load("./tmp_src_img.jpg", &src_img_width, &src_img_height, &src_channels, 1);
  if (src_image == NULL) {
    cout << "Error in loading the image\n";
    exit(1);
  }
  cout << "Loaded src image with a width of " << src_img_width << ", a height of "
       << src_img_height << " and " << src_channels << " channels\n";
  
  // loading the template image
  int template_img_width, template_img_height, template_channels;
  // 加载图片 模板图片
  uint8_t *template_image = stbi_load("./tmp_template_img.jpg", &template_img_width, &template_img_height, &template_channels, 1);
  if (template_image == NULL) {
    cout << "Error in loading the image\n";
    exit(1);
  }
  cout << "Loaded template image with a width of " << template_img_width << ", a height of "
       << template_img_height << " and " << template_channels << " channels\n";

  if (src_img_width < template_img_width || src_img_height < template_img_height) {
    cout << "Error: The template is larger than the picture\n";
    exit(1);
  }
  
  
  // 分配的结果内存
  size_t num_counts = src_img_height * src_img_width;
  size_t src_size = src_img_height * src_img_width;
  size_t template_size = template_img_width * template_img_height;
  // 分配输出图像的内存
  // allocating memory for output images
  float *result = new float[num_counts];
  // 初始化
  // memset(image_ref, 0, num_counts * sizeof(float));

  // Create a device selector which rates available devices in the preferred
  // order for the runtime to select the highest rated device
  // Note: This is only to illustrate the usage of a custom device selector.
  // default_selector can be used if no customization is required.
  // 选择合适的设备
  
    //device_selector sel;

  // Using these events to time command group execution
  event e1, e2;
  
  // Wrap main SYCL API calls into a try/catch to diagnose potential errors
  try {
    // Create a command queue using the device selector and request profiling
    // 选择最适合的设备
    auto prop_list = property_list{property::queue::enable_profiling()};
    queue q(default_selector{}, dpc_common::exception_handler, prop_list);

    // See what device was actually selected for this queue.
    cout << "Running on " << q.get_device().get_info<info::device::name>()
         << "\n";
    // 源图像buffer
    buffer src_image_buf(src_image, range(src_size));
    // 模板图像buffer
    // This is the output buffer device writes to
    buffer template_image_buf(template_image, range(template_size));
    // 结果的buffer
    buffer result_buf(result, range(num_counts));
    cout << "Submitting lambda kernel...\n";

    // Submit a command group for execution. Returns immediately, not waiting
    // for command group completion.
    // 得到输出的比较之后的结果
    e1 = q.submit([&](auto &h) {
      accessor src_image_acc(src_image_buf, h, read_only);
      accessor template_image_acc(template_image_buf, h, read_only);
      accessor result_acc(result_buf, h, write_only);
      // 使用二维线程数
      h.parallel_for(range<2>{(size_t)src_img_height, (size_t)src_img_width}, [=](id<2> index) {
		// 内核程序执行
        ApplyFilter(src_image_acc.get_pointer(), template_image_acc.get_pointer(), result_acc.get_pointer(), index[0], index[1], src_img_width, src_img_height, template_img_width, template_img_height);
      });
    });
    q.wait_and_throw();  
  } catch (sycl::exception e) {
    cout << "SYCL exception caught: " << e.what() << "\n";
    return 1;
  }

  // report execution times:
  ReportTime("Lambda kernel time: ", e1);
  // cout << result[0] << " " << result[1];
    
  // 得到匹配位置的最小值
  int x,y;
  float minresult = result[0];
  for (int i = 0; i < src_img_height - template_img_height + 1; i++) {
    for (int j = 0; j < src_img_width - template_img_width + 1; j++) {
        if (minresult > result[i * src_img_width + j]) {
            y = i;
            x = j;
            minresult = result[i * src_img_width + j];
        }
    }
  }
  
  int x1 = x;
  int x2 = x + template_img_width - 1;
  int y1 = y;
  int y2 = y + template_img_height - 1;
    
  cout << x1 << "  " << x2 << "  " << y1 << "  " << y2 << "  ";
  
  // 对图片进行保存
    
  // 先标记两条横线
  for (int i = x1; i <= x2; i++) {
      src_image[y1 * src_img_width + i] = 0;
      src_image[y2 * src_img_width + i] = 0;
  }
  for (int i = y1 + 1; i < y2; i++) {
      src_image[i * src_img_width + x1] = 0;
      src_image[i * src_img_width + x2] = 0;
  }
    
  stbi_write_png("sepia_ref.png", src_img_width, src_img_height, src_channels, src_image, src_img_width * src_channels);
  return 0;
}
