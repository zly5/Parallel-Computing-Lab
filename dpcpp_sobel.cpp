//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <chrono>
#include <cmath>
#include <iostream>
#include "CL/sycl.hpp"
#include "device_selector.hpp"

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
using namespace chrono;

// Few useful acronyms.
constexpr auto sycl_read = access::mode::read;
constexpr auto sycl_write = access::mode::write;
constexpr auto sycl_global_buffer = access::target::global_buffer;

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
__attribute__((always_inline)) static void ApplyFilter(uint8_t *src_image,
                                                       uint8_t *dst_image,
                                                       int i) {
  i *= 3;
  float temp;
  temp = (0.393f * src_image[i]) + (0.769f * src_image[i + 1]) +
         (0.189f * src_image[i + 2]);
  dst_image[i] = temp > 255 ? 255 : temp;
  temp = (0.349f * src_image[i]) + (0.686f * src_image[i + 1]) +
         (0.168f * src_image[i + 2]);
  dst_image[i + 1] = temp > 255 ? 255 : temp;
  temp = (0.272f * src_image[i]) + (0.534f * src_image[i + 1]) +
         (0.131f * src_image[i + 2]);
  dst_image[i + 2] = temp > 255 ? 255 : temp;
}

//sobel filter kernel
__attribute__((always_inline)) static void ApplySFilter(uint8_t *src_image,
                                                        uint8_t *dst_image,
                                                        int w,
                                                        int h) {
  cout<<"start filter;\n";
  int Gx = 0;
  int Gy = 0;
  float temp;
  for (int i=1;i<(h-1);i++)
  {
    for (int j=1;j<(w-1);j++)
    {  
      Gy = src_image[(i+1)*w+(j-1)]*1+src_image[(i+1)*w+(j)]*2+src_image[(i+1)*w+(j+1)]*1-(src_image[(i-1)*w+(j-1)]*1+src_image[(i-1)*w+(j)]*2+src_image[(i-1)*w+(j+1)]*1);
      Gx = src_image[(i-1)*w+(j+1)]*1+src_image[(i)*w+(j+1)]*2+src_image[(i+1)*w+(j+1)]*1-(src_image[(i-1)*w+(j-1)]*1+src_image[(i)*w+(j-1)]*2+src_image[(i+1)*w+(j-1)]*1);
      temp = (abs(Gx)+abs(Gy))/2.0f;
      dst_image[i*w+j] = temp>200?255:temp;
    }
  }
    
}
//并行 ss
__attribute__((always_inline)) static void ApplySSFilter(uint8_t *src_image,
                                                         uint8_t *dst_image,
                                                         int i,
                                                         int w,
                                                         int h
                                                         ) {

  int Gx = 0;
  int Gy = 0;
  float temp;
  if (i>w)
  {
      Gy = src_image[i+w-1]*1+src_image[i+w]*2+src_image[i+w+1]*1-(src_image[i-w-1]*1+src_image[i-w]*2+src_image[i-w+1]*1);
      Gx = src_image[i-w+1]*1+src_image[i+1]*2+src_image[i+w+1]*1-(src_image[i-w-1]*1+src_image[i-1]*2+src_image[i+w-1]*1);
      temp = (abs(Gx)+abs(Gy))/2.0f;
      dst_image[i] = temp>200?255:temp;
  }
    
}


int main(int argc, char **argv) {
    
  // loading the input image
  int img_width, img_height, channels;
  uint8_t *image = stbi_load("1.jpg", &img_width, &img_height, &channels, 0);
  if (image == NULL) {
    cout << "Error in loading the image\n";
    exit(1);
  }
  cout << "Loaded image with a width of " << img_width << ", a height of "
       << img_height << " and " << channels << " channels"<<"\n";

  //像素个数，图像尺寸
  size_t num_pixels = img_width * img_height;
  // size_t img_size = img_width * img_height * channels;
  size_t img_size = img_width * img_height;

  // allocating memory for output images
  uint8_t *image_gray = new uint8_t[img_size];
  uint8_t *image_ref = new uint8_t[img_size];
  uint8_t *image_exp1 = new uint8_t[img_size];

  memset(image_gray, 0, img_size * sizeof(uint8_t));
  memset(image_ref, 0, img_size * sizeof(uint8_t));
  memset(image_exp1, 0, img_size * sizeof(uint8_t));
  
  //gray灰度化
  for (int p=0;p<img_width*img_height;p++)
  { 
      image_gray[p] = (image[3*p]+image[3*p+1]+image[3*p+2])/3.0f;
  }

  // Create a device selector which rates available devices in the preferred
  // order for the runtime to select the highest rated device
  // Note: This is only to illustrate the usage of a custom device selector.
  // default_selector can be used if no customization is required.
  MyDeviceSelector sel;

  // Using these events to time command group execution
  event e1;

  // Wrap main SYCL API calls into a try/catch to diagnose potential errors
  try {
    // Create a command queue using the device selector and request profiling
    auto prop_list = property_list{property::queue::enable_profiling()};
    queue q(sel, dpc_common::exception_handler, prop_list);

    // See what device was actually selected for this queue.
    cout << "Running on " << q.get_device().get_info<info::device::name>()
        << "\n";

    // Create SYCL buffer representing source data .
    // By default, this buffers will be created with global_buffer access
    // target, which means the buffer "projection" to the device (actual
    // device memory chunk allocated or mapped on the device to reflect
    // buffer's data) will belong to the SYCL global address space - this
    // is what host data usually maps to. Other address spaces are:
    // private, local and constant.
    // Notes:
    // - access type (read/write) is not specified when creating a buffer -
    //   this is done when actual accessor is created
    // - there can be multiple accessors to the same buffer in multiple command
    //   groups
    // - 'image' pointer was passed to the constructor, so this host memory
    //   will be used for "host projection", no allocation will happen on host
    buffer image_buf(image_gray, range(img_size));

    // This is the output buffer device writes to
    buffer image_buf_exp1(image_exp1, range(img_size));
    cout << "Submitting lambda kernel...\n";

    // Submit a command group for execution. Returns immediately, not waiting
    // for command group completion.
    e1 = q.submit([&](auto &h) {
      // This lambda defines a "command group" - a set of commands for the
      // device sharing some state and executed in-order - i.e. creation of
      // accessors may lead to on-device memory allocation, only after that
      // the kernel will be enqueued.
      // A command group can contain at most one parallel_for, single_task or
      // parallel_for_workgroup construct.
      accessor image_acc(image_buf, h, read_only);
      accessor image_exp_acc(image_buf_exp1, h, write_only);

      // This is the simplest form cl::sycl::handler::parallel_for -
      // - it specifies "flat" 1D ND range(num_pixels), runtime will select
      //   local size
      // - kernel lambda accepts single cl::sycl::id argument, which has very
      //   limited API; see the spec for more complex forms
      // the lambda parameter of the parallel_for is the kernel, which
      // actually executes on device


      h.parallel_for(range<1>(num_pixels), [=](auto i) {
        ApplySSFilter(image_acc.get_pointer(), image_exp_acc.get_pointer(), i,img_width,img_height);
      });
    });
    q.wait_and_throw();

  }catch (sycl::exception e) {
    // This catches only synchronous exceptions that happened in current thread
    // during execution. The asynchronous exceptions caused by execution of the
    // command group are caught by the asynchronous exception handler
    // registered. Synchronous exceptions are usually those which are thrown
    // from the SYCL runtime code, such as on invalid constructor arguments. An
    // example of asynchronous exceptions is error occurred during execution of
    // a kernel. Make sure sycl::exception is caught, not std::exception.
    cout << "SYCL exception caught: " << e.what() << "\n";
    return 1;
  }

  // report execution times:
  ReportTime("Lambda kernel time: ", e1);

  // get reference result
  // 计时开始
  auto start = system_clock::now();
  ApplySFilter(image_gray, image_ref, img_width, img_height);
  auto end = system_clock::now();
  auto duration = duration_cast<milliseconds>(end - start);
  cout << "Serial time: " << double(duration.count()) << " milliseconds\n";

  stbi_write_png("sobel.png", img_width, img_height, 1, image_ref,
                  img_width*1);
  stbi_write_png("sobel_lambda.png", img_width, img_height, 1,
                 image_exp1, img_width * 1);

  stbi_image_free(image);
  delete[] image_ref;

  cout << "Successfully applied to image! \n";
  return 0;
}
