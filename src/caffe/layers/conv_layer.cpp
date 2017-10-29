#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
  // 计算卷积输出map维度[h w]
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

// 举例：假设5*5@3*3，stride=2且padding=0，则输出应该为2*2
// 假设im：
//  0   1   2   3   4
//  5   6   7   8   9
//  10  11  12  13  14
//  15  16  17  18  19
//  20  21  22  23  24
//  卷积核为：
//  1 0 1
//  1 0 1
//  1 0 1
//  经过im2col后，输入为
//  0   2   10  12
//  1   3   11  13
//  2   4   12  14
//  5   7   15  17
//  6   8   16  18
//  7   9   17  19
//  10  12  20  22
//  11  13  21  23
//  12  14  22  24
//  则输出化为矩阵相乘
//                      0   2   10  12 
//                      1   3   11  13
//                      2   4   12  14
//                      5   7   15  17
//  1 0 1 1 0 1 1 0 1 * 6   8   16  18  = 36 48  96  118
//                      7   9   17  19
//                      10  12  20  22
//                      11  13  21  23
//                      12  14  22  24
//  在经过col2im，得到：
//  36  48
//  96  118
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      // 对每个样本计算矩阵乘法
      // bottom_dim_：输入维度 c*h*w
      // top_dim_：输出维度 c*h*w
      // diff_data = weight * bottom_data
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    // 这里有两种情况需要判断
    // 权重就在这一层，马上更新 & 权重在下一层，误差还需要往下传播
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        // 权重就在这一层，马上更新，需要计算 delta yn / delta wn-1，即yn-1
        // 因此权重的导数 delta L / delta wn-1 = (delta L / delta yn) .* (delta yn / delta wn-1)
        // 写成矩阵形式，即weight_diff = top_diff * (bottom_data).T
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        // 权重在下一层，误差还需要往下传播，那么需要计算 delta yn / delta yn-1，即wn-1
        // 因此该层传递的误差 delta L / delta yn-1 = (delta L / delta yn) .* (delta yn / delta yn-1)
        // 写成矩阵的形式，即bottom_diff = (weight).T * top_diff
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
