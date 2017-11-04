https://github.com/BVLC/caffe/wiki/Development#forward-only-layers

Developing new layers:

1. Add a class declaration for your layer to include/caffe/layers/your_layer.hpp.
1) Include an inline implementation of type overriding the method virtual inline const char* type() const { return "YourLayerName"; } replacing YourLayerName with your layer's name.
2) Implement the {*}Blobs() methods to specify blob number requirements; see /caffe/include/caffe/layers.hpp to enforce strict top and bottom Blob counts using the inline {*}Blobs() methods.
3) Omit the *_gpu declarations if you'll only be implementing CPU code.

2. Implement your layer in src/caffe/layers/your_layer.cpp.
1) (optional) LayerSetUp for one-time initialization: reading parameters, fixed-size allocations, etc.
2) Reshape for computing the sizes of top blobs, allocating buffers, and any other work that depends on the shapes of bottom blobs
3) Forward_cpu for the function your layer computes
4) Backward_cpu for its gradient (Optional -- a layer can be forward-only)

3. (Optional) Implement the GPU versions Forward_gpu and Backward_gpu in layers/your_layer.cu.

4. If needed, declare parameters in proto/caffe.proto, using (and then incrementing) the "next available layer-specific ID" declared in a comment above message LayerParameter

5. Instantiate and register your layer in your cpp file with the macro provided in layer_factory.hpp. Assuming that you have a new layer MyAwesomeLayer, you can achieve it with the following command:
INSTANTIATE_CLASS(MyAwesomeLayer);
REGISTER_LAYER_CLASS(MyAwesome);

6. Note that you should put the registration code in your own cpp file, so your implementation of a layer is self-contained.

7. Optionally, you can also register a Creator if your layer has multiple engines. For an example on how to define a creator function and register it, see GetConvolutionLayer in caffe/layer_factory.cpp.

8. Write tests in test/test_your_layer.cpp. Use test/test_gradient_check_util.hpp to check that your Forward and Backward implementations are in numerical agreement.

Forward-Only Layers:

If you want to write a layer that you will only ever include in a test net, you do not have to code the backward pass. For example, you might want a layer that measures performance metrics at test time that haven't already been implemented. Doing this is very simple. You can write an inline implementation of Backward_cpu (or Backward_gpu) together with the definition of your layer in include/caffe/your_layer.hpp that looks like:

virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

The NOT_IMPLEMENTED macro (defined in common.hpp) throws an error log saying "Not implemented yet". For examples, look at the accuracy layer (accuracy_layer.hpp) and threshold layer (threshold_layer.hpp) definitions.