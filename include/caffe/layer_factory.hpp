/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers can be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Layer;

template <typename Dtype>
class LayerRegistry {
 public:
  // *creator指的是函数指针，指向的函数以LayerParameter类型为参数，
  // 并返回shared_ptr<Layer<Dtype> >类型
  typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
  // map类型，每一种层类型type对应要给creator函数指针，理解为注册表
  typedef std::map<string, Creator> CreatorRegistry;

  // 新建注册表
  static CreatorRegistry& Registry() {
    // 注意static，只执行一次并放在静态区
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
  // 确保相同类型只注册一次，因为每种层对应的构造函数是一样的，只是参数不一样
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    // 判断是否第一次注册，否则输出“已经注册”信息
    CHECK_EQ(registry.count(type), 0)
        << "Layer type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a layer using a LayerParameter.
  static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param) {
    if (Caffe::root_solver()) {
      LOG(INFO) << "Creating layer " << param.name();
    }
    const string& type = param.type();
    CreatorRegistry& registry = Registry();
    // 判断创建的层是否在注册表里，找不到则输出“未知类型，已知的类型有。。。”信息
    CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
        << " (known types: " << LayerTypeListString() << ")";
    // 调用creator指针指向的函数，该函数将以param为参数
    return registry[type](param);
  }

  // 列举注册表中的层类型
  static vector<string> LayerTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> layer_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      layer_types.push_back(iter->first);
    }
    return layer_types;
  }

 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.
  LayerRegistry() {}

  // 打印注册表中的层类型，用逗号隔开
  static string LayerTypeListString() {
    vector<string> layer_types = LayerTypeList();
    string layer_types_str;
    for (vector<string>::iterator iter = layer_types.begin();
         iter != layer_types.end(); ++iter) {
      if (iter != layer_types.begin()) {
        layer_types_str += ", ";
      }
      layer_types_str += *iter;
    }
    return layer_types_str;
  }
};


template <typename Dtype>
class LayerRegisterer {
 public:
  // 构造函数，参数为层类型字符串和函数指针 
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&)) {
    // LOG(INFO) << "Registering layer type: " << type;
    LayerRegistry<Dtype>::AddCreator(type, creator);
  }
};

// 宏定义，可以多行，最后的不用分号
// 宏定义中#表示将宏的参数字符串化（LayerRegisterer第一个参数是string类型）
// ##则表示符号的连接
#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
