#ifndef _HAILOPOST_HPP_
#define _HAILOPOST_HPP_

#include <cstring>

#include <engine_interface.hpp>

#include "hailo_post/double_buffer.hpp"
#include "hailo_post/hailo_objects.hpp"
#include "hailo_post/yolo_hailortpp.hpp"

using namespace hailort;


#ifndef _DETRESULT_
#define _DETRESULT_
typedef struct{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    float id;
} DetResult;
#endif //_DETRESULT_

template <typename T>
class FeatureData {
public:
    FeatureData(uint32_t buffers_size, float32_t qp_zp, float32_t qp_scale, uint32_t width, hailo_vstream_info_t vstream_info) :
    m_buffers(buffers_size), m_qp_zp(qp_zp), m_qp_scale(qp_scale), m_width(width), m_vstream_info(vstream_info)
    {}
    static bool sort_tensors_by_size (std::shared_ptr<FeatureData> i, std::shared_ptr<FeatureData> j) { return i->m_width < j->m_width; };

    DoubleBuffer<T> m_buffers;
    float32_t m_qp_zp;
    float32_t m_qp_scale;
    uint32_t m_width;
    hailo_vstream_info_t m_vstream_info;
};

template <typename T>
static hailo_status create_feature(hailo_vstream_info_t vstream_info, size_t output_frame_size, std::shared_ptr<FeatureData<T>> &feature) {
    feature = std::make_shared<FeatureData<T>>(static_cast<uint32_t>(output_frame_size), vstream_info.quant_info.qp_zp,
        vstream_info.quant_info.qp_scale, vstream_info.shape.width, vstream_info);

    return HAILO_SUCCESS;
}

template <typename T>
static hailo_status hailo_postprocess(tflite::Interpreter * interpreter, int model_mode, std::vector<DetResult> & results, int cur_batch = -1){
    auto status = HAILO_SUCCESS;
    const auto & output_vstreams = interpreter->get_hailo_vstreams()->second;
    auto output_vstreams_size = output_vstreams.size();

    std::vector<std::shared_ptr<FeatureData<T>>> features;
    features.reserve(output_vstreams_size);
    for(size_t i = 0; i < output_vstreams_size; i++) {
        std::shared_ptr<FeatureData<T>> feature(nullptr);
        auto status = create_feature(output_vstreams[i].get_info(), output_vstreams[i].get_frame_size(), feature);
        if (HAILO_SUCCESS != status) {
            std::cerr << "ERROR: Failed creating feature with status = " << status << std::endl;
            return status;
        }

        features.emplace_back(feature);

        if(cur_batch == -1){
            TfLiteTensor* output_tensor_i = interpreter->output_tensor(i);
            TfLiteIntArray* output_dims = output_tensor_i->dims;

            int size = 1;
            for(int j = 1; j < output_dims->size; j++)
                size *= output_dims->data[j];

            if(output_tensor_i->type == kTfLiteUInt8){
                uint8_t * output_ptr = interpreter->typed_output_tensor<uint8_t>(i);
                std::vector<T>& buffer = features[i]->m_buffers.get_write_buffer();
                memcpy(buffer.data(), output_ptr, size * sizeof(uint8_t));
                features[i]->m_buffers.release_write_buffer();
            }
            else if(output_tensor_i->type == kTfLiteFloat32){
                float * output_ptr = interpreter->typed_output_tensor<float>(i);
                std::vector<T>& buffer = features[i]->m_buffers.get_write_buffer();
                memcpy(buffer.data(), output_ptr, size * sizeof(float));
                features[i]->m_buffers.release_write_buffer();
            }
        }
        else{
            int tensor_index;
            switch(model_mode){
                case 1:
                case 2:
                case 3:
                    tensor_index = i + 4;
                    break;
                case 4:
                case 5:
                case 6:
                    tensor_index = i + 1;
                    break;
            }

            TfLiteTensor* output_tensor_i = interpreter->output_tensor(tensor_index);
            TfLiteIntArray* output_dims = output_tensor_i->dims;

            int size = 1;
            for(int j = 1; j < output_dims->size; j++)
                size *= output_dims->data[j];

            if(interpreter->output_tensor(tensor_index)->type == kTfLiteUInt8){
                uint8_t * output_ptr = interpreter->typed_output_tensor<uint8_t>(tensor_index) + cur_batch * size;
                std::vector<T>& buffer = features[i]->m_buffers.get_write_buffer();
                memcpy(buffer.data(), output_ptr, size * sizeof(uint8_t));
                features[i]->m_buffers.release_write_buffer();
            }
            else if(interpreter->output_tensor(tensor_index)->type == kTfLiteFloat32){
                float * output_ptr = interpreter->typed_output_tensor<float>(tensor_index) + cur_batch * size;
                std::vector<T>& buffer = features[i]->m_buffers.get_write_buffer();
                memcpy(buffer.data(), output_ptr, size * sizeof(float));
                features[i]->m_buffers.release_write_buffer();
            }
        }
    }

    std::sort(features.begin(), features.end(), &FeatureData<T>::sort_tensors_by_size);

    HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));
        
    for(uint j = 0; j < features.size(); j++) {
        roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<T*>(features[j]->m_buffers.get_read_buffer().data()), features[j]->m_vstream_info));
    }

    filter(roi);

    for(auto &feature : features) {
        feature->m_buffers.release_read_buffer();
    }

    std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);

    results.reserve(detections.size());

    for(auto &detection : detections) {
        if (detection->get_confidence() == 0) {
            continue;
        }
        DetResult result;
        result.score = detection->get_confidence();

        HailoBBox bbox = detection->get_bbox();

        result.xmin = bbox.xmin();
        result.ymin = bbox.ymin();
        result.xmax = bbox.xmax();
        result.ymax = bbox.ymax();
        result.id = detection->get_class_id();

        results.push_back(result);
    }

    return status;
}

#endif //_HAILOPOST_HPP_