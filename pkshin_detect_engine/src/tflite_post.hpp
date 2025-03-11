#ifndef _TFLITEPOST_HPP_
#define _TFLITEPOST_HPP_

#include <vector>

#include <engine_interface.hpp>

#include <opencv2/opencv.hpp>


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

static bool tflite_post(tflite::Interpreter * interpreter, int model_mode, std::vector<DetResult> & results, bool multi, int cur_batch = 0){
    switch(model_mode){
        case 1: //ssd_mobilenet
        {
            int output_box_idx = 0;
            int output_class_idx = 1;
            int output_score_idx = 2;
            int output_num_idx = 3;

            for(int i = 0; i < interpreter->outputs().size(); i++){
                if(strstr(interpreter->GetOutputName(i), "box")){
                    output_box_idx = i;
                }
                else if(strstr(interpreter->GetOutputName(i), "class")){
                    output_class_idx = i;
                }
                else if(strstr(interpreter->GetOutputName(i), "score")){
                    output_score_idx = i;
                }
                else if(strstr(interpreter->GetOutputName(i), "num")){
                    output_num_idx = i;
                }
            }

            int size_box = cur_batch;
            for(int i = 1; i < interpreter->output_tensor(output_box_idx)->dims->size; i++)
                size_box *= interpreter->output_tensor(output_box_idx)->dims->data[i];

            float * output_locations = interpreter->typed_output_tensor<float>(output_box_idx) + size_box;

            int size_class = cur_batch;
            for(int i = 1; i < interpreter->output_tensor(output_class_idx)->dims->size; i++)
                size_class *= interpreter->output_tensor(output_class_idx)->dims->data[i];

            float * output_classes = interpreter->typed_output_tensor<float>(output_class_idx) + size_class;

            int size_score = cur_batch;
            for(int i = 1; i < interpreter->output_tensor(output_score_idx)->dims->size; i++)
                size_score *= interpreter->output_tensor(output_score_idx)->dims->data[i];

            float * output_scores = interpreter->typed_output_tensor<float>(output_score_idx) + size_score;

            int size_num = cur_batch;
            for(int i = 1; i < interpreter->output_tensor(output_num_idx)->dims->size; i++)
                size_num *= interpreter->output_tensor(output_num_idx)->dims->data[i];

            int output_nums = (int) *(interpreter->typed_output_tensor<float>(output_num_idx) + size_num);

            results.reserve(output_nums);

            for (int i = 0; i < output_nums; i++){
                //std::cout << i <<  ": output_classes: " << output_classes[i] << ", output_scores: " << output_scores[i] << ", output_locations: [" << output_locations[i * 4] << "," << output_locations[i * 4 + 1] << "," << output_locations[i * 4 + 2] << ","<< output_locations[i * 4 + 3] << "]\n";

                float score =  output_scores[i];

                float ymin = output_locations[i * 4];
                float xmin = output_locations[i * 4 + 1];
                float ymax = output_locations[i * 4 + 2];
                float xmax = output_locations[i * 4 + 3];

                int id =  (int)(output_classes[i]);

                DetResult result;

                result.score =  score;
                result.xmin = xmin;
                result.ymin = ymin;
                result.xmax = xmax;
                result.ymax = ymax;
                result.id =  id;

                results.push_back(result);
            }

            break;
        }
        case 2: //efficientdet
        case 3: //efficientdet lite
        {
            int output_num_idx = 0;
            int output_score_idx = 1;
            int output_class_idx = 2;
            int output_box_idx = 3;

            for(int i = 0; i < interpreter->outputs().size(); i++){
                if(strstr(interpreter->GetOutputName(i), ":0")){
                    output_num_idx = i;
                }
                else if(strstr(interpreter->GetOutputName(i), ":1")){
                    output_score_idx = i;
                }
                else if(strstr(interpreter->GetOutputName(i), ":2")){
                    output_class_idx = i;
                }
                else if(strstr(interpreter->GetOutputName(i), ":3")){
                    output_box_idx = i;
                }
            }

            int size_box = cur_batch;
            for(int i = 1; i < interpreter->output_tensor(output_box_idx)->dims->size; i++)
                size_box *= interpreter->output_tensor(output_box_idx)->dims->data[i];

            float * output_locations = interpreter->typed_output_tensor<float>(output_box_idx) + size_box;

            int size_class = cur_batch;
            for(int i = 1; i < interpreter->output_tensor(output_class_idx)->dims->size; i++)
                size_class *= interpreter->output_tensor(output_class_idx)->dims->data[i];

            float * output_classes = interpreter->typed_output_tensor<float>(output_class_idx) + size_class;

            int size_score = cur_batch;
            for(int i = 1; i < interpreter->output_tensor(output_score_idx)->dims->size; i++)
                size_score *= interpreter->output_tensor(output_score_idx)->dims->data[i];

            float * output_scores = interpreter->typed_output_tensor<float>(output_score_idx) + size_score;

            int size_num = cur_batch;
            for(int i = 1; i < interpreter->output_tensor(output_num_idx)->dims->size; i++)
                size_num *= interpreter->output_tensor(output_num_idx)->dims->data[i];

            int output_nums = (int) *(interpreter->typed_output_tensor<float>(output_num_idx) + size_num);

            results.reserve(output_nums);

            for(int i = 0; i < output_nums; i++){     
                //std::cout << i <<  ": output_classes: " << output_classes[i] << ", output_scores: " << output_scores[i] << ", output_locations: [" << output_locations[i * 4] << "," << output_locations[i * 4 + 1] << "," << output_locations[i * 4 + 2] << ","<< output_locations[i * 4 + 3] << "]\n";

                float score =  output_scores[i];

                float ymin = output_locations[i * 4];
                float xmin = output_locations[i * 4 + 1];
                float ymax = output_locations[i * 4 + 2];
                float xmax = output_locations[i * 4 + 3];

                int id =  (int)(output_classes[i]);

                DetResult result;

                result.score =  score;
                result.xmin = xmin;
                result.ymin = ymin;
                result.xmax = xmax;
                result.ymax = ymax;
                result.id =  id;

                results.push_back(result);
            }

            break;
        }
        case 4: //yolo
        {
            // Get the input tensor size info
            TfLiteTensor* input_tensor_0 = interpreter->input_tensor(0);
            TfLiteIntArray* input_dims = input_tensor_0->dims;
            int input_height = input_dims->data[1];
            int input_width = input_dims->data[2];
            int input_channel = input_dims->data[3];

            // Get the output tensor size info
            TfLiteTensor* output_tensor_0 = interpreter->output_tensor(0);
            TfLiteIntArray* output_dims = output_tensor_0->dims;
            int output_height = output_dims->data[1];
            int output_width = output_dims->data[2];

            // Parse output and apply nms
            float * output = interpreter->typed_output_tensor<float>(0) + cur_batch * output_height * output_width;
            float (*output_arr)[output_width] = (float(*)[output_width])output;

            int max_detections = 300;
            float conf_threshold = 0.001;
            float iou_threshold = 0.7;
            bool multi_label = true; // If true, nms is done per class. slow but accurate.

            std::vector<int> class_ids;
            std::vector<float> scores;
            std::vector<cv::Rect2d> boxes;

            for(int i = 0; i < output_width; i++){
                float x = output_arr[0][i];
                float y = output_arr[1][i];
                float w = output_arr[2][i];
                float h = output_arr[3][i];

                float xmin = (x - 0.5 * w) * input_width;
                float ymin = (y - 0.5 * h) * input_height;
                float width = w * input_width;
                float height = h * input_height;

                float max_score = 0;
                int max_class_id = 0;

                for(int j = 4; j < output_height; j++){
                    float score = output_arr[j][i];

                    int class_id = j - 4;

                    if(score > max_score){
                        max_score = score;
                        max_class_id = class_id;
                    }

                    if(multi_label){
                        if(score > conf_threshold){
                            scores.push_back(score);
                            class_ids.push_back(class_id);
                            boxes.push_back(cv::Rect2d(xmin, ymin, width, height));
                        }
                    }
                    //std::cout << i << ", "<< j <<  ": class_id: " << class_id << ", score: " << score << ", xn: " << x << ", y: " << y << ", w: " << w << ", h: "<< h << std::endl;
                }

                //std::cout << i <<  ": max_class_id: " << max_class_id << ", max_score: " << max_score << ", xmin: " << xmin << ", ymin: " << ymin << ", width: " << width << ", height: "<< height << std::endl;

                if(!multi_label){
                    if(max_score > conf_threshold){
                        scores.push_back(max_score);
                        class_ids.push_back(max_class_id);
                        boxes.push_back(cv::Rect2d(xmin, ymin, width, height));
                    }
                }
            }
            
            std::vector<int> nms_result;
            if(multi_label){
                // Batched nms trick since batched nms function is only available opencv > 4.7.0
                std::vector<cv::Rect2d> _boxes = boxes;

                for(int i = 0; i < _boxes.size(); i++){
                    cv::Point2d offset(class_ids[i] * input_width, class_ids[i] * input_height);
                    _boxes[i] = _boxes[i] + offset;
                }

                cv::dnn::NMSBoxes(_boxes, scores, conf_threshold, iou_threshold, nms_result, 1.0, max_detections);
            }
            else{
                cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, nms_result, 1.0, max_detections);
            }

            results.reserve(nms_result.size());

            for(int i = 0; i < nms_result.size(); i++){
                int idx = nms_result[i];

                //std::cout << i <<  ": class_id: " << class_ids[idx] << ", score: " << scores[idx] << ", xmin: " << boxes[idx].x << ", ymin: " << boxes[idx].y << ", width: " << boxes[idx].width << ", height: "<< boxes[idx].height << std::endl;

                float score =  scores[idx];

                float xmin = boxes[idx].x;
                float ymin = boxes[idx].y;
                float xmax = boxes[idx].x + boxes[idx].width;
                float ymax = boxes[idx].y + boxes[idx].height;

                int id =  class_ids[idx];

                DetResult result;

                result.score =  score;
                result.xmin = xmin;
                result.ymin = ymin;
                result.xmax = xmax;
                result.ymax = ymax;
                result.id =  id;

                results.push_back(result);
            }

            break;
        }
        case 5: //yolov10
        {
            // Get the output tensor size info
            TfLiteTensor* output_tensor_0 = interpreter->output_tensor(0);
            TfLiteIntArray* output_dims = output_tensor_0->dims;
            int output_height = output_dims->data[1];
            int output_width = output_dims->data[2];

            // Parse output and apply nms
            float * output = interpreter->typed_output_tensor<float>(0) + cur_batch * output_height * output_width;
            float (*output_arr)[output_width] = (float(*)[output_width])output;

            results.reserve(output_height);

            for(int i = 0; i < output_height; i++){
                //std::cout << i <<  ": id: " << output_arr[i][5] << ", score: " << output_arr[i][4] << ", xmin: " << output_arr[i][0] << ", ymin: " << output_arr[i][1] << ", xmax: " << output_arr[i][2] << ", ymax: "<< output_arr[i][3] << std::endl;

                float score = output_arr[i][4];
                if(score < 0.001)
                    continue;

                float xmin = output_arr[i][0];
                float ymin = output_arr[i][1];
                float xmax = output_arr[i][2];
                float ymax = output_arr[i][3];

                int id =  output_arr[i][5];

                DetResult result;

                result.score =  score;
                result.xmin = xmin;
                result.ymin = ymin;
                result.xmax = xmax;
                result.ymax = ymax;
                result.id =  id;

                results.push_back(result);
            }

            break;
        }
    }

    return true;
}

#endif // _TFLITEPOST_HPP_