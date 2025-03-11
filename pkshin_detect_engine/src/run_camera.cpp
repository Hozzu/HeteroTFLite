#include <iostream>
#include <cstddef>
#include <cstdio>
#include <unistd.h>
#include <chrono>
#include <string>
#include <vector>

#include <engine_interface.hpp>
#include "hailo_post.hpp"
#include "maccel_post.hpp"

#include <opencv2/opencv.hpp>
#include <fastcv/fastcv.h>
#include <qcarcam.h>
#include <qcarcam_client.h>


static tflite::Interpreter * interpreter;
static int model_mode;
static std::vector<std::string> * labels_ptr;

static unsigned long average_inference_time = 0;
static unsigned int num_inference = 0;

static float score_threshold = 0.5;

static int input_height;
static int input_width;
static int input_channel;


// Callback function called when the camera frame is refreshed
void qcarcam_event_handler(int input_id, unsigned char* buf_ptr, size_t buf_len){
    std::vector<std::string> & labels = *labels_ptr;

    // Get the camera info
    unsigned int queryNumInputs = 0, queryFilled = 0;
    qcarcam_input_t * pInputs;

    if(qcarcam_query_inputs(NULL, 0, &queryNumInputs) != QCARCAM_RET_OK || queryNumInputs == 0){
        std::cerr << "ERROR: The camera is not found.\n";
        exit(-1);
    }

    if(queryNumInputs <= input_id){
        std::cerr << "ERROR: The number of available cameras is smaller than the camera id. Check that all the cameras are connected in order.\n";
        exit(-1);
    }

    pInputs = (qcarcam_input_t *)calloc(queryNumInputs, sizeof(*pInputs));       
    if(!pInputs){
        std::cerr << "ERROR: Failed to calloc\n";
        exit(-1);
    }

    if(qcarcam_query_inputs(pInputs, queryNumInputs, &queryFilled) != QCARCAM_RET_OK || queryFilled != queryNumInputs){
        std::cerr << "ERROR: Failed to get the camera info\n";
        exit(-1);
    }

    int camera_height = pInputs[input_id].res[0].height;
    int camera_width = pInputs[input_id].res[0].width;

    free(pInputs);

    // Change color format from uyuv to rgb
    uint8_t * uv = (uint8_t *)fcvMemAlloc(camera_width * camera_height, 16);
    uint8_t * y = (uint8_t *)fcvMemAlloc(camera_width * camera_height, 16);
    if(uv == NULL || y == NULL){
        std::cerr << "ERROR: Failed to fcvMemAlloc\n";
        exit(-1);
    }

    uint8_t * rgb_buf_ptr = new uint8_t[camera_height * camera_width * 3];
    if(rgb_buf_ptr == NULL){
        std::cerr << "ERROR: Failed memory allocation\n";
        exit(-1);
    }

    fcvDeinterleaveu8(buf_ptr, camera_width, camera_height, camera_width * 2, (uint8_t *)uv, camera_width, (uint8_t *)y, camera_width);
    fcvColorYCbCr422PseudoPlanarToRGB888u8((uint8_t *)y, (uint8_t *)uv, camera_width, camera_height, camera_width, camera_width, (uint8_t *)rgb_buf_ptr, camera_width * 3);

    // Resize image
    uint8_t * r_buf_ptr = new uint8_t[camera_height * camera_width];
    uint8_t * g_buf_ptr = new uint8_t[camera_height * camera_width];
    uint8_t * b_buf_ptr = new uint8_t[camera_height * camera_width];

    for(int i = 0; i < camera_height * camera_width; i++){
        r_buf_ptr[i] = rgb_buf_ptr[3 * i];
        g_buf_ptr[i] = rgb_buf_ptr[3 * i + 1];
        b_buf_ptr[i] = rgb_buf_ptr[3 * i + 2];
    }

    float scale_height = (float) input_height / camera_height;
    float scale_width = (float) input_width / camera_width;

    switch(model_mode){
        case 1: //ssd_mobilenet
            break;
        default:
        {
            float scale = scale_height < scale_width ? scale_height : scale_width;

            scale_height = scale;
            scale_width = scale;
        }
    }

    int resize_height = camera_height * scale_height;
    int resize_width = camera_width * scale_width;

    uint8_t * resize_img_ptr = new uint8_t[resize_height * resize_width * input_channel];
    uint8_t * r_resize_img_ptr = new uint8_t[resize_height * resize_width];
    uint8_t * g_resize_img_ptr = new uint8_t[resize_height * resize_width];
    uint8_t * b_resize_img_ptr = new uint8_t[resize_height * resize_width];

    fcvScaleu8(r_buf_ptr, camera_width, camera_height, camera_width, r_resize_img_ptr, resize_width, resize_height, resize_width, FASTCV_INTERPOLATION_TYPE_BILINEAR);
    fcvScaleu8(g_buf_ptr, camera_width, camera_height, camera_width, g_resize_img_ptr, resize_width, resize_height, resize_width, FASTCV_INTERPOLATION_TYPE_BILINEAR);
    fcvScaleu8(b_buf_ptr, camera_width, camera_height, camera_width, b_resize_img_ptr, resize_width, resize_height, resize_width, FASTCV_INTERPOLATION_TYPE_BILINEAR);

    for(int i = 0; i < resize_height * resize_width; i++){
        resize_img_ptr[3 * i] = r_resize_img_ptr[i];
        resize_img_ptr[3 * i + 1] = g_resize_img_ptr[i];
        resize_img_ptr[3 * i + 2] = b_resize_img_ptr[i];
    }

    delete r_buf_ptr;
    delete g_buf_ptr;
    delete b_buf_ptr;
    delete r_resize_img_ptr;
    delete g_resize_img_ptr;
    delete b_resize_img_ptr;

    // Preprocess
    switch(model_mode){
        case 1: //ssd_mobilenet
        {
            if(interpreter->input_tensor(0)->type == kTfLiteUInt8){
                uint8_t * input_img_ptr = new uint8_t[input_height * input_width * input_channel];
                memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(uint8_t));

                for(int i = 0; i < resize_height; i++){
                    for(int j = 0; j < resize_width; j++){
                        int input_idx = i * input_width + j;
                        int resize_idx = i * resize_width + j;

                        input_img_ptr[3 * input_idx] = resize_img_ptr[3 * resize_idx];
                        input_img_ptr[3 * input_idx + 1] = resize_img_ptr[3 * resize_idx + 1];
                        input_img_ptr[3 * input_idx + 2] = resize_img_ptr[3 * resize_idx + 2];
                    }
                }

                memcpy(interpreter->typed_input_tensor<uint8_t>(0), input_img_ptr, input_height * input_width * input_channel * sizeof(uint8_t));
                delete input_img_ptr;
            }
            else if(interpreter->input_tensor(0)->type == kTfLiteFloat32){
                float * input_img_ptr = new float[input_height * input_width * input_channel];
                memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(float));

                for(int i = 0; i < resize_height; i++){
                    for(int j = 0; j < resize_width; j++){
                        int input_idx = i * input_width + j;
                        int resize_idx = i * resize_width + j;

                        input_img_ptr[3 * input_idx] = (resize_img_ptr[3 * resize_idx] - 127.5) / 127.5;
                        input_img_ptr[3 * input_idx + 1] = (resize_img_ptr[3 * resize_idx + 1] - 127.5) / 127.5;
                        input_img_ptr[3 * input_idx + 2] = (resize_img_ptr[3 * resize_idx + 2] - 127.5) / 127.5;
                    }
                }

                memcpy(interpreter->typed_input_tensor<float>(0), input_img_ptr, input_height * input_width * input_channel * sizeof(float));
                delete input_img_ptr;
            }
            
            break;
        }
        case 2: //efficientdet
        {
            float mean_rgb[3] = {0.485 * 255, 0.456 * 255, 0.406 * 255};
            float stddev_rgb[3] = {0.229 * 255, 0.224 * 255, 0.225 * 255};
            float scale = 0.01862778328359127;
            int zero_point = 114;

            if(interpreter->input_tensor(0)->type == kTfLiteUInt8){
                uint8_t * input_img_ptr = new uint8_t[input_height * input_width * input_channel];
                memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(uint8_t));

                for(int i = 0; i < resize_height; i++){
                    for(int j = 0; j < resize_width; j++){
                        int input_idx = i * input_width + j;
                        int resize_idx = i * resize_width + j;

                        float normalized_r = (resize_img_ptr[3 * resize_idx] - mean_rgb[0]) / stddev_rgb[0];
                        float normalized_g = (resize_img_ptr[3 * resize_idx + 1] - mean_rgb[1]) / stddev_rgb[1];
                        float normalized_b = (resize_img_ptr[3 * resize_idx + 2] - mean_rgb[2]) / stddev_rgb[2];

                        input_img_ptr[3 * input_idx] = normalized_r / scale + zero_point;
                        input_img_ptr[3 * input_idx + 1] = normalized_g / scale + zero_point;
                        input_img_ptr[3 * input_idx + 2] = normalized_b / scale + zero_point;
                    }
                }

                memcpy(interpreter->typed_input_tensor<uint8_t>(0), input_img_ptr, input_height * input_width * input_channel * sizeof(uint8_t));
                delete input_img_ptr;
            }
            else if(interpreter->input_tensor(0)->type == kTfLiteFloat32){
                float * input_img_ptr = new float[input_height * input_width * input_channel];
                memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(float));

                for(int i = 0; i < resize_height; i++){
                    for(int j = 0; j < resize_width; j++){
                        int input_idx = i * input_width + j;
                        int resize_idx = i * resize_width + j;

                        input_img_ptr[3 * input_idx] = (resize_img_ptr[3 * resize_idx] - mean_rgb[0]) / stddev_rgb[0];
                        input_img_ptr[3 * input_idx + 1] = (resize_img_ptr[3 * resize_idx + 1] - mean_rgb[1]) / stddev_rgb[1];
                        input_img_ptr[3 * input_idx + 2] = (resize_img_ptr[3 * resize_idx + 2] - mean_rgb[2]) / stddev_rgb[2];
                    }
                }

                memcpy(interpreter->typed_input_tensor<float>(0), input_img_ptr, input_height * input_width * input_channel * sizeof(float));
                delete input_img_ptr;
            }

            break;
        }
        case 3: //efficientdet lite
        {
            float mean_rgb[3] = {127.0, 127.0, 127.0};
            float stddev_rgb[3] = {128.0, 128.0, 128.0};

            if(interpreter->input_tensor(0)->type == kTfLiteUInt8){
                uint8_t * input_img_ptr = new uint8_t[input_height * input_width * input_channel];
                memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(uint8_t));

                for(int i = 0; i < resize_height; i++){
                    for(int j = 0; j < resize_width; j++){
                        int input_idx = i * input_width + j;
                        int resize_idx = i * resize_width + j;

                        input_img_ptr[3 * input_idx] = resize_img_ptr[3 * resize_idx];
                        input_img_ptr[3 * input_idx + 1] = resize_img_ptr[3 * resize_idx + 1];
                        input_img_ptr[3 * input_idx + 2] = resize_img_ptr[3 * resize_idx + 2];
                    }
                }

                memcpy(interpreter->typed_input_tensor<uint8_t>(0), input_img_ptr, input_height * input_width * input_channel * sizeof(uint8_t));
                delete input_img_ptr;
            }
            else if(interpreter->input_tensor(0)->type == kTfLiteFloat32){
                float * input_img_ptr = new float[input_height * input_width * input_channel];
                memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(float));

                for(int i = 0; i < resize_height; i++){
                    for(int j = 0; j < resize_width; j++){
                        int input_idx = i * input_width + j;
                        int resize_idx = i * resize_width + j;

                        input_img_ptr[3 * input_idx] = (resize_img_ptr[3 * resize_idx] - mean_rgb[0]) / stddev_rgb[0];
                        input_img_ptr[3 * input_idx + 1] = (resize_img_ptr[3 * resize_idx + 1] - mean_rgb[1]) / stddev_rgb[1];
                        input_img_ptr[3 * input_idx + 2] = (resize_img_ptr[3 * resize_idx + 2] - mean_rgb[2]) / stddev_rgb[2];
                    }
                }

                memcpy(interpreter->typed_input_tensor<float>(0), input_img_ptr, input_height * input_width * input_channel * sizeof(float));
                delete input_img_ptr;
            }
            
            break;
        }
        case 4: //yolo int
        case 5: //yolov10 int
        case 6: //yolo obb int
	    {
	        if(interpreter->input_tensor(0)->type == kTfLiteUInt8){
                uint8_t * input_img_ptr = new uint8_t[input_height * input_width * input_channel];
                memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(uint8_t));

                for(int i = 0; i < resize_height; i++){
                    for(int j = 0; j < resize_width; j++){
                        int input_idx = i * input_width + j;
                        int resize_idx = i * resize_width + j;

                        input_img_ptr[3 * input_idx] = resize_img_ptr[3 * resize_idx];
                        input_img_ptr[3 * input_idx + 1] = resize_img_ptr[3 * resize_idx + 1];
                        input_img_ptr[3 * input_idx + 2] = resize_img_ptr[3 * resize_idx + 2];
                    }
                }

                memcpy(interpreter->typed_input_tensor<uint8_t>(0), input_img_ptr, input_height * input_width * input_channel * sizeof(uint8_t));
                delete input_img_ptr;
            }
            else if(interpreter->input_tensor(0)->type == kTfLiteFloat32){
                float * input_img_ptr = new float[input_height * input_width * input_channel];
                memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(float));

                for(int i = 0; i < resize_height; i++){
                    for(int j = 0; j < resize_width; j++){
                        int input_idx = i * input_width + j;
                        int resize_idx = i * resize_width + j;

                        input_img_ptr[3 * input_idx] = resize_img_ptr[3 * resize_idx] / 255.0;
                        input_img_ptr[3 * input_idx + 1] = resize_img_ptr[3 * resize_idx + 1] / 255.0;
                        input_img_ptr[3 * input_idx + 2] = resize_img_ptr[3 * resize_idx + 2] / 255.0;
                    }
                }

                memcpy(interpreter->typed_input_tensor<float>(0), input_img_ptr, input_height * input_width * input_channel * sizeof(float));
                delete input_img_ptr;
            }

            break;
	    }
    }

    // Inference
    auto start = std::chrono::high_resolution_clock::now();
    if(interpreter->Invoke() != kTfLiteOk){
        std::cerr << "ERROR: Model execute failed\n";
        exit(-1);
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - start;

    average_inference_time = (average_inference_time * num_inference + std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()) / (num_inference + 1);
    num_inference++;

    // Change color format from rgb to bgr for opencv image
    for(int i = 0; i < camera_width * camera_height * 3; i += 3){
        uint8_t tmp = rgb_buf_ptr[i];
        rgb_buf_ptr[i] = rgb_buf_ptr[i + 2];
        rgb_buf_ptr[i + 2] = tmp;
    }

    cv::Mat cvimg(camera_height, camera_width, CV_8UC3, rgb_buf_ptr);

    // Output of inference
    if(interpreter->is_hailo_output()){
        std::vector<DetResult> results;
        auto status = hailo_postprocess<uint8_t>(interpreter, model_mode, results);
        if (HAILO_SUCCESS != status) {
            std::cerr << "ERROR: hailo postprocess failed\n";
            exit(-1);
        }

        // Output of inference
        for(int i = 0; i < results.size(); i++){
            float score =  results[i].score;
            if(score < score_threshold)
                continue;

            float xmin = results[i].xmin * input_width / scale_width;
            float ymin = results[i].ymin * input_height / scale_height;
            float xmax = results[i].xmax * input_width / scale_width;
            float ymax = results[i].ymax * input_height / scale_height;

            int id = results[i].id - 1;

            switch(model_mode){
                case 1: //ssd mobilenet
                case 3: //efficientdet lite
                    break;
                default:
                {
                    if(id < 11)
                        id = id;
                    else if(id < 24)
                        id = id + 1;
                    else if(id < 26)
                        id = id + 2;
                    else if(id < 40)
                        id = id + 4;
                    else if(id < 60)
                        id = id + 5;
                    else if(id < 61)
                        id = id + 6;
                    else if(id < 62)
                        id = id + 8;
                    else if(id < 73)
                        id = id + 9;
                    else
                        id = id + 10;

                    break;
                }
            }

            //std::cout << i <<  ": output_classes: " << id << ", output_scores: " << score << ", output_locations: [" << xmin << "," << ymin << "," << xmax << ","<< ymax << "]\n";

            char str[100];
            sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
            cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
            cv::rectangle(cvimg, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(0, 0, 255), 2);
        }
    }
    else if(interpreter->is_maccel_output()){
        std::vector<DetResult> results;
        maccel_post(interpreter, model_mode, results, false);

        // Output of inference
        switch(model_mode){
            case 4: //yolo
            {
                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    if(score < score_threshold)
                        continue;

                    float xmin = results[i].xmin / scale_width;
                    float ymin = results[i].ymin / scale_height;
                    float xmax = results[i].xmax / scale_width;
                    float ymax = results[i].ymax / scale_height;

                    int id = results[i].id;

                    float width = xmax - xmin;
                    float height = ymax - ymin;

                    if(id < 11)
                        id = id;
                    else if(id < 24)
                        id = id + 1;
                    else if(id < 26)
                        id = id + 2;
                    else if(id < 40)
                        id = id + 4;
                    else if(id < 60)
                        id = id + 5;
                    else if(id < 61)
                        id = id + 6;
                    else if(id < 62)
                        id = id + 8;
                    else if(id < 73)
                        id = id + 9;
                    else
                        id = id + 10;

                    char str[100];
                    sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
                    cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
                    cv::rectangle(cvimg, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(0, 0, 255), 2);
                }

                break;
            }
        }
    }
    else if(interpreter->is_tflite_output()){
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

                float * output_locations = interpreter->typed_output_tensor<float>(output_box_idx);
                float * output_classes = interpreter->typed_output_tensor<float>(output_class_idx);
                float * output_scores = interpreter->typed_output_tensor<float>(output_score_idx);
                int output_nums = (int) *(interpreter->typed_output_tensor<float>(output_num_idx));

                for (int i = 0; i < output_nums; i++){
                    //std::cout << i <<  ": , output_classes: " << output_classes[i] << ", output_scores: " << output_scores[i] << ", output_locations: [" << output_locations[i * 4] << "," << output_locations[i * 4 + 1] << "," << output_locations[i * 4 + 2] << ","<< output_locations[i * 4 + 3] << "]\n";

                    float score =  output_scores[i];
                    if(score < score_threshold)
                        continue;

                    float ymin = output_locations[i * 4] * input_height / scale_height;
                    float xmin = output_locations[i * 4 + 1] * input_width / scale_width;
                    float ymax = output_locations[i * 4 + 2] * input_height / scale_height;
                    float xmax = output_locations[i * 4 + 3] * input_width / scale_width;

                    int id =  (int)(output_classes[i]);

                    char str[100];
                    sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
                    cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
                    cv::rectangle(cvimg, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(0, 0, 255), 2);
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

                int output_nums = (int) *(interpreter->typed_output_tensor<float>(output_num_idx));
                float * output_scores = interpreter->typed_output_tensor<float>(output_score_idx);
                float * output_classes = interpreter->typed_output_tensor<float>(output_class_idx);
                float * output_locations = interpreter->typed_output_tensor<float>(output_box_idx);

                for (int i = 0; i < output_nums; i++){
                    //std::cout << i <<  ": , output_classes: " << output_classes[i] << ", output_scores: " << output_scores[i] << ", output_locations: [" << output_locations[i * 4] << "," << output_locations[i * 4 + 1] << "," << output_locations[i * 4 + 2] << ","<< output_locations[i * 4 + 3] << "]\n";

                    float score =  output_scores[i];
                    if(score < score_threshold)
                        continue;

                    float ymin = output_locations[i * 4] * input_height / scale_height;
                    float xmin = output_locations[i * 4 + 1] * input_width / scale_width;
                    float ymax = output_locations[i * 4 + 2] * input_height / scale_height;
                    float xmax = output_locations[i * 4 + 3] * input_width / scale_width;

                    int id =  (int)(output_classes[i]);

                    char str[100];
                    sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
                    cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
                    cv::rectangle(cvimg, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(0, 0, 255), 2);
                }

                break;
            }
            case 4: //yolo
            {
                // Get the output tensor size info
                TfLiteTensor* output_tensor_0 = interpreter->output_tensor(0);
                TfLiteIntArray* output_dims = output_tensor_0->dims;
                int output_height = output_dims->data[1];
                int output_width = output_dims->data[2];

                // Parse output and apply nms
                float * output = interpreter->typed_output_tensor<float>(0);
                float (*output_arr)[output_width] = (float(*)[output_width])output;

                int max_detections = 300;
                float conf_threshold = score_threshold;
                float iou_threshold = 0.7;
                bool multi_label = false; // If true, nms is done per class. slow but accurate.

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

                    if(max_score == 0 && xmin == 0 && ymin == 0 && width == 0 && height == 0)
                        break;

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

                for(int i = 0; i < nms_result.size(); i++){
                    int idx = nms_result[i];

                    //std::cout << i <<  ": , class_id: " << class_ids[idx] << ", score: " << scores[idx] << ", xmin: " << boxes[idx].x << ", ymin: " << boxes[idx].y << ", width: " << boxes[idx].width << ", height: "<< boxes[idx].height << std::endl;

                    float score =  scores[idx];

                    float xmin = boxes[idx].x / scale_width;
                    float ymin = boxes[idx].y / scale_height;
                    float width = boxes[idx].width / scale_width;
                    float height = boxes[idx].height / scale_height;
                    float xmax = xmin + width;
                    float ymax = ymin + height;

                    int id =  class_ids[idx];
                    if(id < 11)
                        id = id;
                    else if(id < 24)
                        id = id + 1;
                    else if(id < 26)
                        id = id + 2;
                    else if(id < 40)
                        id = id + 4;
                    else if(id < 60)
                        id = id + 5;
                    else if(id < 61)
                        id = id + 6;
                    else if(id < 62)
                        id = id + 8;
                    else if(id < 73)
                        id = id + 9;
                    else
                        id = id + 10;

                    char str[100];
                    sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
                    cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
                    cv::rectangle(cvimg, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(0, 0, 255), 2);
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
                float * output = interpreter->typed_output_tensor<float>(0);
                float (*output_arr)[output_width] = (float(*)[output_width])output;

                for(int i = 0; i < output_height; i++){
                        //std::cout << i <<  ": id: " << output_arr[i][5] << ", score: " << output_arr[i][4] << ", xmin: " << output_arr[i][0] << ", ymin: " << output_arr[i][1] << ", xmax: " << output_arr[i][2] << ", ymax: "<< output_arr[i][3] << std::endl;

                    float score = output_arr[i][4];
                    if(score < 0.001)
                        continue;

                    float xmin = output_arr[i][0] * input_width / scale_width;
                    float ymin = output_arr[i][1] * input_height / scale_height;
                    float xmax = output_arr[i][2] * input_width / scale_width;
                    float ymax = output_arr[i][3] * input_height / scale_height;

                    float width = xmax - xmin;
                    float height = ymax - ymin;

                    int id =  output_arr[i][5];
                    if(id < 11)
                        id = id;
                    else if(id < 24)
                        id = id + 1;
                    else if(id < 26)
                        id = id + 2;
                    else if(id < 40)
                        id = id + 4;
                    else if(id < 60)
                        id = id + 5;
                    else if(id < 61)
                        id = id + 6;
                    else if(id < 62)
                        id = id + 8;
                    else if(id < 73)
                        id = id + 9;
                    else
                        id = id + 10;

                    char str[100];
                    sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
                    cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
                    cv::rectangle(cvimg, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(0, 0, 255), 2);
                }

                break;
            }
            case 6:    // yolo obb
            {
                // Get the output tensor size info
                TfLiteTensor *output_tensor_0 = interpreter->output_tensor(0);
                TfLiteIntArray *output_dims = output_tensor_0->dims;
                int output_height = output_dims->data[1];
                int output_width = output_dims->data[2];

                // Parse output and apply nms
                float *output = interpreter->typed_output_tensor<float>(0);
                float(*output_arr)[output_width] = (float(*)[output_width])output;

                int max_detections = 300;
                float conf_threshold = score_threshold;
                float iou_threshold = 0.7;
                bool multi_label = false;   // If true, nms is done per class. slow but accurate.

                float pi = 3.14159265358979323846;

                std::vector<int> class_ids;
                std::vector<float> scores;
                std::vector<cv::RotatedRect> boxes;

                for(int i = 0; i < output_width; i++){
                    float cx = output_arr[0][i] * input_width;                  // center x
                    float cy = output_arr[1][i] * input_height;                 // center y
                    float width = output_arr[2][i] * input_width;               // width
                    float height = output_arr[3][i] * input_height;             // height
                    float angle = output_arr[output_height - 1][i] * 180 / pi;  // angle

                    float max_score = 0;
                    int max_class_id = 0;

                    for(int j = 4; j < output_height - 1; j++){
                        float score = output_arr[j][i];

                        int class_id = j - 4;

                        if (score > max_score) {
                            max_score = score;
                            max_class_id = class_id;
                        }

                        if (multi_label) {
                            if (score > conf_threshold) {
                                scores.push_back(score);
                                class_ids.push_back(class_id);
                                cv::RotatedRect rotatedRect(cv::Point2f(cx, cy), cv::Size2f(width, height), angle);
                                boxes.push_back(rotatedRect);
                                //std::cout << i << ", "<< j <<  ": class_id: " << class_id << ", score: " << score << ", cx: " << cx << ", cy: " << cy << ", width: " << width << ", height: "<< height << ", angle: " << angle << std::endl;
                            }
                        }
                    }

                    //std::cout << i << ": max_class_id: " << max_class_id << ", max_score: " << max_score << ", cx: " << cx << ", cy: " << cy << ", width: " << width << ", height: "<< height << ", angle: " << angle << std::endl;

                    if (max_score == 0 && cx == 0 && cy == 0 && width == 0 && height == 0 && angle == 0)
                        break;

                    if (!multi_label) {
                        if (max_score > conf_threshold) {
                            scores.push_back(max_score);
                            class_ids.push_back(max_class_id);
                            cv::RotatedRect rotatedRect(cv::Point2f(cx, cy), cv::Size2f(width, height), angle);
                            boxes.push_back(rotatedRect);
                        }
                    }
                }

                std::vector<int> nms_result;
                if (multi_label) {
                    // Batched nms trick since batched nms function is only available opencv > 4.7.0
                    std::vector<cv::RotatedRect> _boxes = boxes;

                    for (int i = 0; i < _boxes.size(); i++) {
                        cv::Point2f offset(class_ids[i] * input_width, class_ids[i] * input_height);
                        _boxes[i].center = _boxes[i].center + offset;
                    }

                    cv::dnn::NMSBoxes(_boxes, scores, conf_threshold, iou_threshold, nms_result, 1.0, max_detections);
                }
                else {
                    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, nms_result, 1.0, max_detections);
                }

                for (int i = 0; i < nms_result.size(); i++) {
                    int idx = nms_result[i];

                    //std::cout << i <<  ": class_id: " << class_ids[idx] << ", score: " << scores[idx] << ", cx: " << boxes[idx].center.x << ", cy: " << boxes[idx].center.y << ", width: " << boxes[idx].size.height << ", height: "<< boxes[idx].size.width << std::endl;

                    int id = class_ids[idx];
                    float score = scores[idx];

                    cv::Point2f center = boxes[idx].center;
                    cv::Size2f size = boxes[idx].size;
                    float angle = boxes[idx].angle;

                    float cx = center.x / scale_width;
                    float cy = center.y / scale_height;
                    float width = size.width / scale_width;
                    float height = size.height / scale_height;

                    float xmin = cx - 0.5 * width;
                    float xmax = cx + 0.5 * width;
                    float ymin = cy - 0.5 * height;
                    float ymax = cy + 0.5 * height;

                    char str[100];
                    sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
                    cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));

                    cv::RotatedRect rotatedRect(cv::Point2f(cx, cy), cv::Size2f(width, height), angle);
                    cv::Point2f vertices[4];
                    rotatedRect.points(vertices);

                    for (int i = 0; i < 4; i++)
                        line(cvimg, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 0, 255), 2);
                }
                break;
            }
        }
    }

    memcpy(rgb_buf_ptr, cvimg.data, camera_width * camera_height * 3 * sizeof(uint8_t));

    // Change color format from bgr to uyuv
    fcvColorRGB888ToYCbCr422PseudoPlanaru8(rgb_buf_ptr, camera_width, camera_height, camera_width * 3, y, uv, camera_width, camera_width);
    fcvInterleaveu8(uv, y, camera_width, camera_height, camera_width, camera_width, buf_ptr, camera_width * 2);

    // Free memory
    delete rgb_buf_ptr;
    delete resize_img_ptr;
    fcvMemFree(uv);
    fcvMemFree(y);
}

bool run_qcarcam(tflite::Interpreter * interpreter_arg, int model_mode_arg, std::vector<std::string> * labels_arg, char * display_path){
    interpreter = interpreter_arg;
    model_mode = model_mode_arg;
    labels_ptr = labels_arg;

    // Get the input tensor size info
    TfLiteTensor* input_tensor_0 = interpreter->input_tensor(0);
    TfLiteIntArray* input_dims = input_tensor_0->dims;
    input_height = input_dims->data[1];
    input_width = input_dims->data[2];
    input_channel = input_dims->data[3];

    // Run qcarcam
    if(qcarcam_client_start_preview(display_path, qcarcam_event_handler) != QCARCAM_RET_OK){
        std::cerr << "ERROR: Cannot connect to the qcarcam. Please check the display setting file.\n";
        exit(-1);
    }

    // Wait the exit
    std::cout << "\nPress ctrl+c to exit.\n\n";
    int secs = 0;
    while (true){
        sleep(10);
        secs += 10;
        std::cout << std::fixed;
        std::cout.precision(3);
        std::cout << "Average inference speed(0~" << secs << "s): " << 1000000.0 / average_inference_time << "fps\n";
        std::cout << "Average inference speed(0~" << secs << "s): " << average_inference_time << "us\n\n";
    }

    // Stop qcarcam
    if(qcarcam_client_stop_preview() != QCARCAM_RET_OK){
        std::cerr << "ERROR: Cannot disconnect the qcarcam.\n";
        exit(-1);
    }

    return true;
}