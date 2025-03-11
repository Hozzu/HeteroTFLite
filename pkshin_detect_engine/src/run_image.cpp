#include <iostream>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <chrono>
#include <string>
#include <vector>
#include <thread>
#include <mutex>

#define _GNU_SOURCE 
#include <pthread.h>

#include <engine_interface.hpp>
#include "hailo_post.hpp"
#include "maccel_post.hpp"
#include "tflite_post.hpp"

#include <opencv2/opencv.hpp>
#include <fastcv/fastcv.h>
#include <dirent.h>
#include <jpeglib.h>
#include <json-c/json.h>

static auto preprocess_start = std::chrono::high_resolution_clock::now();
static auto invoke_start = std::chrono::high_resolution_clock::now();

static unsigned int num_preproces = 0;
static unsigned int num_postprocess = 0;
static double sum_preprocess_time = 0;
static double sum_postprocess_time = 0;

static unsigned int num_turnaround = 0;
static double sum_turnaround = 0;
static double max_turnaround = 0;

static std::mutex in_postprocess_mutex;
static std::mutex in_preprocess_mutex;


void preprocess_thread(tflite::Interpreter * interpreter, int model_mode, std::string filename, json_object * json_images, int cur_batch, std::vector<int> &img_heights, std::vector<int> &img_widths, int image_id){
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    CPU_SET(1, &cpuset);
    CPU_SET(2, &cpuset);
    CPU_SET(3, &cpuset);
    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    
    // Decode the image
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    FILE * fp = fopen(filename.c_str(), "rb");
    if(fp == NULL) {
        std::cerr << "ERROR: Cannot open the image: " << filename << std::endl;
        exit(-1);
    }
    jpeg_stdio_src(&cinfo, fp);

    jpeg_read_header(&cinfo, TRUE);

    cinfo.out_color_space = JCS_RGB;
    cinfo.output_components = 3;

    jpeg_start_decompress(&cinfo);

    // Get the image data
    int img_height = cinfo.output_height;
    int img_width = cinfo.output_width;
    int row_stride = cinfo.output_width * 3;

    img_heights[cur_batch] = img_height;
    img_widths[cur_batch] = img_width;

    uint8_t * rgb_buf_ptr = (uint8_t *)malloc(sizeof(uint8_t) * img_height * img_width * 3);
    while(cinfo.output_scanline < cinfo.output_height){
        uint8_t * rowptr = rgb_buf_ptr + row_stride * cinfo.output_scanline; 
        jpeg_read_scanlines(&cinfo, &rowptr, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    fclose(fp);

    // Write json images
    json_object * json_image = json_object_new_object();

    json_object_object_add(json_image, "id", json_object_new_int(image_id));
    json_object_object_add(json_image, "file_name", json_object_new_string(filename.c_str()));
    json_object_object_add(json_image, "width", json_object_new_int(img_width));
    json_object_object_add(json_image, "height", json_object_new_int(img_height));

    in_preprocess_mutex.lock();
    json_object_array_add(json_images, json_image);
    in_preprocess_mutex.unlock();

    // Get the input tensor size info
    TfLiteTensor* input_tensor_0 = interpreter->input_tensor(0);
    TfLiteIntArray* input_dims = input_tensor_0->dims;
    int input_height = input_dims->data[1];
    int input_width = input_dims->data[2];
    int input_channel = input_dims->data[3];

    // Calculate scale
    float scale_height = (float) input_height / img_height;
    float scale_width = (float) input_width / img_width;

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

    // Resize image
    int resize_height = img_height * scale_height;
    int resize_width = img_width * scale_width;

    cv::Mat cvImg(cv::Size(img_width, img_height), CV_8UC3, rgb_buf_ptr);

    cv::Mat resizecvImg;
    cv::resize(cvImg, resizecvImg, cv::Size(resize_width, resize_height));
    uint8_t * resize_img_ptr = resizecvImg.data;

    free(rgb_buf_ptr);

    for(int i = 0; i < interpreter->inputs().size(); i++){
        // Normalize and pad input
        void * input_img_ptr_arg;
        if(interpreter->input_tensor(i)->type == kTfLiteUInt8){
            input_img_ptr_arg = interpreter->typed_input_tensor<uint8_t>(i) + cur_batch * input_height * input_width * input_channel;

        }
        else if(interpreter->input_tensor(i)->type == kTfLiteFloat32){
            input_img_ptr_arg = interpreter->typed_input_tensor<float>(i) + cur_batch * input_height * input_width * input_channel;
        }

        switch(model_mode){
            case 1: //ssd_mobilenet
            {
                if(interpreter->input_tensor(i)->type == kTfLiteUInt8){
                    uint8_t * input_img_ptr = (uint8_t *) input_img_ptr_arg;
                    memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(uint8_t));

                    for(int j = 0; j < resize_height; j++){
                        for(int k = 0; k < resize_width; k++){
                            int input_idx = j * input_width + k;
                            int resize_idx = j * resize_width + k;

                            input_img_ptr[3 * input_idx] = resize_img_ptr[3 * resize_idx];
                            input_img_ptr[3 * input_idx + 1] = resize_img_ptr[3 * resize_idx + 1];
                            input_img_ptr[3 * input_idx + 2] = resize_img_ptr[3 * resize_idx + 2];
                        }
                    }
                }
                else if(interpreter->input_tensor(i)->type == kTfLiteFloat32){
                    float * input_img_ptr = (float *) input_img_ptr_arg;
                    memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(float));

                    for(int j = 0; j < resize_height; j++){
                        for(int k = 0; k < resize_width; k++){
                            int input_idx = j * input_width + k;
                            int resize_idx = j * resize_width + k;

                            input_img_ptr[3 * input_idx] = (resize_img_ptr[3 * resize_idx] - 127.5) / 127.5;
                            input_img_ptr[3 * input_idx + 1] = (resize_img_ptr[3 * resize_idx + 1] - 127.5) / 127.5;
                            input_img_ptr[3 * input_idx + 2] = (resize_img_ptr[3 * resize_idx + 2] - 127.5) / 127.5;
                        }
                    }
                }
                
                break;
            }
            case 2: //efficientdet
            {
                float mean_rgb[3] = {0.485 * 255, 0.456 * 255, 0.406 * 255};
                float stddev_rgb[3] = {0.229 * 255, 0.224 * 255, 0.225 * 255};
                float scale = 0.01862778328359127;
                int zero_point = 114;

                if(interpreter->input_tensor(i)->type == kTfLiteUInt8){
                    uint8_t * input_img_ptr = (uint8_t *) input_img_ptr_arg;
                    memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(uint8_t));

                    for(int j = 0; j < resize_height; j++){
                        for(int k = 0; k < resize_width; k++){
                            int input_idx = j * input_width + k;
                            int resize_idx = j * resize_width + k;

                            float normalized_r = (resize_img_ptr[3 * resize_idx] - mean_rgb[0]) / stddev_rgb[0];
                            float normalized_g = (resize_img_ptr[3 * resize_idx + 1] - mean_rgb[1]) / stddev_rgb[1];
                            float normalized_b = (resize_img_ptr[3 * resize_idx + 2] - mean_rgb[2]) / stddev_rgb[2];

                            input_img_ptr[3 * input_idx] = normalized_r / scale + zero_point;
                            input_img_ptr[3 * input_idx + 1] = normalized_g / scale + zero_point;
                            input_img_ptr[3 * input_idx + 2] = normalized_b / scale + zero_point;
                        }
                    }
                }
                else if(interpreter->input_tensor(i)->type == kTfLiteFloat32){
                    float * input_img_ptr = (float *) input_img_ptr_arg;
                    memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(float));

                    for(int j = 0; j < resize_height; j++){
                        for(int k = 0; k < resize_width; k++){
                            int input_idx = j * input_width + k;
                            int resize_idx = j * resize_width + k;

                            input_img_ptr[3 * input_idx] = (resize_img_ptr[3 * resize_idx] - mean_rgb[0]) / stddev_rgb[0];
                            input_img_ptr[3 * input_idx + 1] = (resize_img_ptr[3 * resize_idx + 1] - mean_rgb[1]) / stddev_rgb[1];
                            input_img_ptr[3 * input_idx + 2] = (resize_img_ptr[3 * resize_idx + 2] - mean_rgb[2]) / stddev_rgb[2];
                        }
                    }
                }

                break;
            }
            case 3: //efficientdet lite
            {
                float mean_rgb[3] = {127.0, 127.0, 127.0};
                float stddev_rgb[3] = {128.0, 128.0, 128.0};

                if(interpreter->input_tensor(i)->type == kTfLiteUInt8){
                    uint8_t * input_img_ptr = (uint8_t *) input_img_ptr_arg;
                    memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(uint8_t));

                    for(int j = 0; j < resize_height; j++){
                        for(int k = 0; k < resize_width; k++){
                            int input_idx = j * input_width + k;
                            int resize_idx = j * resize_width + k;

                            input_img_ptr[3 * input_idx] = resize_img_ptr[3 * resize_idx];
                            input_img_ptr[3 * input_idx + 1] = resize_img_ptr[3 * resize_idx + 1];
                            input_img_ptr[3 * input_idx + 2] = resize_img_ptr[3 * resize_idx + 2];
                        }
                    }
                }
                else if(interpreter->input_tensor(i)->type == kTfLiteFloat32){
                    float * input_img_ptr = (float *) input_img_ptr_arg;
                    memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(float));

                    for(int j = 0; j < resize_height; j++){
                        for(int k = 0; k < resize_width; k++){
                            int input_idx = j * input_width + k;
                            int resize_idx = j * resize_width + k;

                            input_img_ptr[3 * input_idx] = (resize_img_ptr[3 * resize_idx] - mean_rgb[0]) / stddev_rgb[0];
                            input_img_ptr[3 * input_idx + 1] = (resize_img_ptr[3 * resize_idx + 1] - mean_rgb[1]) / stddev_rgb[1];
                            input_img_ptr[3 * input_idx + 2] = (resize_img_ptr[3 * resize_idx + 2] - mean_rgb[2]) / stddev_rgb[2];
                        }
                    }
                }
                
                break;
            }
            case 4: //yolo
            case 5: //yolov10
            case 6: //yolo obb
            {
                if(interpreter->input_tensor(i)->type == kTfLiteUInt8){
                    uint8_t * input_img_ptr = (uint8_t *) input_img_ptr_arg;
                    memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(uint8_t));

                    for(int j = 0; j < resize_height; j++){
                        for(int k = 0; k < resize_width; k++){
                            int input_idx = j * input_width + k;
                            int resize_idx = j * resize_width + k;

                            input_img_ptr[3 * input_idx] = resize_img_ptr[3 * resize_idx];
                            input_img_ptr[3 * input_idx + 1] = resize_img_ptr[3 * resize_idx + 1];
                            input_img_ptr[3 * input_idx + 2] = resize_img_ptr[3 * resize_idx + 2];
                        }
                    }
                }
                else if(interpreter->input_tensor(i)->type == kTfLiteFloat32){
                    float * input_img_ptr = (float *) input_img_ptr_arg;
                    memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(float));

                    for(int j = 0; j < resize_height; j++){
                        for(int k = 0; k < resize_width; k++){
                            int input_idx = j * input_width + k;
                            int resize_idx = j * resize_width + k;

                            input_img_ptr[3 * input_idx] = resize_img_ptr[3 * resize_idx] / 255.0;
                            input_img_ptr[3 * input_idx + 1] = resize_img_ptr[3 * resize_idx + 1] / 255.0;
                            input_img_ptr[3 * input_idx + 2] = resize_img_ptr[3 * resize_idx + 2] / 255.0;
                        }
                    }
                }

                break;
            }
        }
    }

    auto preprocess_elapsed = std::chrono::high_resolution_clock::now() - preprocess_start;
    in_preprocess_mutex.lock();
    sum_preprocess_time += std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_elapsed).count();
    num_preproces++;
    in_preprocess_mutex.unlock();
}

void postprocess_thread(tflite::Interpreter * interpreter, int model_mode, std::vector<int> &img_heights, std::vector<int> &img_widths, std::vector<int> &image_ids, json_object * json_annotations, int cur_batch){
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    CPU_SET(1, &cpuset);
    CPU_SET(2, &cpuset);
    CPU_SET(3, &cpuset);
    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    
    // Get the input tensor size info
    TfLiteTensor* input_tensor_0 = interpreter->input_tensor(0);
    TfLiteIntArray* input_dims = input_tensor_0->dims;
    int input_height = input_dims->data[1];
    int input_width = input_dims->data[2];

    // Calculate scale
    float scale_height = (float) input_height / img_heights[cur_batch];
    float scale_width = (float) input_width / img_widths[cur_batch];

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

    // Postprocess
    if(interpreter->is_tflite_output(cur_batch)){
        std::vector<DetResult> results;
        tflite_post(interpreter, model_mode, results, true, cur_batch);

        // Output of inference   
        switch(model_mode){
            case 1: //ssd mobilenet
            case 2: //efficientdet
            case 3: //efficientdet lite
            {
                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    int id = results[i].id;

                    float xmin = results[i].xmin * input_width / scale_width;
                    float ymin = results[i].ymin * input_height / scale_height;
                    float xmax = results[i].xmax * input_width / scale_width;
                    float ymax = results[i].ymax * input_height / scale_height;

                    float width = xmax - xmin;
                    float height = ymax - ymin;

                    //std::cout << i <<  ": output_classes: " << id << ", output_scores: " << score << ", output_locations: [" << xmin << "," << ymin << "," << xmax << ","<< ymax << "]\n";

                    // Write json annotations
                    json_object * json_annotation = json_object_new_object();

                    json_object * json_bbox = json_object_new_array();

                    json_object_array_add(json_bbox, json_object_new_int(round(xmin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(ymin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(width)));
                    json_object_array_add(json_bbox, json_object_new_int(round(height)));

                    json_object_object_add(json_annotation, "image_id", json_object_new_int(image_ids[cur_batch]));
                    json_object_object_add(json_annotation, "bbox", json_bbox);
                    json_object_object_add(json_annotation, "category_id", json_object_new_int(id + 1));
                    json_object_object_add(json_annotation, "score", json_object_new_double(score));
                    
                    in_postprocess_mutex.lock();
                    json_object_array_add(json_annotations, json_annotation);
                    in_postprocess_mutex.unlock();
                }

                break;
            }
            case 4: //yolo
            {
                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    int id = results[i].id;

                    float xmin = results[i].xmin / scale_width;
                    float ymin = results[i].ymin / scale_height;
                    float xmax = results[i].xmax / scale_width;
                    float ymax = results[i].ymax / scale_height;

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

                    // Write json annotations
                    json_object * json_annotation = json_object_new_object();

                    json_object * json_bbox = json_object_new_array();

                    json_object_array_add(json_bbox, json_object_new_int(round(xmin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(ymin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(width)));
                    json_object_array_add(json_bbox, json_object_new_int(round(height)));

                    json_object_object_add(json_annotation, "image_id", json_object_new_int(image_ids[cur_batch]));
                    json_object_object_add(json_annotation, "bbox", json_bbox);
                    json_object_object_add(json_annotation, "category_id", json_object_new_int(id + 1));
                    json_object_object_add(json_annotation, "score", json_object_new_double(score));

                    in_postprocess_mutex.lock();
                    json_object_array_add(json_annotations, json_annotation);
                    in_postprocess_mutex.unlock();
                }

                break;
            }
            case 5: //yolov10
            {
                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    int id = results[i].id;

                    float xmin = results[i].xmin * input_width / scale_width;
                    float ymin = results[i].ymin * input_height / scale_height;
                    float xmax = results[i].xmax * input_width / scale_width;
                    float ymax = results[i].ymax * input_height / scale_height;

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

                    // Write json annotations
                    json_object * json_annotation = json_object_new_object();

                    json_object * json_bbox = json_object_new_array();

                    json_object_array_add(json_bbox, json_object_new_int(round(xmin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(ymin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(width)));
                    json_object_array_add(json_bbox, json_object_new_int(round(height)));

                    json_object_object_add(json_annotation, "image_id", json_object_new_int(image_ids[cur_batch]));
                    json_object_object_add(json_annotation, "bbox", json_bbox);
                    json_object_object_add(json_annotation, "category_id", json_object_new_int(id + 1));
                    json_object_object_add(json_annotation, "score", json_object_new_double(score));

                    in_postprocess_mutex.lock();
                    json_object_array_add(json_annotations, json_annotation);
                    in_postprocess_mutex.unlock();
                }

                break;
            }
            case 6: //yolo obb
            {
                // Get the output tensor size info
                TfLiteTensor *output_tensor_0 = interpreter->output_tensor(0);
                TfLiteIntArray *output_dims = output_tensor_0->dims;
                int output_height = output_dims->data[1];
                int output_width = output_dims->data[2];

                // Parse output and apply nms
                float *output = interpreter->typed_output_tensor<float>(0) + cur_batch * output_height * output_width;
                float(*output_arr)[output_width] = (float(*)[output_width])output;

                int max_detections = 300;
                float conf_threshold = 0.001;
                float iou_threshold = 0.7;
                bool multi_label = true;   // If true, nms is done per class. slow but accurate.

                float pi = 3.14159265358979323846;

                std::vector<int> class_ids;
                std::vector<float> scores;
                std::vector<cv::RotatedRect> boxes;

                for(int i = 0; i < output_width; i++){
                    float cx = output_arr[0][i] * input_width;		            // center x
                    float cy = output_arr[1][i] * input_height;		            // center y
                    float width = output_arr[2][i] * input_width;	            // width
                    float height = output_arr[3][i] * input_height;	            // height
                    float angle = output_arr[output_height - 1][i] * 180 / pi;    // angle

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

                    cv::RotatedRect rotatedRect(cv::Point2f(cx, cy), cv::Size2f(width, height), angle);
                    cv::Point2f vertices[4];
                    rotatedRect.points(vertices);

                    // Write json annotations
                    json_object *json_annotation = json_object_new_object();

                    json_object *rbox = json_object_new_array();
                    json_object *poly = json_object_new_array();

                    json_object_array_add(rbox, json_object_new_double(cx));
                    json_object_array_add(rbox, json_object_new_double(cy));
                    json_object_array_add(rbox, json_object_new_double(width));
                    json_object_array_add(rbox, json_object_new_double(height));
                    json_object_array_add(rbox, json_object_new_double(angle));

                    json_object_array_add(poly, json_object_new_double(vertices[0].x));
                    json_object_array_add(poly, json_object_new_double(vertices[0].y));
                    json_object_array_add(poly, json_object_new_double(vertices[1].x));
                    json_object_array_add(poly, json_object_new_double(vertices[1].y));
                    json_object_array_add(poly, json_object_new_double(vertices[2].x));
                    json_object_array_add(poly, json_object_new_double(vertices[2].y));
                    json_object_array_add(poly, json_object_new_double(vertices[3].x));
                    json_object_array_add(poly, json_object_new_double(vertices[3].y));

                    json_object_object_add(json_annotation, "image_id", json_object_new_int(image_ids[cur_batch]));
                    json_object_object_add(json_annotation, "rbox", rbox);
                    json_object_object_add(json_annotation, "poly", poly);
                    json_object_object_add(json_annotation, "category_id", json_object_new_int(id + 1));
                    json_object_object_add(json_annotation, "score", json_object_new_double(score));

                    in_postprocess_mutex.lock();
                    json_object_array_add(json_annotations, json_annotation);
                    in_postprocess_mutex.unlock();
                }
                break;
            }
        }
    }
    else if(interpreter->is_hailo_output(cur_batch)){
        std::vector<DetResult> results;
        auto status = hailo_postprocess<uint8_t>(interpreter, model_mode, results, cur_batch);
        if (HAILO_SUCCESS != status) {
            std::cerr << "ERROR: hailo postprocess failed\n";
            exit(-1);
        }

        // Output of inference
        for(int i = 0; i < results.size(); i++){
            float score =  results[i].score;

            float xmin = results[i].xmin * input_width / scale_width;
            float ymin = results[i].ymin * input_height / scale_height;
            float xmax = results[i].xmax * input_width / scale_width;
            float ymax = results[i].ymax * input_height / scale_height;

            int id = results[i].id - 1;

            float width = xmax - xmin;
            float height = ymax - ymin;

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

            // Write json annotations
            json_object * json_annotation = json_object_new_object();

            json_object * json_bbox = json_object_new_array();

            json_object_array_add(json_bbox, json_object_new_int(round(xmin)));
            json_object_array_add(json_bbox, json_object_new_int(round(ymin)));
            json_object_array_add(json_bbox, json_object_new_int(round(width)));
            json_object_array_add(json_bbox, json_object_new_int(round(height)));

            json_object_object_add(json_annotation, "image_id", json_object_new_int(image_ids[cur_batch]));
            json_object_object_add(json_annotation, "bbox", json_bbox);
            json_object_object_add(json_annotation, "category_id", json_object_new_int(id + 1));
            json_object_object_add(json_annotation, "score", json_object_new_double(score));

            in_postprocess_mutex.lock();
            json_object_array_add(json_annotations, json_annotation);
            in_postprocess_mutex.unlock();
        }
    }
    else if(interpreter->is_maccel_output(cur_batch)){
        std::vector<DetResult> results;
        maccel_post(interpreter, model_mode, results, true, cur_batch);

        // Output of inference
        switch(model_mode){
            case 4: //yolo
            {
               
                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;

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

                    //std::cout << i <<  ": output_classes: " << id << ", output_scores: " << score << ", output_locations: [" << xmin << "," << ymin << "," << xmax << ","<< ymax << "]\n";

                    // Write json annotations
                    json_object * json_annotation = json_object_new_object();

                    json_object * json_bbox = json_object_new_array();

                    json_object_array_add(json_bbox, json_object_new_int(round(xmin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(ymin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(width)));
                    json_object_array_add(json_bbox, json_object_new_int(round(height)));

                    json_object_object_add(json_annotation, "image_id", json_object_new_int(image_ids[cur_batch]));
                    json_object_object_add(json_annotation, "bbox", json_bbox);
                    json_object_object_add(json_annotation, "category_id", json_object_new_int(id + 1));
                    json_object_object_add(json_annotation, "score", json_object_new_double(score));

                    in_postprocess_mutex.lock();
                    json_object_array_add(json_annotations, json_annotation);
                    in_postprocess_mutex.unlock();
                }

                break;
            }
        }
    }

    auto postprocess_elapsed = std::chrono::high_resolution_clock::now() - invoke_start;
    in_postprocess_mutex.lock();
    sum_postprocess_time += std::chrono::duration_cast<std::chrono::milliseconds>(postprocess_elapsed).count();
    num_postprocess++;
    in_postprocess_mutex.unlock();
}

void infer(tflite::Interpreter * interpreter, int model_mode, char * directory_path, json_object * json_images, json_object * json_annotations, int batch_size){
    for(int i = 0; i < interpreter->inputs().size(); i++){
        // Get the input tensor size info
        TfLiteTensor* input_tensor_i = interpreter->input_tensor(i);
        TfLiteIntArray* input_dims = input_tensor_i->dims;
        int input_height = input_dims->data[1];
        int input_width = input_dims->data[2];
        int input_channel = input_dims->data[3];

        // Resize input tensor
        int input_tensor_idx = interpreter->inputs()[i];
        if(interpreter->ResizeInputTensor(input_tensor_idx, {batch_size, input_height, input_width, input_channel}) != kTfLiteOk){
            std::cerr << "ERROR: Input resize failed.\n";
            exit(-1);
        }
    }
    
    if(interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "ERROR: Memory allocation for interpreter failed.\n";
        exit(-1);
    }

    // Find images in the directory
    DIR * dir = opendir(directory_path);
    if(dir == NULL){
        std::cerr << "ERROR: Cannot open the directory.\n";
        exit(-1);
    }

    struct dirent * ent;

    int thread_num = 4;

    int image_id = 0;
    while(true){
        int cur_batch = 0;
        std::vector<int> img_heights;
        std::vector<int> img_widths;
        std::vector<int> image_ids;
        img_heights.resize(batch_size);
        img_widths.resize(batch_size);
        image_ids.resize(batch_size);

        preprocess_start = std::chrono::high_resolution_clock::now();

        while(cur_batch < batch_size){
            std::vector<std::thread> threads;
            int cur_thread_num = 0;
            while(cur_thread_num < thread_num && cur_batch + cur_thread_num < batch_size){
                ent = readdir(dir);

                if(ent == NULL)
                    break;

                const char * filename = ent->d_name;               
                if(strstr(filename, ".jpg")){
                    std::cout << "Detecting " << filename << "..\r";
                    std::cout.flush();
                    image_id++;
                    image_ids[cur_batch + cur_thread_num] = image_id;
                    threads.push_back(std::thread(preprocess_thread, interpreter, model_mode, std::string(filename), json_images, cur_batch + cur_thread_num, std::ref(img_heights), std::ref(img_widths), image_id));
                    cur_thread_num++;
                }
            }

            for(int i = 0; i < cur_thread_num; i++){
                threads[i].join();
            }

            cur_batch += cur_thread_num;

            if(ent == NULL)
                break;
        }

        if(cur_batch == 0)
            break;
            
        invoke_start = std::chrono::high_resolution_clock::now();

        // Invoke
        if(interpreter->Invoke() != kTfLiteOk){
            std::cerr << "ERROR: Model execute failed\n";
            exit(-1);
        }

        int cur_thread_num = 0;
        for(int i = 0; i < cur_batch; i += cur_thread_num){
            std::vector<std::thread> threads;
    
            cur_thread_num = 0;
            for(int j = 0; j < thread_num && i + j < cur_batch; j++){
                threads.push_back(std::thread(postprocess_thread, interpreter, model_mode, std::ref(img_heights), std::ref(img_widths), std::ref(image_ids), json_annotations, i + j));
                cur_thread_num++;
            }

            for(int j = 0; j < cur_thread_num; j++){
                threads[j].join();
            }
        }

        max_turnaround += interpreter->GetMaxTurnAroundTime();
        sum_turnaround += interpreter->GetSumTurnAroundTime();
        num_turnaround++;
    }

    closedir(dir);
}

bool run_image(tflite::Interpreter * interpreter, int model_mode, std::vector<std::string> * labels_arg, char * directory_path, char * result_path, int batch_size, std::vector<float> perfs, std::vector<float> ori_score_thrs, std::vector<float> new_score_thrs){
    std::cout << "batch size: " << batch_size << ", perfs:";
    for(int i = 0; i < perfs.size(); i++)
        std::cout << " " << perfs[i];
    std::cout << ", ori_score_thrs:";
    for(int i = 0; i < ori_score_thrs.size(); i++)
        std::cout << " " << ori_score_thrs[i];
    std::cout << ", new_score_thrs:";
    for(int i = 0; i < new_score_thrs.size(); i++)
        std::cout << " " << new_score_thrs[i];
    std::cout << std::endl;
    
    std::vector<std::string> & labels = *labels_arg;

    auto application_start = std::chrono::high_resolution_clock::now();

    // Create json object
    json_object * json_result = json_object_new_object();
    json_object * json_categories = json_object_new_array();
    json_object * json_images = json_object_new_array();
    json_object * json_annotations = json_object_new_array();

    // Write json categories from label
    for(int i = 0; i < labels.size(); i++){
        if(labels[i] == "???" || labels[i] == "")
            continue;

        json_object * json_category = json_object_new_object();

        json_object_object_add(json_category, "id", json_object_new_int(i + 1));
        json_object_object_add(json_category, "name", json_object_new_string(labels[i].c_str()));

        json_object_array_add(json_categories, json_category);
    }

    char current_dir[500];
    getcwd(current_dir, 500);
    chdir(directory_path);

    interpreter->SetSchedulerParams(perfs);
    interpreter->SetPostProcessParams(ori_score_thrs, new_score_thrs);

    infer(interpreter, model_mode, directory_path, json_images, json_annotations, batch_size);

    auto application_elapsed = std::chrono::high_resolution_clock::now() - application_start;
    auto application_latency = std::chrono::duration_cast<std::chrono::milliseconds>(application_elapsed).count();

    std::cout << std::fixed;
    std::cout.precision(3);
    std::cout << "\nAverage Turnaround time:\t" << sum_turnaround / num_preproces << " ms\n";
    std::cout << "Average preprocess time:\t" << sum_preprocess_time / num_preproces << " ms, Num preprocess: " << num_preproces << "\n";
    std::cout << "Average turnaround + postprocess time:\t" << sum_postprocess_time / num_postprocess << " ms, Num postprocess: " << num_postprocess << "\n";
    std::cout << "Maximum Turnaround time:\t" << max_turnaround / num_turnaround << " ms, Num turnaroud: " << num_turnaround << "\n";
    std::cout << "Application latency:\t" << application_latency / 1000.0 << " s\n";
    std::cout << "pkshinresult " << sum_turnaround / num_preproces << " " << sum_preprocess_time / num_preproces << " " << sum_postprocess_time / num_postprocess << " " << max_turnaround / num_turnaround << " " << application_latency / 1000.0 << std::endl << std::endl;

    // Write json file
    chdir(current_dir);

    json_object_object_add(json_result, "categories", json_categories);
    json_object_object_add(json_result, "images", json_images);
    json_object_object_add(json_result, "annotations", json_annotations);

    json_object_to_file(result_path, json_result);

    json_object_put(json_result);

    return true;
}