#ifndef _MACCELPOST_HPP_
#define _MACCELPOST_HPP_

#include <vector>
#include <cmath>

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

static std::vector<std::vector<int>> generate_grids(int input_h, int input_w, std::vector<int> strides){

	std::vector<std::vector<int>> all_grids;
	for(int i = 0; i < strides.size(); i++){
		int grid_h = input_h  / strides[i];
		int grid_w = input_w / strides[i];
		int grid_size =  grid_h * grid_w * 2;

		std::vector<int> grids;
		for(int j = 0; j < grid_size; j++){
			if(j % 2 == 0){
				grids.push_back(((int) j / 2) % grid_w);
			} else{
				grids.push_back(((int) j / 2) / grid_w);
			}
		}

		all_grids.push_back(grids);
	}
	return all_grids;
}

static float softmax_inplace_idx(const std::vector<float> &output, int start_idx, int end_idx){
	
	float sum = 0, result = 0;
	for (int i = start_idx; i < end_idx; i++){
		sum += exp(output[i]);
	}
	for (int i = start_idx; i < end_idx; i++){
		result += exp(output[i]) / sum * (i - start_idx);
	}
	return result;
}

static bool maccel_post(tflite::Interpreter * interpreter, int model_mode, std::vector<DetResult> & results, bool multi, int cur_batch = -1){
    int max_detections = 300;
    float conf_threshold = interpreter->GetPostProcessParams()[0] == -1 ? 0.001 : interpreter->GetPostProcessParams()[0];
    float iou_threshold = 0.7;
    bool multi_label = multi; // If true, nms is done per class. slow but accurate.

    float inverse_conf_thres = -log(1 / conf_threshold - 1);

    TfLiteTensor* input_tensor_0 = interpreter->input_tensor(0);
    TfLiteIntArray* input_dims = input_tensor_0->dims;
    int input_height = input_dims->data[1];
    int input_width = input_dims->data[2];
    std::vector<int> strides = {8, 16, 32};
    std::vector<std::vector<int>> grids = generate_grids(input_height, input_width, strides);

    std::vector<int> class_ids;
    std::vector<float> scores;
    std::vector<cv::Rect2d> boxes;

    for(int i = 0; i < 3; i++){
        float * output_ptr;
        if(cur_batch == -1){
            output_ptr = interpreter->typed_output_tensor<float>(i);
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

            // Get the output tensor size info
            TfLiteTensor* output_tensor_i = interpreter->output_tensor(tensor_index);
            TfLiteIntArray* output_dims = output_tensor_i->dims;

            int size = cur_batch;
            for(int j = 1; j < output_dims->size; j++)
                size *= output_dims->data[j];

            output_ptr = interpreter->typed_output_tensor<float>(tensor_index) + size;
        }

        int grid_h = input_height / strides[i];
	    int grid_w = input_width / strides[i];

        for(int j = 0; j < grid_h * grid_w; j++){
            std::array<float, 4> pred_box = {-1, -1, -1, -1};
            float max_score = 0;
            int max_class_id = 0;

            for(int k = 0; k < 80; k++){
                if(output_ptr[j * 144 + 64 + k] > inverse_conf_thres){
                    float score = 1 / (1 + exp(-output_ptr[j * 144 + 64 + k]));

                    std::array<float, 4> box;
                    if(pred_box[0] == -1){
                        for (int l = 0; l < 4; l++){
                            float sum = 0;
                            float box_value = 0;

                            for (int m = 0; m < 16; m++) {
                                sum += exp(output_ptr[j * 144 + l * 16 + m]);
                            }

                            for (int m = 0; m < 16; m++) {
                                box_value += exp(output_ptr[j * 144 + l * 16 + m]) / sum * m;
                            }

                            box[l] = box_value;
                        }
                    }

                    float xmin = grids[i][j * 2 + 0] - box[0] + 0.5;
                    float ymin = grids[i][j * 2 + 1] - box[1] + 0.5;
                    float xmax = grids[i][j * 2 + 0] + box[2] + 0.5;
                    float ymax = grids[i][j * 2 + 1] + box[3] + 0.5;

                    pred_box = {xmin * strides[i], ymin * strides[i], xmax * strides[i], ymax * strides[i]};

                    if(score > max_score){
                        max_score = score;
                        max_class_id = k;
                    }

                    if(multi_label){
                        scores.push_back(score);
                        class_ids.push_back(k);
                        boxes.push_back(cv::Rect2d(pred_box[0], pred_box[1], pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]));
                    }
                }
            }

            if(!multi_label){
                scores.push_back(max_score);
                class_ids.push_back(max_class_id);
                boxes.push_back(cv::Rect2d(pred_box[0], pred_box[1], pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]));
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
        DetResult result;
        int idx = nms_result[i];

        result.score =  scores[idx];
        result.xmin = boxes[idx].x;
        result.ymin = boxes[idx].y;
        result.xmax = boxes[idx].x + boxes[idx].width;
        result.ymax = boxes[idx].y + boxes[idx].height;
        result.id =  class_ids[idx];

        results.push_back(result);
    }

    return true;
}

#endif // _MACCELPOST_HPP_