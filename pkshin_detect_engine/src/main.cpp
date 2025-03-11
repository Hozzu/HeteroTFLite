#include <iostream>
#include <cstring>
#include <cstddef>
#include <string>
#include <vector>
#include <fstream>

#include <engine_interface.hpp>

//bool run_qcarcam(tflite::Interpreter * interpreter, int model_mode, std::vector<std::string> * labels, char * display_path);
bool run_image(tflite::Interpreter * interpreter, int model_mode, std::vector<std::string> * labels, char * directory_path, char * result_path, int batch_size, std::vector<float> perfs, std::vector<float> ori_score_thrs, std::vector<float> new_score_thrs);

int main(int argc, char * argv[]){
    int model_mode; // 1 for ssd_mobilenet
                    // 2 for efficientdet
                    // 3 for efficientdet-lite
                    // 4 for yolo
                    // 5 for yolov10
                    // 6 for yolo obb
    std::vector<std::string> labels;

    //usage guide
    if(!argv[1] || strcmp(argv[1], "-help") == 0 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "--h") == 0){
        std::cout << "Usage: pkshin_detect camera [MODEL] [LABEL] [DISPLAY] [ACCELERATOR]\n";
        std::cout << "camera mode runs the object detection using qcarcam API.\n";
        std::cout << "[MODEL] is path of the model file.\n";
        std::cout << "[LABEL] is path of the label file.\n";
        std::cout << "[DISPLAY] is path of the file defining the display setting.\n";
        std::cout << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, NPU is supported. Default value is CPU.\n\n";
        std::cout << "Usage: pkshin_detect image [MODEL] [LABEL] [IMG_DIR] [RESULT] [ACCELERATOR]\n";
        std::cout << "image mode runs the object detection with jpeg images.\n";
        std::cout << "[MODEL] is path of the model file.\n";
        std::cout << "[LABEL] is path of the label file.\n";
        std::cout << "[IMG_DIR] is path of the directory containing images.\n";
        std::cout << "[RESULT] is path of the result json file.\n";
        std::cout << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, NPU is supported. Default value is CPU.\n\n";
        return true;
    }

    // Argument error checking
    if( (strcmp(argv[1], "camera") != 0 && strcmp(argv[1], "image") != 0) || (strcmp(argv[1], "camera") == 0 && argc < 5) || (strcmp(argv[1], "image") == 0 && argc < 6) ){
        std::cerr << "ERROR: The first argument must be camera or image. camera mode requires at least 3 more arguments and image mode requires at least 4 more arguments\n\n";
        std::cerr << "Usage: pkshin_detect camera [MODEL] [LABEL] [DISPLAY] [ACCELERATOR]\n";
        std::cerr << "camera mode runs the object detection using qcarcam API.\n";
        std::cerr << "[MODEL] is path of the model file.\n";
        std::cerr << "[LABEL] is path of the label file.\n";
        std::cerr << "[DISPLAY] is path of the file defining the display setting.\n";
        std::cerr << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, NPU is supported. Default value is CPU.\n\n";
        std::cerr << "Usage: pkshin_detect image [MODEL] [LABEL] [IMG_DIR] [RESULT] [ACCELERATOR]\n";
        std::cerr << "image mode runs the object detection with jpeg images.\n";
        std::cerr << "[MODEL] is path of the model file.\n";
        std::cerr << "[LABEL] is path of the label file.\n";
        std::cerr << "[IMG_DIR] is path of the directory containing images.\n";
        std::cerr << "[RESULT] is path of the result json file.\n";
        std::cerr << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, NPU is supported. Default value is CPU.\n\n";
        return false;
    }

    // Load the model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(argv[2]);
    if(model == NULL){
        std::cerr << "ERROR: Model load failed. Check the model name.\n";
        return false;
    }

    // Build the interpreter
    //tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    if(interpreter == NULL){
        std::cerr << "ERROR: Interpreter build failed.\n";
        return false;
    }

    interpreter->SetNumThreads(1);
    std::cout << "INFO: Interperter uses 1 thread.\n";

    // Set the delegate
    if( (strcmp(argv[1], "camera") == 0 && argc == 5) || (strcmp(argv[1], "image") == 0 && argc == 6) || (strcmp(argv[1], "camera") == 0 && (strcmp(argv[5], "CPU") == 0 || strcmp(argv[5], "cpu") == 0)) || (strcmp(argv[1], "image") == 0 && (strcmp(argv[6], "CPU") == 0 || strcmp(argv[6], "cpu") == 0)) ){
        std::cout << "INFO: Run with CPU only.\n";
    }
    else if( (strcmp(argv[1], "camera") == 0 && (strcmp(argv[5], "GPU") == 0 || strcmp(argv[5], "gpu") == 0)) || (strcmp(argv[1], "image") == 0 && (strcmp(argv[6], "GPU") == 0 || strcmp(argv[6], "gpu") == 0)) ){
        TfLiteGpuDelegateOptionsV2 gpu_delegate_options = TfLiteGpuDelegateOptionsV2Default();
	gpu_delegate_options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        gpu_delegate_options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
        gpu_delegate_options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
        gpu_delegate_options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;

        auto * gpu_delegate_ptr = TfLiteGpuDelegateV2Create(&gpu_delegate_options);
        if(gpu_delegate_ptr == NULL){
            std::cout << "WARNING: Cannot create gpu delegate. Run without gpu delegate.\n";
            TfLiteGpuDelegateV2Delete(gpu_delegate_ptr);
        }
        else{
            tflite::Interpreter::TfLiteDelegatePtr gpu_delegate(gpu_delegate_ptr, &TfLiteGpuDelegateV2Delete);

            std::cout << "INFO: Run with gpu delegate.\n";
            if (interpreter->ModifyGraphWithDelegate(gpu_delegate.get()) != kTfLiteOk){
                std::cerr << "ERROR: Cannot convert model with gpu delegate\n";
                return false;
            }
        }
    }
    else if( (strcmp(argv[1], "camera") == 0 && (strcmp(argv[5], "NPU") == 0 || strcmp(argv[5], "npu") == 0)) || (strcmp(argv[1], "image") == 0 && (strcmp(argv[6], "NPU") == 0 || strcmp(argv[6], "npu") == 0)) ){
        TfLiteHexagonInitWithPath("/usr/lib");

        TfLiteHexagonDelegateOptions npu_delegate_params = {0};
        auto* npu_delegate_ptr = TfLiteHexagonDelegateCreate(&npu_delegate_params);
        if (npu_delegate_ptr == NULL) {
            TfLiteHexagonDelegateDelete(npu_delegate_ptr);
            std::cout << "WARNING: Cannot create hexagon delegate. Check whether the hexagon library is in /usr/lib/. Run without hexagon delegate.\n";
        }
        else{
            tflite::Interpreter::TfLiteDelegatePtr npu_delegate(npu_delegate_ptr, &TfLiteHexagonDelegateDelete);

            std::cout << "INFO: Run with hexagon delegate.\n";
            if(interpreter->ModifyGraphWithDelegate(npu_delegate.get()) != kTfLiteOk){
                std::cerr << "ERROR: Cannot convert model with hexagon delegate\n";
                return false;
            }
        }
    }
    else{
        std::cout << "WARNING: [ACCELERATOR] should be CPU, GPU, or NPU.\n";
    }


    if(interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "ERROR: Memory allocation for interpreter failed.\n";
        return false;
    }

    // Parse the model info
    if(strstr(argv[2], "mobilenet")){
        if(strstr(argv[2], "ssd")){
            std::cout << "INFO: Model file: ssd_mobilenet\n";
            model_mode = 1;
        }
    }
    else if(strstr(argv[2], "efficientdet")){
        if(strstr(argv[2], "lite")){
            std::cout << "INFO: Model file: efficientdet lite\n";
            model_mode = 3;
        }
        else{
            std::cout << "INFO: Model file: efficientdet\n";
            model_mode = 2;
        }
    }
    else if(strstr(argv[2], "yolo")){
        if(strstr(argv[2], "v3")){
            std::cout << "INFO: Model file: yolov3\n";
            model_mode = 4;
        }
        else if(strstr(argv[2], "v5")){
            std::cout << "INFO: Model file: yolov5\n";
            model_mode = 4;
        }
        else if(strstr(argv[2], "v8")){
            if (strstr(argv[2], "obb")) {
                std::cout << "INFO: Model file: yolov8 obb\n";
                model_mode = 6;
            } else {
                std::cout << "INFO: Model file: yolov8\n";
                model_mode = 4;
            }
        }
        else if(strstr(argv[2], "v9")){
            std::cout << "INFO: Model file: yolov9\n";
            model_mode = 4;
        }
        else if(strstr(argv[2], "v10")){
            std::cout << "INFO: Model file: yolov10\n";
            model_mode = 5;
        }
        else if(strstr(argv[2], "v11")){
            if (strstr(argv[2], "obb")) {
                std::cout << "INFO: Model file: yolov11 obb\n";
                model_mode = 6;
            } else {
                std::cout << "INFO: Model file: yolov11\n";
                model_mode = 4;
            }
        }
    }
    else{
        std::cerr << "ERROR: Check the model file name. Currently, ssd mobilenet, efficientdet, efficientdet lite, yolov3, yolov5, yolov8, yolov8 obb, yolov9, yolov10, yolov11, yolov11 obb are only supported\n";
        return false;
    }

    // Print input tensors
    for(int i = 0; i < interpreter->inputs().size(); i++){
        std::string input_dims_str("[");
        input_dims_str += std::to_string(interpreter->input_tensor(i)->dims->data[0]);
        for(int j = 1; j < interpreter->input_tensor(i)->dims->size; j++){
            input_dims_str += std::string("x");
            input_dims_str += std::to_string(interpreter->input_tensor(i)->dims->data[j]);
        }
        input_dims_str += std::string("]");

        std::cout << "INFO: Graph input " << i << ": " << interpreter->GetInputName(i) << " (" << TfLiteTypeGetName(interpreter->input_tensor(i)->type) << input_dims_str << ")" << std::endl;
    }

    // Print output tensors
    for(int i = 0; i < interpreter->outputs().size(); i++){
        std::string output_dims_str("[");
        output_dims_str += std::to_string(interpreter->output_tensor(i)->dims->data[0]);
        for(int j = 1; j < interpreter->output_tensor(i)->dims->size; j++){
            output_dims_str += std::string("x");
            output_dims_str += std::to_string(interpreter->output_tensor(i)->dims->data[j]);
        }
        output_dims_str += std::string("]");

        std::cout << "INFO: Graph output " << i << ": " << interpreter->GetOutputName(i) << " (" << TfLiteTypeGetName(interpreter->output_tensor(i)->type) << output_dims_str << ")" << std::endl;
    }

    // Parse the label
    std::ifstream labelfile(argv[3]);
    if(!labelfile.is_open()){
        std::cerr << "ERROR: Cannot open the label file.\n";
        return false;
    }

    while(true){
        std::string line_string;

        if(std::getline(labelfile, line_string))
            labels.push_back(line_string);
        else
            break;
    }

    labelfile.close();

    // Call the appropriate functons for mode
    if(strcmp(argv[1], "camera") == 0){
        std::cout << "INFO: Running the object detection using qcarcam API.\n";
        //return run_qcarcam(interpreter.get(), model_mode, &labels, argv[4]);
    }
    else if(strcmp(argv[1], "image") == 0){
        std::cout << "INFO: Running the object detection with jpeg images.\n";

        char perfs_str[100];
        strcpy(perfs_str, argv[8]);
        char *ret_token = NULL;
        std::vector<float> perfs;

        ret_token = strtok(perfs_str, ",");
        while(ret_token != NULL) {
            perfs.push_back(atof(ret_token));
            ret_token = strtok(NULL, ",");
        }

        char scores_str[100];
        strcpy(scores_str, argv[9]);
        std::vector<float> ori_score_thrs;
        std::vector<float> new_score_thrs;

        ret_token = strtok(scores_str, ",");
        while(ret_token != NULL) {
            ori_score_thrs.push_back(atof(ret_token));
            ret_token = strtok(NULL, ",");
            new_score_thrs.push_back(atof(ret_token));
            ret_token = strtok(NULL, ",");
        }

        return run_image(interpreter.get(), model_mode, &labels, argv[4], argv[5], atoi(argv[7]), perfs, ori_score_thrs, new_score_thrs);
    }
}
