#include "engine.hpp"

namespace pkshin{
    template<typename TO, typename FROM>
    std::unique_ptr<TO> static_unique_pointer_cast (std::unique_ptr<FROM>&& old){
        return std::unique_ptr<TO>{static_cast<TO*>(old.release())};
    }

    hailort::Expected<std::shared_ptr<hailort::ConfiguredNetworkGroup>> configure_network_group(hailort::VDevice &vdevice, hailort::Hef &hef, uint16_t batch_size){
        auto configure_params = hef.create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
        if (!configure_params) {
            return hailort::make_unexpected(configure_params.status());
        }

        auto network_groups = vdevice.configure(hef, configure_params.value());
        if (!network_groups) {
            return hailort::make_unexpected(network_groups.status());
        }

        if (1 != network_groups->size()) {
            std::cerr << "Invalid amount of network groups" << std::endl;
            return hailort::make_unexpected(HAILO_INTERNAL_FAILURE);
        }

        return std::move(network_groups->at(0));
    }
}

using namespace pkshin;

namespace tflite{
    namespace pkshin{
        FlatBufferModel::FlatBufferModel() : ::tflite::FlatBufferModel(NULL){

        }

        std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromFile(const char* filename, ::tflite::ErrorReporter* error_reporter){
            //std::cout << "BuildFromFile start\n";

            size_t len = strlen(filename);

            char suffix1[10] = ".tflite";
            size_t suffix1len = strlen(suffix1);

            char suffix2[10] = ".mxq";
            size_t suffix2len = strlen(suffix2);

            char suffix3[10] = ".hef";
            size_t suffix3len = strlen(suffix3);

            char suffix4[20] = ".tflitemxq";
            size_t suffix4len = strlen(suffix4);

            char suffix5[20] = ".tflitehef";
            size_t suffix5len = strlen(suffix5);

            if(suffix1len <= len && strncmp(filename + len - suffix1len, suffix1, suffix1len) == 0){
                mode_= 0;
                std::cout << "INFO: tflite model file detected\n";

                std::unique_ptr<::tflite::FlatBufferModel> oldTypeFlatBufferModel = ::tflite::FlatBufferModel::BuildFromFile(filename);

                return static_unique_pointer_cast<FlatBufferModel, ::tflite::FlatBufferModel>(std::move(oldTypeFlatBufferModel));
            }
            else if(suffix2len <= len && strncmp(filename + len - suffix2len, suffix2, suffix2len) == 0){
                mode_= 1;
                std::cout << "INFO: mobilint model file detected\n";

                strncpy(filename_, filename, len);

                return std::make_unique<FlatBufferModel>();
            }
            else if(suffix3len <= len && strncmp(filename + len - suffix3len, suffix3, suffix3len) == 0) {
                mode_= 2;
                std::cout << "INFO: hailo model file detected\n";

                strncpy(filename_, filename, len);

                return std::make_unique<FlatBufferModel>();
            }
            else if(suffix4len <= len && strncmp(filename + len - suffix4len, suffix4, suffix4len) == 0) {
                mode_= 3;
                std::cout << "INFO: tflite+maccel detected\n";

                strncpy(filename_, filename, len - suffix4len);
                strcat(filename_, ".mxq");

                strncpy(tflite_filename_, filename, len - 3);

                return std::make_unique<FlatBufferModel>();
            }
            else if(suffix5len <= len && strncmp(filename + len - suffix5len, suffix5, suffix5len) == 0) {
                mode_= 4;
                std::cout << "INFO: tflite+hailo detected\n";

                strncpy(filename_, filename, len - suffix5len);
                strcat(filename_, ".hef");

                strncpy(tflite_filename_, filename, len - 3);

                return std::make_unique<FlatBufferModel>();
            }
            else{
                std::cerr << "ERROR: model file is invalid\n";
                return NULL;
            }
        }
    }

    namespace ops {
        namespace builtin {
            namespace pkshin{
                BuiltinOpResolver::BuiltinOpResolver() : ::tflite::ops::builtin::BuiltinOpResolver(){
                    //std::cout << "BuiltinOpResolver constructor\n";
                }

                BuiltinOpResolverWithoutDefaultDelegates::BuiltinOpResolverWithoutDefaultDelegates() 
                : ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates() {
                    //std::cout << "BuiltinOpResolverWithoutDefaultDelegates constructor\n";
                }
            }
        }
    }

    namespace pkshin{
        void Invoke_gpu_queue();
        void Invoke_hexagon_queue();
        void Invoke_maccel_queue();
        void Invoke_hailo_queue();
        void Invoke_hailo_queue2();
        void Invoke_hailo_queue3();
        void Invoke_thread();

        Interpreter::Interpreter(::tflite::ErrorReporter* error_reporter) : ::tflite::Interpreter(error_reporter){
            //std::cout << "Interpreter Constructor\n";

            switch(mode_){
                case 0:
                {
                    break;
                }
                case 1:
                case 3:
                {
                    mobilint::StatusCode sc;

                    static auto acc = mobilint::Accelerator::create(sc);
                    if (!sc) {
                        std::cerr << "ERROR: Failed to open device. status code: " << int(sc) << std::endl;
                        std::cerr << "Hint: Please make sure that you have proper privilege.\n";
                        std::cerr << "Hint: Please make sure that the driver is correctly loaded.\n";
                        exit(-1);
                    }

                    static auto model = mobilint::Model::create(filename_, sc);
                    if (!sc) {
                        std::cerr << "ERROR: Failed to create a model. status code: " << int(sc) << std::endl;
                        exit(-1);
                    }

                    sc = model->launch(*acc);
                    if (!sc) {
                        std::cerr << "ERROR: Failed to launch a model. status code: " << int(sc) << std::endl;
                        exit(-1);
                    }

                    mobilintModel_ = model.get();

                    break;
                }
                case 2:
                {
                    static auto scan_res = hailort::Device::scan();
                    if (!scan_res) {
                        std::cerr << "ERROR: Failed to scan, status = " << scan_res.status() << std::endl;
                        exit(-1);
                    }
                    std::cout << "INFO: Found " << scan_res.value().size() << " hailo accelerators" << std::endl;

                    static hailo_vdevice_params_t params;
                    auto status = hailo_init_vdevice_params(&params);
                    if (HAILO_SUCCESS != status) {
                        std::cerr << "ERROR: Failed init vdevice_params, status = " << status << std::endl;
                        exit(-1);
                    }

                    params.device_count = static_cast<uint32_t>(scan_res->size());
                    static auto vdevice = hailort::VDevice::create(params);
                    if (!vdevice) {
                        std::cerr << "ERROR: Failed create vdevice, status = " << vdevice.status() << std::endl;
                        exit(-1);
                    }
                    
                    static auto hef = hailort::Hef::create(filename_);
                    if (!hef) {
                        std::cerr << "ERROR: Failed to create hef: " << filename_ << ", status = " << hef.status() << std::endl;
                        exit(-1);
                    }

                    static auto network_group = configure_network_group(*vdevice.value(), hef.value(), 1);
                    if (!network_group) {
                        std::cerr << "ERROR: Failed to configure network group" << std::endl;
                        exit(-1);
                    }

                    static auto vstreams = hailort::VStreamsBuilder::create_vstreams(*network_group.value(), {}, HAILO_FORMAT_TYPE_AUTO);
                    if (!vstreams) {
                        std::cerr << "ERROR: Failed creating vstreams " << vstreams.status() << std::endl;
                        exit(-1);
                    }

                    hailoVstreams_ = &vstreams.value();

                    break;
                }
                case 4:
                {
                    static auto scan_res = hailort::Device::scan();
                    if (!scan_res) {
                        std::cerr << "ERROR: Failed to scan, status = " << scan_res.status() << std::endl;
                        exit(-1);
                    }
                    std::cout << "INFO: Found " << scan_res.value().size() << " hailo device ids" << std::endl;

                    static auto vdevice1 = hailort::VDevice::create(std::vector<std::string>({scan_res.value()[0]}));
                    if (!vdevice1) {
                        std::cerr << "ERROR: Failed create vdevice, status = " << vdevice1.status() << std::endl;
                        exit(-1);
                    }
                    
                    static auto hef = hailort::Hef::create(filename_);
                    if (!hef) {
                        std::cerr << "ERROR: Failed to create hef: " << filename_ << ", status = " << hef.status() << std::endl;
                        exit(-1);
                    }

                    static auto network_group1 = configure_network_group(*vdevice1.value(), hef.value(), 1);
                    if (!network_group1) {
                        std::cerr << "ERROR: Failed to configure network group" << std::endl;
                        exit(-1);
                    }

                    static auto vstreams1 = hailort::VStreamsBuilder::create_vstreams(*network_group1.value(), {}, HAILO_FORMAT_TYPE_AUTO);
                    if (!vstreams1) {
                        std::cerr << "ERROR: Failed creating vstreams " << vstreams1.status() << std::endl;
                        exit(-1);
                    }

                    hailoVstreams_ = &vstreams1.value();

                    static auto vdevice2 = hailort::VDevice::create(std::vector<std::string>({scan_res.value()[1]}));
                    if (!vdevice2) {
                        std::cerr << "ERROR: Failed create vdevice, status = " << vdevice2.status() << std::endl;
                        exit(-1);
                    }

                    static auto network_group2 = configure_network_group(*vdevice2.value(), hef.value(), 1);
                    if (!network_group2) {
                        std::cerr << "ERROR: Failed to configure network group" << std::endl;
                        exit(-1);
                    }

                    static auto vstreams2 = hailort::VStreamsBuilder::create_vstreams(*network_group2.value(), {}, HAILO_FORMAT_TYPE_AUTO);
                    if (!vstreams2) {
                        std::cerr << "ERROR: Failed creating vstreams " << vstreams2.status() << std::endl;
                        exit(-1);
                    }

                    hailoVstreams2_ = &vstreams2.value();

                    static auto vdevice3 = hailort::VDevice::create(std::vector<std::string>({scan_res.value()[2]}));
                    if (!vdevice3) {
                        std::cerr << "ERROR: Failed create vdevice, status = " << vdevice3.status() << std::endl;
                        exit(-1);
                    }

                    static auto network_group3 = configure_network_group(*vdevice3.value(), hef.value(), 1);
                    if (!network_group3) {
                        std::cerr << "ERROR: Failed to configure network group" << std::endl;
                        exit(-1);
                    }

                    static auto vstreams3 = hailort::VStreamsBuilder::create_vstreams(*network_group3.value(), {}, HAILO_FORMAT_TYPE_AUTO);
                    if (!vstreams3) {
                        std::cerr << "ERROR: Failed creating vstreams " << vstreams3.status() << std::endl;
                        exit(-1);
                    }

                    hailoVstreams3_ = &vstreams3.value();
                    
                    break;
                }
            }

            switch(mode_){
                case 3:
                case 4:
                {
                    gpu_thread_ = std::thread(Invoke_gpu_queue);

                    std::unique_lock<std::mutex> lk(gpu_thread_mutex_);
                    gpu_thread_set_cv_.wait(lk);
                    gpu_thread_mutex_.unlock();

                    static std::unique_ptr<::tflite::FlatBufferModel> model = ::tflite::FlatBufferModel::BuildFromFile(tflite_filename_);
                    if(model == NULL){
                        std::cerr << "ERROR: Model load failed. Check the model name.\n";
                        exit(-1);
                    }
                    
                    static ::tflite::ops::builtin::BuiltinOpResolver resolver;
                    static ::tflite::InterpreterBuilder builder(*model, resolver);
                    static std::unique_ptr<::tflite::Interpreter> interpreter;
                    builder(&interpreter);
                    if(interpreter == NULL){
                        std::cerr << "ERROR: Interpreter build failed.\n";
                        exit(-1);
                    }
                    
                    TfLiteHexagonInitWithPath("/usr/lib");

                    static TfLiteHexagonDelegateOptions npu_delegate_params = {0};
                    static auto * npu_delegate_ptr = TfLiteHexagonDelegateCreate(&npu_delegate_params);
                    if (npu_delegate_ptr == NULL) {
                        TfLiteHexagonDelegateDelete(npu_delegate_ptr);
                        std::cout << "WARNING: Cannot create hexagon delegate. Check whether the hexagon library is in /usr/lib/. Run without hexagon delegate.\n";
                        hexagonInterpreter_ = nullptr;
                    }
                    else{
                        static ::tflite::Interpreter::TfLiteDelegatePtr npu_delegate(npu_delegate_ptr, &TfLiteHexagonDelegateDelete);

                        if(interpreter->ModifyGraphWithDelegate(npu_delegate.get()) != kTfLiteOk){
                            std::cout << "WARNING: Cannot convert model with hexagon delegate. Run without hexagon delegate.\n";
                            hexagonInterpreter_ = nullptr;
                        }
                        else{
                            if(interpreter->AllocateTensors() != kTfLiteOk) {
                                std::cerr << "ERROR: Memory allocation for interpreter failed.\n";
                                exit(-1);
                            }
                            
                            hexagonInterpreter_ = interpreter.release();
                        }
                    }

                    break;
                }
                case 0:
                case 1:
                case 2:
                {
                    break;
                }
            }

            switch(mode_){
                case 0:
                {
                    break;
                }
                case 1:
                {
                    int input_size = mobilintModel_->getModelInputShape().size();
                    int output_size = mobilintModel_->getModelOutputShape().size();

                    inputs_ = std::vector<int>(input_size);
                    outputs_ = std::vector<int>(output_size);

                    input_dims_.resize(input_size);
                    input_tensors_.resize(input_size);
                    input_datas_.resize(input_size);
                    input_names_.resize(input_size);
                    output_dims_.resize(output_size);
                    output_tensors_.resize(output_size);
                    output_datas_.resize(output_size);
                    output_names_.resize(output_size);

                    for(int i = 0; i < input_size; i++){
                        auto input_shape_info = mobilintModel_->getModelInputShape()[i];

                        input_dims_[i] = (TfLiteIntArray *) malloc(sizeof(int) * 5);

                        if(input_shape_info[0] == 0){
                            input_dims_[i]->size = 0;
                        }
                        else if(input_shape_info[1] == 0){
                            input_dims_[i]->size = 2;
                            input_dims_[i]->data[0] = 1;
                            input_dims_[i]->data[1] = input_shape_info[0];

                            input_datas_[i] = malloc(sizeof(float) * input_shape_info[0]);
                        }
                        else if(input_shape_info[2] == 0){
                            input_dims_[i]->size = 3;
                            input_dims_[i]->data[0] = 1;
                            input_dims_[i]->data[1] = input_shape_info[0];
                            input_dims_[i]->data[2] = input_shape_info[1];

                            input_datas_[i] = malloc(sizeof(float) * input_shape_info[0] * input_shape_info[1]);
                        }
                        else{
                            input_dims_[i]->size = 4;
                            input_dims_[i]->data[0] = 1;
                            input_dims_[i]->data[1] = input_shape_info[0];
                            input_dims_[i]->data[2] = input_shape_info[1];
                            input_dims_[i]->data[3] = input_shape_info[2];

                            input_datas_[i] = malloc(sizeof(float) * input_shape_info[0] * input_shape_info[1] * input_shape_info[2]);
                        }

                        input_tensors_[i] = (TfLiteTensor *) malloc(sizeof(TfLiteTensor));

                        input_tensors_[i]->dims = input_dims_[i];
                        input_tensors_[i]->type = kTfLiteFloat32;
                        input_names_[i] = (char *) malloc(sizeof(char) * 2);
                        strcpy(input_names_[i], " ");
                    }

                    for(int i = 0; i < output_size; i++){
                        auto output_shape_info = mobilintModel_->getModelOutputShape()[i];

                        output_dims_[i] = (TfLiteIntArray *) malloc(sizeof(int) * 5);

                        if(output_shape_info[0] == 0){
                            output_dims_[i]->size = 0;
                        }
                        else if(output_shape_info[1] == 0){
                            output_dims_[i]->size = 2;
                            output_dims_[i]->data[0] = 1;
                            output_dims_[i]->data[1] = output_shape_info[0];

                            output_datas_[i] = malloc(sizeof(float) * output_shape_info[0]);
                        }
                        else if(output_shape_info[2] == 0){
                            output_dims_[i]->size = 3;
                            output_dims_[i]->data[0] = 1;
                            output_dims_[i]->data[1] = output_shape_info[0];
                            output_dims_[i]->data[2] = output_shape_info[1];

                            output_datas_[i] = malloc(sizeof(float) * output_shape_info[0] * output_shape_info[1]);
                        }
                        else{
                            output_dims_[i]->size = 4;
                            output_dims_[i]->data[0] = 1;
                            output_dims_[i]->data[1] = output_shape_info[0];
                            output_dims_[i]->data[2] = output_shape_info[1];
                            output_dims_[i]->data[3] = output_shape_info[2];

                            output_datas_[i] = malloc(sizeof(float) * output_shape_info[0] * output_shape_info[1] * output_shape_info[2]);
                        }

                        output_tensors_[i] = (TfLiteTensor *) malloc(sizeof(TfLiteTensor));

                        output_tensors_[i]->dims = output_dims_[i];
                        output_tensors_[i]->type = kTfLiteFloat32;
                        output_names_[i] = (char *) malloc(sizeof(char) * 2);
                        strcpy(output_names_[i], " ");
                    }

                    break;
                }
                case 2:
                {
                    int input_size = hailoVstreams_->first.size();
                    int output_size = hailoVstreams_->second.size();

                    inputs_ = std::vector<int>(input_size);
                    outputs_ = std::vector<int>(output_size);

                    input_dims_.resize(input_size);
                    input_tensors_.resize(input_size);
                    input_datas_.resize(input_size);
                    input_names_.resize(input_size);
                    output_dims_.resize(output_size);
                    output_tensors_.resize(output_size);
                    output_datas_.resize(output_size);
                    output_names_.resize(output_size);

                    for(int i = 0; i < input_size; i++){
                        int frame_size = hailoVstreams_->first[i].get_frame_size();
                        auto input_info = hailoVstreams_->first[i].get_info();
                        auto shape = input_info.shape;
                        auto format = input_info.format;
                        auto quant_info = input_info.quant_info;
                        auto name = input_info.name;

                        input_dims_[i] = (TfLiteIntArray *) malloc(sizeof(int) * 5);

                        if(shape.height == 0){
                            input_dims_[i]->size = 0;
                        }
                        else if(shape.width == 0){
                            input_dims_[i]->size = 2;
                            input_dims_[i]->data[0] = 1;
                            input_dims_[i]->data[1] = frame_size;
                        }
                        else if(shape.features == 0){
                            input_dims_[i]->size = 3;
                            input_dims_[i]->data[0] = 1;
                            input_dims_[i]->data[1] = shape.height;
                            input_dims_[i]->data[2] = frame_size / shape.height;
                        }
                        else{
                            input_dims_[i]->size = 4;
                            input_dims_[i]->data[0] = 1;
                            input_dims_[i]->data[1] = shape.height;
                            input_dims_[i]->data[2] = shape.width;
                            input_dims_[i]->data[3] = frame_size / shape.height / shape.width;
                        }

                        input_tensors_[i] = (TfLiteTensor *) malloc(sizeof(TfLiteTensor));

                        input_tensors_[i]->dims = input_dims_[i];

                        switch(format.type){
                            case HAILO_FORMAT_TYPE_AUTO:
                            {
                                input_tensors_[i]->type = kTfLiteVariant;
                                break;
                            }
                            case HAILO_FORMAT_TYPE_UINT8:
                            {
                                input_tensors_[i]->type = kTfLiteUInt8;

                                input_dims_[i]->data[input_dims_[i]->size - 1] /= sizeof(uint8_t);

                                int size = 1;
                                for(int j = 0; j < input_dims_[i]->size; j++)
                                    size *= input_dims_[i]->data[j];

                                input_datas_[i] = malloc(sizeof(uint8_t) * size);
                                break;
                            }
                            case HAILO_FORMAT_TYPE_UINT16:
                            {
                                input_tensors_[i]->type = kTfLiteUInt16;

                                input_dims_[i]->data[input_dims_[i]->size - 1] /= sizeof(uint16_t);

                                int size = 1;
                                for(int j = 0; j < input_dims_[i]->size; j++)
                                    size *= input_dims_[i]->data[j];

                                input_datas_[i] = malloc(sizeof(uint16_t) * size);
                                break;
                            }
                            case HAILO_FORMAT_TYPE_FLOAT32:
                            {
                                input_tensors_[i]->type = kTfLiteFloat32;

                                input_dims_[i]->data[input_dims_[i]->size - 1] /= sizeof(float32_t);

                                int size = 1;
                                for(int j = 0; j < input_dims_[i]->size; j++)
                                    size *= input_dims_[i]->data[j];

                                input_datas_[i] = malloc(sizeof(float32_t) * size);
                                break;
                            }
                        }
                        
                        TfLiteQuantizationParams params;
                        params.zero_point = quant_info.qp_zp;
                        params.scale = quant_info.qp_scale;
                        input_tensors_[i]->params = params;

                        input_names_[i] = (char *) malloc(sizeof(char) * strlen(name));
                        strncpy(input_names_[i], name, strlen(name));
                        input_tensors_[i]->name = input_names_[i];
                    }

                    for(int i = 0; i < output_size; i++){
                        int frame_size = hailoVstreams_->second[i].get_frame_size();
                        auto output_info = hailoVstreams_->second[i].get_info();
                        auto shape = output_info.shape;
                        auto format = output_info.format;
                        auto quant_info = output_info.quant_info;
                        auto name = output_info.name;

                        output_dims_[i] = (TfLiteIntArray *) malloc(sizeof(int) * 5);

                        if(shape.height == 0){
                            output_dims_[i]->size = 0;
                        }
                        else if(shape.width == 0){
                            output_dims_[i]->size = 2;
                            output_dims_[i]->data[0] = 1;
                            output_dims_[i]->data[1] = frame_size;
                        }
                        else if(shape.features == 0){
                            output_dims_[i]->size = 3;
                            output_dims_[i]->data[0] = 1;
                            output_dims_[i]->data[1] = shape.height;
                            output_dims_[i]->data[2] = frame_size / shape.height;
                        }
                        else{
                            output_dims_[i]->size = 4;
                            output_dims_[i]->data[0] = 1;
                            output_dims_[i]->data[1] = shape.height;
                            output_dims_[i]->data[2] = shape.width;
                            output_dims_[i]->data[3] = frame_size / shape.height / shape.width;
                        }

                        output_tensors_[i] = (TfLiteTensor *) malloc(sizeof(TfLiteTensor));

                        output_tensors_[i]->dims = output_dims_[i];

                        switch(format.type){
                            case HAILO_FORMAT_TYPE_AUTO:
                            {
                                output_tensors_[i]->type = kTfLiteVariant;
                                break;
                            }
                            case HAILO_FORMAT_TYPE_UINT8:
                            {
                                output_tensors_[i]->type = kTfLiteUInt8;

                                output_dims_[i]->data[output_dims_[i]->size - 1] /= sizeof(uint8_t);

                                int size = 1;
                                for(int j = 0; j < output_dims_[i]->size; j++)
                                    size *= output_dims_[i]->data[j];

                                output_datas_[i] = malloc(sizeof(uint8_t) * size);
                                break;
                            }
                            case HAILO_FORMAT_TYPE_UINT16:
                            {
                                output_tensors_[i]->type = kTfLiteUInt16;

                                output_dims_[i]->data[output_dims_[i]->size - 1] /= sizeof(uint16_t);

                                int size = 1;
                                for(int j = 0; j < output_dims_[i]->size; j++)
                                    size *= output_dims_[i]->data[j];

                                output_datas_[i] = malloc(sizeof(uint16_t) * size);
                                break;
                            }
                            case HAILO_FORMAT_TYPE_FLOAT32:
                            {
                                output_tensors_[i]->type = kTfLiteFloat32;

                                output_dims_[i]->data[output_dims_[i]->size - 1] /= sizeof(float32_t);

                                int size = 1;
                                for(int j = 0; j < output_dims_[i]->size; j++)
                                    size *= output_dims_[i]->data[j];

                                output_datas_[i] = malloc(sizeof(float32_t) * size);
                                break;
                            }
                        }

                        TfLiteQuantizationParams params;
                        params.zero_point = quant_info.qp_zp;
                        params.scale = quant_info.qp_scale;
                        output_tensors_[i]->params = params;

                        output_names_[i] = (char *) malloc(sizeof(char) * strlen(name));
                        strncpy(output_names_[i], name, strlen(name));
                        output_tensors_[i]->name = output_names_[i];
                    }
                    
                    break;
                }
                case 3:
                {
                    ::tflite::Interpreter * interpreter;
                    int tflite_input_size, tflite_output_size;
                    if(hexagonInterpreter_ != nullptr){
                        interpreter = hexagonInterpreter_;
                        tflite_input_size = interpreter->inputs().size();
                        tflite_output_size = interpreter->outputs().size();
                    }
                    else if(gpuInterpreter_ != nullptr){
                        interpreter = gpuInterpreter_;
                        tflite_input_size = interpreter->inputs().size();
                        tflite_output_size = interpreter->outputs().size();
                    }
                    else{
                        tflite_input_size = 0;
                        tflite_output_size = 0;
                    }

                    int mobilint_input_size = mobilintModel_->getModelInputShape().size();
                    int mobilint_output_size = mobilintModel_->getModelOutputShape().size();

                    int input_size = tflite_input_size + mobilint_input_size;
                    int output_size = tflite_output_size + mobilint_output_size;

                    inputs_ = std::vector<int>(input_size);
                    outputs_ = std::vector<int>(output_size);

                    input_dims_.resize(input_size);
                    input_tensors_.resize(input_size);
                    input_datas_.resize(input_size);
                    input_names_.resize(input_size);
                    output_dims_.resize(output_size);
                    output_tensors_.resize(output_size);
                    output_datas_.resize(output_size);
                    output_names_.resize(output_size);

                    for(int i = 0; i < tflite_input_size; i++){
                        TfLiteTensor* input_tensor_i = interpreter->input_tensor(i);
                        TfLiteIntArray* input_dims = input_tensor_i->dims;

                        input_dims_[i] = (TfLiteIntArray *) malloc(sizeof(int) * 5);

                        int malloc_size = 1;

                        input_dims_[i]->size = input_dims->size;
                        for(int j = 0; j < input_dims->size; j++){
                            input_dims_[i]->data[j] = input_dims->data[j];
                            malloc_size *= input_dims->data[j];
                        }

                        input_tensors_[i] = (TfLiteTensor *) malloc(sizeof(TfLiteTensor));

                        if(input_tensor_i->type == kTfLiteUInt8){
                            input_tensors_[i]->type = kTfLiteUInt8;
                            input_datas_[i] = malloc(sizeof(uint8_t) * malloc_size);
                        }
                        else if(input_tensor_i->type == kTfLiteFloat32){
                            input_tensors_[i]->type = kTfLiteFloat32;
                            input_datas_[i] = malloc(sizeof(float) * malloc_size);
                        }

                        input_tensors_[i]->dims = input_dims_[i];

                        TfLiteQuantizationParams params;
                        params.scale = input_tensor_i->params.scale;
                        params.zero_point = input_tensor_i->params.zero_point;
                        input_tensors_[i]->params = params;

                        const char * name = interpreter->GetInputName(i);
                        input_names_[i] = (char *) malloc(sizeof(char) * strlen(name));
                        strncpy(input_names_[i], name, strlen(name));
                        input_tensors_[i]->name = input_names_[i];
                    }

                    for(int i = tflite_input_size; i < input_size; i++){
                        auto input_shape_info = mobilintModel_->getModelInputShape()[i - tflite_input_size];

                        input_dims_[i] = (TfLiteIntArray *) malloc(sizeof(int) * 5);

                        if(input_shape_info[0] == 0){
                            input_dims_[i]->size = 0;
                        }
                        else if(input_shape_info[1] == 0){
                            input_dims_[i]->size = 2;
                            input_dims_[i]->data[0] = 1;
                            input_dims_[i]->data[1] = input_shape_info[0];

                            input_datas_[i] = malloc(sizeof(float) * input_shape_info[0]);
                        }
                        else if(input_shape_info[2] == 0){
                            input_dims_[i]->size = 3;
                            input_dims_[i]->data[0] = 1;
                            input_dims_[i]->data[1] = input_shape_info[0];
                            input_dims_[i]->data[2] = input_shape_info[1];

                            input_datas_[i] = malloc(sizeof(float) * input_shape_info[0] * input_shape_info[1]);
                        }
                        else{
                            input_dims_[i]->size = 4;
                            input_dims_[i]->data[0] = 1;
                            input_dims_[i]->data[1] = input_shape_info[0];
                            input_dims_[i]->data[2] = input_shape_info[1];
                            input_dims_[i]->data[3] = input_shape_info[2];

                            input_datas_[i] = malloc(sizeof(float) * input_shape_info[0] * input_shape_info[1] * input_shape_info[2]);
                        }

                        input_tensors_[i] = (TfLiteTensor *) malloc(sizeof(TfLiteTensor));

                        input_tensors_[i]->dims = input_dims_[i];
                        input_tensors_[i]->type = kTfLiteFloat32;
                        input_names_[i] = (char *) malloc(sizeof(char) * 2);
                        strcpy(input_names_[i], " ");
                    }

                    for(int i = 0; i < tflite_output_size; i++){
                        TfLiteTensor* output_tensor_i = interpreter->output_tensor(i);
                        TfLiteIntArray* output_dims = output_tensor_i->dims;

                        output_dims_[i] = (TfLiteIntArray *) malloc(sizeof(int) * 5);

                        int malloc_size = 1;

                        output_dims_[i]->size = output_dims->size;
                        for(int j = 0; j < output_dims->size; j++){
                            output_dims_[i]->data[j] = output_dims->data[j];
                            malloc_size *= output_dims->data[j];
                        }

                        output_tensors_[i] = (TfLiteTensor *) malloc(sizeof(TfLiteTensor));

                        if(output_tensor_i->type == kTfLiteUInt8){
                            output_tensors_[i]->type = kTfLiteUInt8;
                            output_datas_[i] = malloc(sizeof(uint8_t) * malloc_size);
                        }
                        else if(output_tensor_i->type == kTfLiteFloat32){
                            output_tensors_[i]->type = kTfLiteFloat32;
                            output_datas_[i] = malloc(sizeof(float) * malloc_size);
                        }
                        
                        output_tensors_[i]->dims = output_dims_[i];

                        TfLiteQuantizationParams params;
                        params.scale = output_tensor_i->params.scale;
                        params.zero_point = output_tensor_i->params.zero_point;
                        output_tensors_[i]->params = params;

                        const char * name = interpreter->GetOutputName(i);
                        output_names_[i] = (char *) malloc(sizeof(char) * strlen(name));
                        strncpy(output_names_[i], name, strlen(name));
                        output_tensors_[i]->name = output_names_[i];
                    }

                    for(int i = tflite_output_size; i < output_size; i++){
                        auto output_shape_info = mobilintModel_->getModelOutputShape()[i - tflite_output_size];

                        output_dims_[i] = (TfLiteIntArray *) malloc(sizeof(int) * 5);

                        if(output_shape_info[0] == 0){
                            output_dims_[i]->size = 0;
                        }
                        else if(output_shape_info[1] == 0){
                            output_dims_[i]->size = 2;
                            output_dims_[i]->data[0] = 1;
                            output_dims_[i]->data[1] = output_shape_info[0];

                            output_datas_[i] = malloc(sizeof(float) * output_shape_info[0]);
                        }
                        else if(output_shape_info[2] == 0){
                            output_dims_[i]->size = 3;
                            output_dims_[i]->data[0] = 1;
                            output_dims_[i]->data[1] = output_shape_info[0];
                            output_dims_[i]->data[2] = output_shape_info[1];

                            output_datas_[i] = malloc(sizeof(float) * output_shape_info[0] * output_shape_info[1]);
                        }
                        else{
                            output_dims_[i]->size = 4;
                            output_dims_[i]->data[0] = 1;
                            output_dims_[i]->data[1] = output_shape_info[0];
                            output_dims_[i]->data[2] = output_shape_info[1];
                            output_dims_[i]->data[3] = output_shape_info[2];

                            output_datas_[i] = malloc(sizeof(float) * output_shape_info[0] * output_shape_info[1] * output_shape_info[2]);
                        }

                        output_tensors_[i] = (TfLiteTensor *) malloc(sizeof(TfLiteTensor));

                        output_tensors_[i]->dims = output_dims_[i];
                        output_tensors_[i]->type = kTfLiteFloat32;
                        output_names_[i] = (char *) malloc(sizeof(char) * 2);
                        strcpy(output_names_[i], " ");
                    }

                    break;
                }
                case 4:
                {
                    ::tflite::Interpreter * interpreter;
                    int tflite_input_size, tflite_output_size;
                    if(hexagonInterpreter_ != nullptr){
                        interpreter = hexagonInterpreter_;
                        tflite_input_size = interpreter->inputs().size();
                        tflite_output_size = interpreter->outputs().size();
                    }
                    else if(gpuInterpreter_ != nullptr){
                        interpreter = gpuInterpreter_;
                        tflite_input_size = interpreter->inputs().size();
                        tflite_output_size = interpreter->outputs().size();
                    }
                    else{
                        tflite_input_size = 0;
                        tflite_output_size = 0;
                    }

                    int hailo_input_size = hailoVstreams_->first.size();
                    int hailo_output_size = hailoVstreams_->second.size();

                    int input_size = tflite_input_size + hailo_input_size;
                    int output_size = tflite_output_size + hailo_output_size;

                    inputs_ = std::vector<int>(input_size);
                    outputs_ = std::vector<int>(output_size);

                    input_dims_.resize(input_size);
                    input_tensors_.resize(input_size);
                    input_datas_.resize(input_size);
                    input_names_.resize(input_size);
                    output_dims_.resize(output_size);
                    output_tensors_.resize(output_size);
                    output_datas_.resize(output_size);
                    output_names_.resize(output_size);

                    for(int i = 0; i < tflite_input_size; i++){
                        TfLiteTensor* input_tensor_i = interpreter->input_tensor(i);
                        TfLiteIntArray* input_dims = input_tensor_i->dims;

                        input_dims_[i] = (TfLiteIntArray *) malloc(sizeof(int) * 5);

                        int malloc_size = 1;

                        input_dims_[i]->size = input_dims->size;

                        for(int j = 0; j < input_dims->size; j++){
                            input_dims_[i]->data[j] = input_dims->data[j];
                            malloc_size *= input_dims->data[j];
                        }

                        input_tensors_[i] = (TfLiteTensor *) malloc(sizeof(TfLiteTensor));

                        if(input_tensor_i->type == kTfLiteUInt8){
                            input_tensors_[i]->type = kTfLiteUInt8;
                            input_datas_[i] = malloc(sizeof(uint8_t) * malloc_size);
                        }
                        else if(input_tensor_i->type == kTfLiteFloat32){
                            input_tensors_[i]->type = kTfLiteFloat32;
                            input_datas_[i] = malloc(sizeof(float) * malloc_size);
                        }
                      
                        input_tensors_[i]->dims = input_dims_[i];

                        TfLiteQuantizationParams params;
                        params.scale = input_tensor_i->params.scale;
                        params.zero_point = input_tensor_i->params.zero_point;
                        input_tensors_[i]->params = params;

                        const char * name = interpreter->GetInputName(i);
                        input_names_[i] = (char *) malloc(sizeof(char) * strlen(name));
                        strncpy(input_names_[i], name, strlen(name));
                        input_tensors_[i]->name = input_names_[i];
                    }

                    for(int i = tflite_input_size; i < input_size; i++){
                        int frame_size = hailoVstreams_->first[i - tflite_input_size].get_frame_size();
                        auto input_info = hailoVstreams_->first[i - tflite_input_size].get_info();
                        auto shape = input_info.shape;
                        auto format = input_info.format;
                        auto quant_info = input_info.quant_info;
                        auto name = input_info.name;

                        input_dims_[i] = (TfLiteIntArray *) malloc(sizeof(int) * 5);

                        if(shape.height == 0){
                            input_dims_[i]->size = 0;
                        }
                        else if(shape.width == 0){
                            input_dims_[i]->size = 2;
                            input_dims_[i]->data[0] = 1;
                            input_dims_[i]->data[1] = frame_size;
                        }
                        else if(shape.features == 0){
                            input_dims_[i]->size = 3;
                            input_dims_[i]->data[0] = 1;
                            input_dims_[i]->data[1] = shape.height;
                            input_dims_[i]->data[2] = frame_size / shape.height;
                        }
                        else{
                            input_dims_[i]->size = 4;
                            input_dims_[i]->data[0] = 1;
                            input_dims_[i]->data[1] = shape.height;
                            input_dims_[i]->data[2] = shape.width;
                            input_dims_[i]->data[3] = frame_size / shape.height / shape.width;
                        }

                        input_tensors_[i] = (TfLiteTensor *) malloc(sizeof(TfLiteTensor));

                        input_tensors_[i]->dims = input_dims_[i];

                        switch(format.type){
                            case HAILO_FORMAT_TYPE_AUTO:
                            {
                                input_tensors_[i]->type = kTfLiteVariant;
                                break;
                            }
                            case HAILO_FORMAT_TYPE_UINT8:
                            {
                                input_tensors_[i]->type = kTfLiteUInt8;

                                input_dims_[i]->data[input_dims_[i]->size - 1] /= sizeof(uint8_t);

                                int size = 1;
                                for(int j = 0; j < input_dims_[i]->size; j++)
                                    size *= input_dims_[i]->data[j];

                                input_datas_[i] = malloc(sizeof(uint8_t) * size);
                                break;
                            }
                            case HAILO_FORMAT_TYPE_UINT16:
                            {
                                input_tensors_[i]->type = kTfLiteUInt16;

                                input_dims_[i]->data[input_dims_[i]->size - 1] /= sizeof(uint16_t);

                                int size = 1;
                                for(int j = 0; j < input_dims_[i]->size; j++)
                                    size *= input_dims_[i]->data[j];

                                input_datas_[i] = malloc(sizeof(uint16_t) * size);
                                break;
                            }
                            case HAILO_FORMAT_TYPE_FLOAT32:
                            {
                                input_tensors_[i]->type = kTfLiteFloat32;

                                input_dims_[i]->data[input_dims_[i]->size - 1] /= sizeof(float32_t);

                                int size = 1;
                                for(int j = 0; j < input_dims_[i]->size; j++)
                                    size *= input_dims_[i]->data[j];

                                input_datas_[i] = malloc(sizeof(float32_t) * size);
                                break;
                            }
                        }
                        
                        TfLiteQuantizationParams params;
                        params.zero_point = quant_info.qp_zp;
                        params.scale = quant_info.qp_scale;
                        input_tensors_[i]->params = params;

                        input_names_[i] = (char *) malloc(sizeof(char) * strlen(name));
                        strncpy(input_names_[i], name, strlen(name));
                        input_tensors_[i]->name = input_names_[i];
                    }

                    for(int i = 0; i < tflite_output_size; i++){
                        TfLiteTensor* output_tensor_i = interpreter->output_tensor(i);
                        TfLiteIntArray* output_dims = output_tensor_i->dims;

                        output_dims_[i] = (TfLiteIntArray *) malloc(sizeof(int) * 5);

                        int malloc_size = 1;

                        output_dims_[i]->size = output_dims->size;

                        for(int j = 0; j < output_dims->size; j++){
                            output_dims_[i]->data[j] = output_dims->data[j];
                            malloc_size *= output_dims->data[j];
                        }

                        output_tensors_[i] = (TfLiteTensor *) malloc(sizeof(TfLiteTensor));

                        if(output_tensor_i->type == kTfLiteUInt8){
                            output_tensors_[i]->type = kTfLiteUInt8;
                            output_datas_[i] = malloc(sizeof(uint8_t) * malloc_size);
                        }
                        else if(output_tensor_i->type == kTfLiteFloat32){
                            output_tensors_[i]->type = kTfLiteFloat32;
                            output_datas_[i] = malloc(sizeof(float) * malloc_size);
                        }

                        output_tensors_[i]->dims = output_dims_[i];

                        TfLiteQuantizationParams params;
                        params.scale = output_tensor_i->params.scale;
                        params.zero_point = output_tensor_i->params.zero_point;
                        output_tensors_[i]->params = params;

                        const char * name = interpreter->GetOutputName(i);
                        output_names_[i] = (char *) malloc(sizeof(char) * strlen(name));
                        strncpy(output_names_[i], name, strlen(name));
                        output_tensors_[i]->name = output_names_[i];
                    }

                    for(int i = tflite_output_size; i < output_size; i++){
                        int frame_size = hailoVstreams_->second[i - tflite_output_size].get_frame_size();
                        auto output_info = hailoVstreams_->second[i - tflite_output_size].get_info();
                        auto shape = output_info.shape;
                        auto format = output_info.format;
                        auto quant_info = output_info.quant_info;
                        auto name = output_info.name;

                        output_dims_[i] = (TfLiteIntArray *) malloc(sizeof(int) * 5);

                        if(shape.height == 0){
                            output_dims_[i]->size = 0;
                        }
                        else if(shape.width == 0){
                            output_dims_[i]->size = 2;
                            output_dims_[i]->data[0] = 1;
                            output_dims_[i]->data[1] = frame_size;
                        }
                        else if(shape.features == 0){
                            output_dims_[i]->size = 3;
                            output_dims_[i]->data[0] = 1;
                            output_dims_[i]->data[1] = shape.height;
                            output_dims_[i]->data[2] = frame_size / shape.height;
                        }
                        else{
                            output_dims_[i]->size = 4;
                            output_dims_[i]->data[0] = 1;
                            output_dims_[i]->data[1] = shape.height;
                            output_dims_[i]->data[2] = shape.width;
                            output_dims_[i]->data[3] = frame_size / shape.height / shape.width;
                        }

                        output_tensors_[i] = (TfLiteTensor *) malloc(sizeof(TfLiteTensor));

                        output_tensors_[i]->dims = output_dims_[i];

                        switch(format.type){
                            case HAILO_FORMAT_TYPE_AUTO:
                            {
                                output_tensors_[i]->type = kTfLiteVariant;
                                break;
                            }
                            case HAILO_FORMAT_TYPE_UINT8:
                            {
                                output_tensors_[i]->type = kTfLiteUInt8;

                                output_dims_[i]->data[output_dims_[i]->size - 1] /= sizeof(uint8_t);

                                int size = 1;
                                for(int j = 0; j < output_dims_[i]->size; j++)
                                    size *= output_dims_[i]->data[j];

                                output_datas_[i] = malloc(sizeof(uint8_t) * size);
                                break;
                            }
                            case HAILO_FORMAT_TYPE_UINT16:
                            {
                                output_tensors_[i]->type = kTfLiteUInt16;

                                output_dims_[i]->data[output_dims_[i]->size - 1] /= sizeof(uint16_t);

                                int size = 1;
                                for(int j = 0; j < output_dims_[i]->size; j++)
                                    size *= output_dims_[i]->data[j];

                                output_datas_[i] = malloc(sizeof(uint16_t) * size);
                                break;
                            }
                            case HAILO_FORMAT_TYPE_FLOAT32:
                            {
                                output_tensors_[i]->type = kTfLiteFloat32;

                                output_dims_[i]->data[output_dims_[i]->size - 1] /= sizeof(float32_t);

                                int size = 1;
                                for(int j = 0; j < output_dims_[i]->size; j++)
                                    size *= output_dims_[i]->data[j];

                                output_datas_[i] = malloc(sizeof(float32_t) * size);
                                break;
                            }
                        }

                        TfLiteQuantizationParams params;
                        params.zero_point = quant_info.qp_zp;
                        params.scale = quant_info.qp_scale;
                        output_tensors_[i]->params = params;

                        output_names_[i] = (char *) malloc(sizeof(char) * strlen(name));
                        strncpy(output_names_[i], name, strlen(name));
                        output_tensors_[i]->name = output_names_[i];
                    }
                    
                    break;
                }
            }
        }

        Interpreter::~Interpreter(){
            //std::cout << "Interpreter Destructor\n";

            switch(mode_){
                case 0:
                {
                    break;
                }
                case 1:
                case 2:
                {
                    for(int i = 0; i < inputs_.size(); i++){
                        free(input_datas_[i]);
                        free(input_dims_[i]);
                        free(input_names_[i]);
                        free(input_tensors_[i]);
                    }

                    for(int i = 0; i < outputs_.size(); i++){
                        free(output_datas_[i]);
                        free(output_dims_[i]);
                        free(output_names_[i]);
                        free(output_tensors_[i]);
                    }  
                }
                case 3:
                case 4:
                {
                    turnaround_mutex_.lock();
                    turnaround_mutex_.unlock();
                    
                    gpu_thread_start_cv_.notify_one();
                    gpu_thread_.join();

                    for(int i = 0; i < inputs_.size(); i++){
                        free(input_datas_[i]);
                        free(input_dims_[i]);
                        free(input_names_[i]);
                        free(input_tensors_[i]);
                    }

                    for(int i = 0; i < outputs_.size(); i++){
                        free(output_datas_[i]);
                        free(output_dims_[i]);
                        free(output_names_[i]);
                        free(output_tensors_[i]);
                    }  

                    break;
                }
            }
        }

        std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> * Interpreter::get_hailo_vstreams(){
            return hailoVstreams_;
        }

        mobilint::Model * Interpreter::get_mobilint_model(){
            return mobilintModel_;
        }

        bool Interpreter::is_tflite_model(){
            return mode_ == 0;
        }

        bool Interpreter::is_hailo_output(int batch_id){
            if(mode_ == 2)
                return true;
            else if (mode_ == 4){
                batch_mutex_[batch_id].lock();
                if(batch_run_[batch_id] == 3 || batch_run_[batch_id] == 4 || batch_run_[batch_id] == 5){
                    batch_mutex_[batch_id].unlock();
                    return true;
                }
                else{
                    batch_mutex_[batch_id].unlock();
                    return false;
                }
            }
            else
                return false;
        }

        bool Interpreter::is_maccel_output(int batch_id){
            if(mode_ == 1)
                return true;
            else if(mode_ == 3){
                batch_mutex_[batch_id].lock();
                if(batch_run_[batch_id] == 2){
                    batch_mutex_[batch_id].unlock();
                    return true;
                }
                else{
                    batch_mutex_[batch_id].unlock();
                    return false;
                }
            }
            else
                return false;
        }

        bool Interpreter::is_tflite_output(int batch_id){
            if(mode_ == 0)
                return true;
            else if(gpuInterpreter_ == nullptr && hexagonInterpreter_ == nullptr)
                return false;
            else if(mode_ == 3 || mode_ == 4){
                batch_mutex_[batch_id].lock();
                if(batch_run_[batch_id] == 0 || batch_run_[batch_id] == 1){
                    batch_mutex_[batch_id].unlock();
                    return true;
                }
                else{
                    batch_mutex_[batch_id].unlock();
                    return false;
                }
            }
            else
                return false;
        }

        TfLiteStatus Interpreter::ModifyGraphWithDelegate(TfLiteDelegate* delegate){
            switch(mode_){
                case 0:
                {
                    return ::tflite::Interpreter::ModifyGraphWithDelegate(delegate);

                    break;
                }
                case 1:
                case 2:
                case 3:
                case 4:
                {
                    return kTfLiteOk;

                    break;
                }
            }
        }

        TfLiteStatus Interpreter::AllocateTensors(){
            switch(mode_){
                case 0:
                {
                    return ::tflite::Interpreter::AllocateTensors();

                    break;
                }
                case 1:
                case 2:
                case 3:
                case 4:
                {
                    return kTfLiteOk;

                    break;
                }
            }
        }

        TfLiteStatus Interpreter::ResizeInputTensor(int tensor_index, const std::vector<int>& dims){
            //std::cout << "ResizeInputTensor start\n";

            switch(mode_){
                case 0:
                {
                    return ::tflite::Interpreter::ResizeInputTensor(tensor_index, dims);

                    break;
                }
                case 1:
                case 2:
                {
                    if(dims[0] == 1)
                        return kTfLiteOk;
                    else
                        return kTfLiteError;

                    break;
                }
                case 3:
                case 4:
                {
                    if(batch_sizes_ == dims[0])
                        return kTfLiteOk;

                    batch_sizes_ = dims[0];
                    batch_run_.resize(batch_sizes_);
                    batch_mutex_ = std::vector<std::mutex>(batch_sizes_);
                    turnaround_.resize(batch_sizes_);

                    for(int i = 0; i < inputs_.size(); i++){
                        free(input_datas_[i]);

                        input_dims_[i]->data[0] = batch_sizes_;

                        int size = 1;
                        for(int j = 0; j < input_dims_[i]->size; j++)
                            size *= input_dims_[i]->data[j];

                        if(input_tensors_[i]->type == kTfLiteUInt8){
                            input_datas_[i] = malloc(sizeof(uint8_t) * size);
                        }
                        else if(input_tensors_[i]->type == kTfLiteFloat32){
                            input_datas_[i] = malloc(sizeof(float) * size);
                        }
                    }

                    for(int i = 0; i < outputs_.size(); i++){
                        free(output_datas_[i]);

                        output_dims_[i]->data[0] = batch_sizes_;

                        int size = 1;
                        for(int j = 0; j < output_dims_[i]->size; j++)
                            size *= output_dims_[i]->data[j];

                        if(output_tensors_[i]->type == kTfLiteUInt8){
                            output_datas_[i] = malloc(sizeof(uint8_t) * size);
                        }
                        else if(output_tensors_[i]->type == kTfLiteFloat32){
                            output_datas_[i] = malloc(sizeof(float) * size);                
                        }
                    }

                    return kTfLiteOk;

                    break;
                }
            }
        }

        const std::vector<int>& Interpreter::inputs(){
            switch(mode_){
                case 0:
                {
                    return ::tflite::Interpreter::inputs();

                    break;
                }
                case 1:
                case 2:
                case 3:
                case 4:
                {
                    return inputs_;

                    break;
                }
            }
        }

        const std::vector<int>& Interpreter::outputs(){
            switch(mode_){
                case 0:
                {
                    return ::tflite::Interpreter::outputs();

                    break;
                }
                case 1:
                case 2:
                case 3:
                case 4:
                {
                    return outputs_;

                    break;
                }
            }
        }

        const char* Interpreter::GetInputName(int index){
            switch(mode_){
                case 0:
                {
                    return ::tflite::Interpreter::GetInputName(index);

                    break;
                }
                case 1:
                case 2:
                case 3:
                case 4:
                {
                    return input_names_[index];

                    break;
                }
            }
        }

        const char* Interpreter::GetOutputName(int index){
            switch(mode_){
                case 0:
                {
                    return ::tflite::Interpreter::GetOutputName(index);

                    break;
                }
                case 1:
                case 2:
                case 3:
                case 4:
                {
                    return output_names_[index];

                    break;
                }
            }
        }

        TfLiteTensor* Interpreter::input_tensor(size_t index){
            switch(mode_){
                case 0:
                {
                    return ::tflite::Interpreter::input_tensor(index);

                    break;
                }
                case 1:
                case 2:
                case 3:
                case 4:
                {
                    return input_tensors_[index];

                    break;
                }
            }
        }

        TfLiteTensor* Interpreter::output_tensor(size_t index){
            switch(mode_){
                case 0:
                {
                    return ::tflite::Interpreter::output_tensor(index);

                    break;
                }
                case 1:
                case 2:
                case 3:
                case 4:
                {
                    return output_tensors_[index];

                    break;
                }
            }
        }

        void * Interpreter::get_input_data(int index){
            if(mode_ == 3 || mode_ == 4){
                turnaround_mutex_.lock();
                void * return_value = input_datas_[index];
                turnaround_mutex_.unlock();
                return return_value;
            }
            else{
                return input_datas_[index];
            }
        }

        void * Interpreter::get_output_data(int index){
            return output_datas_[index];
        }

        TfLiteStatus Interpreter::SetSchedulerParams(std::vector<float> perfs){
            perfs_ = perfs;

            return kTfLiteOk;
        }

        TfLiteStatus Interpreter::SetPostProcessParams(std::vector<float> ori_score_thrs, std::vector<float> new_score_thrs){
            ori_score_thrs_ = ori_score_thrs;
            new_score_thrs_ = new_score_thrs;
            cur_score_thrs_ = ori_score_thrs;

            return kTfLiteOk;
        }

        std::vector<float> Interpreter::GetPostProcessParams(){
            return cur_score_thrs_;
        }

        double Interpreter::GetSumTurnAroundTime(){
            turnaround_mutex_.lock();
            double return_value = sum_turnaround_;
            turnaround_mutex_.unlock();
            return return_value;
        }

        double Interpreter::GetMaxTurnAroundTime(){
            turnaround_mutex_.lock();
            double return_value = max_turnaround_;
            turnaround_mutex_.unlock();
            return return_value;
        }

        void Invoke_gpu_queue(){
            std::unique_ptr<::tflite::FlatBufferModel> model = ::tflite::FlatBufferModel::BuildFromFile(tflite_filename_);
            if(model == NULL){
                std::cerr << "ERROR: Model load failed. Check the model name.\n";
                exit(-1);
            }
            
            ::tflite::ops::builtin::BuiltinOpResolver resolver;
            ::tflite::InterpreterBuilder builder(*model, resolver);
            std::unique_ptr<::tflite::Interpreter> interpreter;
            builder(&interpreter);
            if(interpreter == NULL){
                std::cerr << "ERROR: Interpreter build failed.\n";
                exit(-1);
            }
            
            TfLiteGpuDelegateOptionsV2 gpu_delegate_options = TfLiteGpuDelegateOptionsV2Default();
            gpu_delegate_options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
            gpu_delegate_options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
            gpu_delegate_options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
            gpu_delegate_options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
            gpu_delegate_options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY;

            auto * gpu_delegate_ptr = TfLiteGpuDelegateV2Create(&gpu_delegate_options);
            if(gpu_delegate_ptr == NULL){
                std::cout << "WARNING: Cannot create gpu delegate. Run without gpu delegate.\n";
                TfLiteGpuDelegateV2Delete(gpu_delegate_ptr);
                gpuInterpreter_ = nullptr;
            }
            else{
                ::tflite::Interpreter::TfLiteDelegatePtr gpu_delegate(gpu_delegate_ptr, &TfLiteGpuDelegateV2Delete);

                if (interpreter->ModifyGraphWithDelegate(gpu_delegate.get()) != kTfLiteOk){
                    std::cout << "WARNING: Cannot convert model with gpu delegate. Run without gpu delegate.\n";
                    gpuInterpreter_ = nullptr;
                }
                else{
                    if(interpreter->AllocateTensors() != kTfLiteOk) {
                        std::cerr << "ERROR: Memory allocation for interpreter failed.\n";
                        exit(-1);
                    }
                    
                    gpuInterpreter_ = interpreter.get();
                }
            }

            gpu_thread_set_cv_.notify_one();

            std::unique_lock<std::mutex> lk(gpu_thread_mutex_);
            while(true){
                gpu_thread_start_cv_.wait(lk);
                gpu_thread_mutex_.unlock();

                if(gpu_queue_.size() == 0){
                    std::cout << "INFO: Terminate gpu thread\n";
                    break;
                }

                for(int i = 0; i < gpu_queue_.size(); i++){
                    //std::cout << "Invoke gpu queue\n";
                    for(int j = 0; j < gpuInterpreter_->inputs().size(); j++){
                        int size = 1;
                        for(int k = 1; k < input_dims_[j]->size; k++)
                            size *= input_dims_[j]->data[k];

                        if(input_tensors_[j]->type == kTfLiteUInt8){
                            uint8_t * input_ptr = (uint8_t *)input_datas_[j];
                            input_ptr += gpu_queue_[i] * size;
                            memcpy(gpuInterpreter_->typed_input_tensor<uint8_t>(j), input_ptr, sizeof(uint8_t) * size);
                        }
                        else if(input_tensors_[j]->type == kTfLiteFloat32){
                            float * input_ptr = (float *)input_datas_[j];
                            input_ptr += gpu_queue_[i] * size;
                            memcpy(gpuInterpreter_->typed_input_tensor<float>(j), input_ptr, sizeof(float) * size);
                        }
                    }
              
                    if(gpuInterpreter_->Invoke() != kTfLiteOk){
                        std::cerr << "ERROR: Model execute failed\n";
                        exit(-1);
                    }

                    for(int j = 0; j < gpuInterpreter_->outputs().size(); j++){
                        int size = 1;
                        for(int k = 1; k < output_dims_[j]->size; k++)
                            size *= output_dims_[j]->data[k];

                        if(output_tensors_[j]->type == kTfLiteUInt8){
                            uint8_t * output_ptr = (uint8_t *)output_datas_[j];
                            output_ptr += gpu_queue_[i] * size;
                            memcpy(output_ptr, gpuInterpreter_->typed_output_tensor<uint8_t>(j), sizeof(uint8_t) * size);
                        }
                        else if(output_tensors_[j]->type == kTfLiteFloat32){
                            float * output_ptr = (float *)output_datas_[j];
                            output_ptr += gpu_queue_[i] * size;
                            memcpy(output_ptr, gpuInterpreter_->typed_output_tensor<float>(j), sizeof(float) * size);
                        }
                    }

                    auto elapsed = std::chrono::high_resolution_clock::now() - invoke_start_;
                    turnaround_[gpu_queue_[i]] = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

                    batch_run_[gpu_queue_[i]] = 0;
                    batch_mutex_[gpu_queue_[i]].unlock();
                }

                gpu_thread_end_cv_.notify_one();
            }
        }

        void Invoke_hexagon_queue(){
            for(int i = 0; i < hexagon_queue_.size(); i++){
                //std::cout << "Invoke hexagon queue\n";
                for(int j = 0; j < hexagonInterpreter_->inputs().size(); j++){
                    int size = 1;
                    for(int k = 1; k < input_dims_[j]->size; k++)
                        size *= input_dims_[j]->data[k];

                    if(input_tensors_[j]->type == kTfLiteUInt8){
                        uint8_t * input_ptr = (uint8_t *)input_datas_[j];
                        input_ptr += hexagon_queue_[i] * size;
                        memcpy(hexagonInterpreter_->typed_input_tensor<uint8_t>(j), input_ptr, sizeof(uint8_t) * size);
                    }
                    else if(input_tensors_[j]->type == kTfLiteFloat32){
                        float * input_ptr = (float *)input_datas_[j];
                        input_ptr += hexagon_queue_[i] * size;
                        memcpy(hexagonInterpreter_->typed_input_tensor<float>(j), input_ptr, sizeof(float) * size);
                    }
                }
                
                if(hexagonInterpreter_->Invoke() != kTfLiteOk){
                    std::cerr << "ERROR: Model execute failed\n";
                    exit(-1);
                }

                for(int j = 0; j < hexagonInterpreter_->outputs().size(); j++){
                    int size = 1;
                    for(int k = 1; k < output_dims_[j]->size; k++)
                        size *= output_dims_[j]->data[k];

                    if(output_tensors_[j]->type == kTfLiteUInt8){
                        uint8_t * output_ptr = (uint8_t *)output_datas_[j];
                        output_ptr += hexagon_queue_[i] * size;
                        memcpy(output_ptr, hexagonInterpreter_->typed_output_tensor<uint8_t>(j), sizeof(uint8_t) * size);
                    }
                    else if(output_tensors_[j]->type == kTfLiteFloat32){
                        float * output_ptr = (float *)output_datas_[j];
                        output_ptr += hexagon_queue_[i] * size;
                        memcpy(output_ptr, hexagonInterpreter_->typed_output_tensor<float>(j), sizeof(float) * size);
                    }
                }

                auto elapsed = std::chrono::high_resolution_clock::now() - invoke_start_;
                turnaround_[hexagon_queue_[i]] = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

                batch_run_[hexagon_queue_[i]] = 1;
                batch_mutex_[hexagon_queue_[i]].unlock();
            }
        }

        void Invoke_maccel_queue(){
            for(int i = 0; i < maccel_queue_.size(); i++){
                //std::cout << "Invoke maccel queue\n";
                mobilint::StatusCode sc;

                ::tflite::Interpreter * interpreter;
                int tflite_input_size, tflite_output_size;
                if(hexagonInterpreter_ != nullptr){
                    interpreter = hexagonInterpreter_;
                    tflite_input_size = interpreter->inputs().size();
                    tflite_output_size = interpreter->outputs().size();
                }
                else if(gpuInterpreter_ != nullptr){
                    interpreter = gpuInterpreter_;
                    tflite_input_size = interpreter->inputs().size();
                    tflite_output_size = interpreter->outputs().size();
                }
                else{
                    tflite_input_size = 0;
                    tflite_output_size = 0;
                }

                std::vector<float *> float_input_datas;
                float_input_datas.resize(inputs_.size() - tflite_input_size);

                for(int j = tflite_input_size; j < inputs_.size(); j++){
                    int size = 1;
                    for(int k = 1; k < input_dims_[j]->size; k++)
                        size *= input_dims_[j]->data[k];

                    float * input_ptr = (float *)input_datas_[j];
                    input_ptr += maccel_queue_[i] * size;

                    float_input_datas[j - tflite_input_size] = input_ptr;
                }
                
                std::vector<std::vector<float>> outputs = mobilintModel_->infer(float_input_datas, sc);
                if (!sc) {
                    std::cerr << "ERROR: Failed to infer an output. error code: " << static_cast<int>(sc) << std::endl;
                    exit(-1);
                }

                for(int j = tflite_output_size; j < outputs_.size(); j++){
                    int size = 1;
                    for(int k = 1; k < output_dims_[j]->size; k++)
                        size *= output_dims_[j]->data[k];

                    float * output_ptr = (float *)output_datas_[j];
                    output_ptr += maccel_queue_[i] * size;
                    
                    memcpy(output_ptr, outputs[j - tflite_output_size].data(), sizeof(float) * size);
                }

                auto elapsed = std::chrono::high_resolution_clock::now() - invoke_start_;
                turnaround_[maccel_queue_[i]] = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

                batch_run_[maccel_queue_[i]] = 2;
                batch_mutex_[maccel_queue_[i]].unlock();
            }
        }

        void Invoke_hailo_queue(){
            ::tflite::Interpreter * interpreter;
            int tflite_input_size, tflite_output_size;
            if(hexagonInterpreter_ != nullptr){
                interpreter = hexagonInterpreter_;
                tflite_input_size = interpreter->inputs().size();
                tflite_output_size = interpreter->outputs().size();
            }
            else if(gpuInterpreter_ != nullptr){
                interpreter = gpuInterpreter_;
                tflite_input_size = interpreter->inputs().size();
                tflite_output_size = interpreter->outputs().size();
            }
            else{
                tflite_input_size = 0;
                tflite_output_size = 0;
            }

            for(int i = 0; i < hailo_queue_.size(); i++){
                //std::cout << "Invoke hailo queue1\n";
                for(int j = tflite_input_size; j < inputs_.size(); j++){
                    auto & inputVStream = hailoVstreams_->first[j - tflite_input_size];

                    int size = 1;
                    for(int k = 1; k < input_dims_[j]->size; k++)
                        size *= input_dims_[j]->data[k];

                    if(input_tensors_[j]->type == kTfLiteUInt8){
                        uint8_t * input_ptr = (uint8_t *)input_datas_[j] + hailo_queue_[i] * size;
                        auto status = inputVStream.write(hailort::MemoryView(input_ptr, size * sizeof(uint8_t)));
                        if (HAILO_SUCCESS != status) {
                            std::cerr << "ERROR: write to input vstream " << j << " failed\n";
                            exit(-1);
                        }
                    }
                    else if(input_tensors_[j]->type == kTfLiteFloat32){
                        float * input_ptr = (float *)input_datas_[j] + hailo_queue_[i] * size;
                        input_ptr += hailo_queue_[i] * size;
                        auto status = inputVStream.write(hailort::MemoryView(input_ptr, size * sizeof(float)));
                        if (HAILO_SUCCESS != status) {
                            std::cerr << "ERROR: write to input vstream " << j << " failed\n";
                            exit(-1);
                        }
                    }
                }

                for(int j = tflite_output_size; j < outputs_.size(); j++){
                    auto & outputVStream = hailoVstreams_->second[j - tflite_output_size];

                    int size = 1;
                    for(int k = 1; k < output_dims_[j]->size; k++)
                        size *= output_dims_[j]->data[k];

                    if(output_tensors_[j]->type == kTfLiteUInt8){
                        uint8_t * output_ptr = (uint8_t *)output_datas_[j] + hailo_queue_[i] * size;
                        auto status = outputVStream.read(hailort::MemoryView(output_ptr, size * sizeof(uint8_t)));
                        if (HAILO_SUCCESS != status) {
                            std::cerr << "ERROR: reading output vstream " << j << " failed\n";
                            exit(-1);
                        }
                    }
                    else if(output_tensors_[j]->type == kTfLiteFloat32){
                        float * output_ptr = (float *)output_datas_[j] + hailo_queue_[i] * size;
                        auto status = outputVStream.read(hailort::MemoryView(output_ptr, size * sizeof(float)));
                        if (HAILO_SUCCESS != status) {
                            std::cerr << "ERROR: reading output vstream " << j << " failed\n";
                            exit(-1);
                        }
                    }
                }
                auto elapsed = std::chrono::high_resolution_clock::now() - invoke_start_;
                turnaround_[hailo_queue_[i]] = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

                batch_run_[hailo_queue_[i]] = 3;
                batch_mutex_[hailo_queue_[i]].unlock();
            }
        }

        void Invoke_hailo_queue2(){
            ::tflite::Interpreter * interpreter;
            int tflite_input_size, tflite_output_size;
            if(hexagonInterpreter_ != nullptr){
                interpreter = hexagonInterpreter_;
                tflite_input_size = interpreter->inputs().size();
                tflite_output_size = interpreter->outputs().size();
            }
            else if(gpuInterpreter_ != nullptr){
                interpreter = gpuInterpreter_;
                tflite_input_size = interpreter->inputs().size();
                tflite_output_size = interpreter->outputs().size();
            }
            else{
                tflite_input_size = 0;
                tflite_output_size = 0;
            }

            for(int i = 0; i < hailo_queue2_.size(); i++){
                //std::cout << "Invoke hailo queue2\n";
                for(int j = tflite_input_size; j < inputs_.size(); j++){
                    auto & inputVStream = hailoVstreams2_->first[j - tflite_input_size];

                    int size = 1;
                    for(int k = 1; k < input_dims_[j]->size; k++)
                        size *= input_dims_[j]->data[k];

                    if(input_tensors_[j]->type == kTfLiteUInt8){
                        uint8_t * input_ptr = (uint8_t *)input_datas_[j] + hailo_queue2_[i] * size;
                        auto status = inputVStream.write(hailort::MemoryView(input_ptr, size * sizeof(uint8_t)));
                        if (HAILO_SUCCESS != status) {
                            std::cerr << "ERROR: write to input vstream " << j << " failed\n";
                            exit(-1);
                        }
                    }
                    else if(input_tensors_[j]->type == kTfLiteFloat32){
                        float * input_ptr = (float *)input_datas_[j] + hailo_queue2_[i] * size;
                        auto status = inputVStream.write(hailort::MemoryView(input_ptr, size * sizeof(float)));
                        if (HAILO_SUCCESS != status) {
                            std::cerr << "ERROR: write to input vstream " << j << " failed\n";
                            exit(-1);
                        }
                    }
                }

                for(int j = tflite_output_size; j < outputs_.size(); j++){
                    auto & outputVStream = hailoVstreams2_->second[j - tflite_output_size];

                    int size = 1;
                    for(int k = 1; k < output_dims_[j]->size; k++)
                        size *= output_dims_[j]->data[k];

                    if(output_tensors_[j]->type == kTfLiteUInt8){
                        uint8_t * output_ptr = (uint8_t *)output_datas_[j] + hailo_queue2_[i] * size;
                        auto status = outputVStream.read(hailort::MemoryView(output_ptr, size * sizeof(uint8_t)));
                        if (HAILO_SUCCESS != status) {
                            std::cerr << "ERROR: reading output vstream " << j << " failed\n";
                            exit(-1);
                        }
                    }
                    else if(output_tensors_[j]->type == kTfLiteFloat32){
                        float * output_ptr = (float *)output_datas_[j] + hailo_queue2_[i] * size;
                        auto status = outputVStream.read(hailort::MemoryView(output_ptr, size * sizeof(float)));
                        if (HAILO_SUCCESS != status) {
                            std::cerr << "ERROR: reading output vstream " << j << " failed\n";
                            exit(-1);
                        }
                    }
                }
                auto elapsed = std::chrono::high_resolution_clock::now() - invoke_start_;
                turnaround_[hailo_queue2_[i]] = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

                batch_run_[hailo_queue2_[i]] = 4;
                batch_mutex_[hailo_queue2_[i]].unlock();
            }
        }

        void Invoke_hailo_queue3(){
            ::tflite::Interpreter * interpreter;
            int tflite_input_size, tflite_output_size;
            if(hexagonInterpreter_ != nullptr){
                interpreter = hexagonInterpreter_;
                tflite_input_size = interpreter->inputs().size();
                tflite_output_size = interpreter->outputs().size();
            }
            else if(gpuInterpreter_ != nullptr){
                interpreter = gpuInterpreter_;
                tflite_input_size = interpreter->inputs().size();
                tflite_output_size = interpreter->outputs().size();
            }
            else{
                tflite_input_size = 0;
                tflite_output_size = 0;
            }

            for(int i = 0; i < hailo_queue3_.size(); i++){
                //std::cout << "Invoke hailo queue3\n";
                for(int j = tflite_input_size; j < inputs_.size(); j++){
                    auto & inputVStream = hailoVstreams3_->first[j - tflite_input_size];

                    int size = 1;
                    for(int k = 1; k < input_dims_[j]->size; k++)
                        size *= input_dims_[j]->data[k];
                    
                    if(input_tensors_[j]->type == kTfLiteUInt8){
                        uint8_t * input_ptr = (uint8_t *)input_datas_[j] + hailo_queue3_[i] * size;
                        auto status = inputVStream.write(hailort::MemoryView(input_ptr, size * sizeof(uint8_t)));
                        if (HAILO_SUCCESS != status) {
                            std::cerr << "ERROR: write to input vstream " << j << " failed\n";
                            exit(-1);
                        }
                    }
                    else if(input_tensors_[j]->type == kTfLiteFloat32){
                        float * input_ptr = (float *)input_datas_[j] + hailo_queue3_[i] * size;
                        auto status = inputVStream.write(hailort::MemoryView(input_ptr, size * sizeof(float)));
                        if (HAILO_SUCCESS != status) {
                            std::cerr << "ERROR: write to input vstream " << j << " failed\n";
                            exit(-1);
                        }
                    }
                }

                for(int j = tflite_output_size; j < outputs_.size(); j++){
                    auto & outputVStream = hailoVstreams3_->second[j - tflite_output_size];

                    int size = 1;
                    for(int k = 1; k < output_dims_[j]->size; k++)
                        size *= output_dims_[j]->data[k];

                    if(output_tensors_[j]->type == kTfLiteUInt8){
                        uint8_t * output_ptr = (uint8_t *)output_datas_[j] + hailo_queue3_[i] * size;
                        auto status = outputVStream.read(hailort::MemoryView(output_ptr, size * sizeof(uint8_t)));
                        if (HAILO_SUCCESS != status) {
                            std::cerr << "ERROR: reading output vstream " << j << " failed\n";
                            exit(-1);
                        }
                    }
                    else if(output_tensors_[j]->type == kTfLiteFloat32){
                        float * output_ptr = (float *)output_datas_[j] + hailo_queue3_[i] * size;
                        auto status = outputVStream.read(hailort::MemoryView(output_ptr, size * sizeof(float)));
                        if (HAILO_SUCCESS != status) {
                            std::cerr << "ERROR: reading output vstream " << j << " failed\n";
                            exit(-1);
                        }
                    }
                }
                auto elapsed = std::chrono::high_resolution_clock::now() - invoke_start_;
                turnaround_[hailo_queue3_[i]] = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

                batch_run_[hailo_queue3_[i]] = 5;
                batch_mutex_[hailo_queue3_[i]].unlock();
            }
        }

        void Invoke_thread(){
            std::thread hexagon_thread;
            std::thread maccel_thread;
            std::thread hailo_thread;
            std::thread hailo_thread2;
            std::thread hailo_thread3;

            if(gpu_queue_.size() > 0){
                gpu_thread_start_cv_.notify_one();
            }

            if(hexagon_queue_.size() > 0)
                hexagon_thread = std::thread(Invoke_hexagon_queue);

            if(maccel_queue_.size() > 0)
                maccel_thread = std::thread(Invoke_maccel_queue);
            
            if(hailo_queue_.size() > 0)
                hailo_thread = std::thread(Invoke_hailo_queue);

            if(hailo_queue2_.size() > 0)
                hailo_thread2 = std::thread(Invoke_hailo_queue2);
            
            if(hailo_queue3_.size() > 0)
                hailo_thread3 = std::thread(Invoke_hailo_queue3);

            if(gpu_queue_.size() > 0){
                std::unique_lock<std::mutex> lk(gpu_thread_mutex_);
                gpu_thread_end_cv_.wait(lk);
                gpu_thread_mutex_.unlock();
            }

            if(hexagon_queue_.size() > 0)
                hexagon_thread.join();
            
            if(maccel_queue_.size() > 0)
                maccel_thread.join();
            
            if(hailo_queue_.size() > 0)
                hailo_thread.join();

            if(hailo_queue2_.size() > 0)
                hailo_thread2.join();

            if(hailo_queue3_.size() > 0)
                hailo_thread3.join();

            gpu_queue_.clear();
            hexagon_queue_.clear();
            maccel_queue_.clear();
            hailo_queue_.clear();
            hailo_queue2_.clear();
            hailo_queue3_.clear();

            auto elapsed = std::chrono::high_resolution_clock::now() - invoke_start_;
            max_turnaround_ = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

            sum_turnaround_ = 0;
            for(int i = 0; i < batch_sizes_; i++)
                sum_turnaround_ += turnaround_[i];

            turnaround_mutex_.unlock();
        }

        TfLiteStatus Interpreter::Invoke(){
            invoke_start_ = std::chrono::high_resolution_clock::now();
            switch(mode_){
                case 0:
                {
                    //std::cout << "Invoke tflite\n";
                    auto status = ::tflite::Interpreter::Invoke();

                    auto elapsed = std::chrono::high_resolution_clock::now() - invoke_start_;
                    sum_turnaround_ = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
                    max_turnaround_ = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

                    return status;
                    
                    break;
                }
                case 1:
                {
                    //std::cout << "Invoke maccel\n";
                    mobilint::StatusCode sc;

                    std::vector<float *> float_input_datas;
                    float_input_datas.resize(input_datas_.size());

                    for(int i = 0; i < input_datas_.size(); i++)
                        float_input_datas[i] = (float *)input_datas_[i];
                    
                    std::vector<std::vector<float>> outputs = mobilintModel_->infer(float_input_datas, sc);
                    if (!sc) {
                        std::cerr << "ERROR: Failed to infer an output. error code: " << static_cast<int>(sc) << std::endl;
                        return kTfLiteError;
                    }

                    for(int i = 0; i < outputs.size(); i++){
                        memcpy(output_datas_[i], outputs[i].data(), outputs[i].size() * sizeof(float));
                    }

                    auto elapsed = std::chrono::high_resolution_clock::now() - invoke_start_;
                    sum_turnaround_ = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
                    max_turnaround_ = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

                    return kTfLiteOk;

                    break;
                }
                case 2:
                {
                    //std::cout << "Invoke hailo\n";
                    for(int i = 0; i < hailoVstreams_->first.size(); i++){
                        auto & inputVStream = hailoVstreams_->first[i];

                        int size = 1;
                        for(int j = 1; j < input_dims_[i]->size; j++)
                            size *= input_dims_[i]->data[j];

                        if(input_tensors_[i]->type == kTfLiteUInt8){
                            auto status = inputVStream.write(hailort::MemoryView(input_datas_[i], size * sizeof(uint8_t)));
                            if (HAILO_SUCCESS != status) {
                                std::cerr << "ERROR: write to input vstream " << i << " failed\n";
                                return kTfLiteError;
                            }
                        }
                        else if(input_tensors_[i]->type == kTfLiteFloat32){
                            auto status = inputVStream.write(hailort::MemoryView(input_datas_[i], size * sizeof(float)));
                            if (HAILO_SUCCESS != status) {
                                std::cerr << "ERROR: write to input vstream " << i << " failed\n";
                                return kTfLiteError;
                            }
                        }
                    }

                    for(int i = 0; i < hailoVstreams_->second.size(); i++){
                        auto & outputVStream = hailoVstreams_->second[i];

                        int size = 1;
                        for(int j = 1; j < output_dims_[i]->size; j++)
                            size *= output_dims_[j]->data[j];

                        if(output_tensors_[i]->type == kTfLiteUInt8){
                            auto status = outputVStream.read(hailort::MemoryView(output_datas_[i], size * sizeof(uint8_t)));
                            if (HAILO_SUCCESS != status) {
                                std::cerr << "ERROR: reading output vstream " << i << " failed\n";
                                return kTfLiteError;
                            }
                        }
                        else if(output_tensors_[i]->type == kTfLiteFloat32){
                            auto status = outputVStream.read(hailort::MemoryView(output_datas_[i], size * sizeof(float)));
                            if (HAILO_SUCCESS != status) {
                                std::cerr << "ERROR: reading output vstream " << i << " failed\n";
                                return kTfLiteError;
                            }
                        }
                    }

                    auto elapsed = std::chrono::high_resolution_clock::now() - invoke_start_;
                    sum_turnaround_ = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
                    max_turnaround_ = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

                    return kTfLiteOk;

                    break;
                }
                case 3:
                {
                    float c[3];
                    c[0] = perfs_[0];
                    c[1] = perfs_[1];
                    c[2] = perfs_[2];
                    if(gpuInterpreter_ == nullptr){
                        c[0] = 2147483647;
                    }
                    if(hexagonInterpreter_ == nullptr){
                        c[1] = 2147483647;
                    }

                    int k[3] = {0, 0, 0};

                    for(int i = 0; i < batch_sizes_; i++){
                        int min_l = 2147483647;
                        int index;

                        for(int j = 0; j < 3; j++){
                            if((k[j] + 1) * c[j] < min_l){
                                min_l = (k[j] + 1) * c[j];
                                index = j;
                            }
                        }

                        k[index]++;

                        switch(index){
                            case 0:
                            {
                                gpu_queue_.push_back(i);       
                                break;
                            }
                            case 1:
                            {
                                hexagon_queue_.push_back(i);
                                break;
                            }
                            case 2:
                            {
                                maccel_queue_.push_back(i);
                                break;
                            }
                        }
                    }

                    if(!gpu_queue_.empty() || !hexagon_queue_.empty()){
                        if(new_score_thrs_[0] >= 0)
                            cur_score_thrs_[0] = new_score_thrs_[0];
                    }
                    else{
                        if(ori_score_thrs_[0] >= 0)
                            cur_score_thrs_[0] = ori_score_thrs_[0];
                    }

                    turnaround_mutex_.lock();

                    for(int i = 0; i < batch_sizes_; i++){
                        batch_mutex_[i].lock();
                    }

                    std::thread invoke_thread(Invoke_thread);
                    invoke_thread.detach();

                    return kTfLiteOk;

                    break;
                }
                case 4:
                {
                    float c[5];
                    c[0] = perfs_[0];
                    c[1] = perfs_[1];
                    c[2] = perfs_[2];
                    c[3] = perfs_[3];
                    c[4] = perfs_[4];
                    if(gpuInterpreter_ == nullptr){
                        c[0] = 2147483647;
                    }
                    if(hexagonInterpreter_ == nullptr){
                        c[1] = 2147483647;
                    }

                    int k[5] = {0, 0, 0, 0, 0};

                    for(int i = 0; i < batch_sizes_; i++){
                        int min_l = 2147483647;
                        int index;

                        for(int j = 0; j < 5; j++){
                            if((k[j] + 1) * c[j] < min_l){
                                min_l = (k[j] + 1) * c[j];
                                index = j;
                            }
                        }

                        k[index]++;

                        switch(index){
                            case 0:
                            {
                                gpu_queue_.push_back(i);
                                break;
                            }
                            case 1:
                            {
                                hexagon_queue_.push_back(i);
                                break;
                            }
                            case 2:
                            {
                                hailo_queue_.push_back(i);
                                break;
                            }
                            case 3:
                            {
                                hailo_queue2_.push_back(i);
                                break;
                            }
                            case 4:
                            {
                                hailo_queue3_.push_back(i);
                                break;
                            }
                        }
                    }

                    ::tflite::Interpreter * interpreter;
                    int tflite_output_size;
                    if(hexagonInterpreter_ != nullptr){
                        interpreter = hexagonInterpreter_;
                        tflite_output_size = interpreter->outputs().size();
                    }
                    else if(gpuInterpreter_ != nullptr){
                        interpreter = gpuInterpreter_;
                        tflite_output_size = interpreter->outputs().size();
                    }
                    else{
                        tflite_output_size = 0;
                    }

                    if(!gpu_queue_.empty() || !hexagon_queue_.empty()){
                        if(new_score_thrs_[0] >= 0){
                            cur_score_thrs_[0] = new_score_thrs_[0];

                            for(int i = 0; i < outputs_.size() - tflite_output_size; i++){
                                auto & outputVStream = hailoVstreams_->second[i];
                                auto & outputVStream2 = hailoVstreams2_->second[i];
                                auto & outputVStream3 = hailoVstreams3_->second[i];

                                auto status = outputVStream.set_nms_score_threshold(new_score_thrs_[0]);
                                if (HAILO_SUCCESS != status) {
                                    std::cerr << "ERROR: Failed to set score threshold to " << new_score_thrs_[0] << " for output vstream " << i << ", status = " << status << std::endl;
                                    return kTfLiteError;
                                }

                                status = outputVStream2.set_nms_score_threshold(new_score_thrs_[0]);
                                if (HAILO_SUCCESS != status) {
                                    std::cerr << "ERROR: Failed to set score threshold to " << new_score_thrs_[0] << " for output vstream " << i << ", status = " << status << std::endl;
                                    return kTfLiteError;
                                }

                                status = outputVStream3.set_nms_score_threshold(new_score_thrs_[0]);
                                if (HAILO_SUCCESS != status) {
                                    std::cerr << "ERROR: Failed to set score threshold to " << new_score_thrs_[0] << " for output vstream " << i << ", status = " << status << std::endl;
                                    return kTfLiteError;
                                }
                            }
                        }
                    }
                    else{
                        if(ori_score_thrs_[0] >= 0){
                            cur_score_thrs_[0] = ori_score_thrs_[0];

                            for(int i = 0; i < outputs_.size() - tflite_output_size; i++){
                                auto & outputVStream = hailoVstreams_->second[i];
                                auto & outputVStream2 = hailoVstreams2_->second[i];
                                auto & outputVStream3 = hailoVstreams3_->second[i];

                                auto status = outputVStream.set_nms_score_threshold(ori_score_thrs_[0]);
                                if (HAILO_SUCCESS != status) {
                                    std::cerr << "ERROR: Failed to set score threshold to " << ori_score_thrs_[0] << " for output vstream " << i << ", status = " << status << std::endl;
                                    return kTfLiteError;
                                }

                                status = outputVStream2.set_nms_score_threshold(ori_score_thrs_[0]);
                                if (HAILO_SUCCESS != status) {
                                    std::cerr << "ERROR: Failed to set score threshold to " << ori_score_thrs_[0] << " for output vstream " << i << ", status = " << status << std::endl;
                                    return kTfLiteError;
                                }

                                status = outputVStream3.set_nms_score_threshold(ori_score_thrs_[0]);
                                if (HAILO_SUCCESS != status) {
                                    std::cerr << "ERROR: Failed to set score threshold to " << ori_score_thrs_[0] << " for output vstream " << i << ", status = " << status << std::endl;
                                    return kTfLiteError;
                                }
                            }
                        }
                    }

                    turnaround_mutex_.lock();

                    for(int i = 0; i < batch_sizes_; i++){
                        batch_mutex_[i].lock();
                    }
                    
                    std::thread invoke_thread(Invoke_thread);
                    invoke_thread.detach();

                    return kTfLiteOk;

                    break;
                }
            }
        }

        InterpreterBuilder::InterpreterBuilder(const ::tflite::FlatBufferModel& model, const ::tflite::OpResolver& op_resolver, const ::tflite::InterpreterOptions* options_experimental)
        : ::tflite::InterpreterBuilder(model, op_resolver, options_experimental){
            //std::cout << "InterpreterBuilder Constructor\n";
        }

        TfLiteStatus InterpreterBuilder::operator()(std::unique_ptr<Interpreter>* interpreter){
            //std::cout << "InterpreterBuilder operator()\n";
            switch(mode_){
                case 0:
                {
                    if (!interpreter) {
                        std::cerr << "Null pointer is passed to InterpreterBuilder\n";
                        return kTfLiteError;
                    }

                    std::unique_ptr<::tflite::Interpreter> oldTypeInterpreter = static_unique_pointer_cast<::tflite::Interpreter, Interpreter>(std::move(*interpreter));
                    TfLiteStatus status = ::tflite::InterpreterBuilder::operator()(&oldTypeInterpreter);

                    *interpreter = static_unique_pointer_cast<Interpreter, ::tflite::Interpreter>(std::move(oldTypeInterpreter));

                    return status;

                    break;
                }
                case 1:
                case 2:
                case 3:
                case 4:
                {
                    *interpreter = std::make_unique<Interpreter>();

                    return kTfLiteOk;

                    break;
                }
            }
        }

        TfLiteStatus InterpreterBuilder::SetNumThreads(int num_threads){
            switch(mode_){
                case 0:
                {
                    return ::tflite::InterpreterBuilder::SetNumThreads(num_threads);

                    break;
                }
                case 1:
                case 2:
                {
                    return kTfLiteOk;

                    break;
                }
                case 3:
                case 4:
                {
                    if(gpuInterpreter_ != nullptr){
                        TfLiteStatus status = gpuInterpreter_->SetNumThreads(num_threads);
                        if(status != kTfLiteOk)
                            return status;
                    }
                    
                    if(hexagonInterpreter_ != nullptr){
                        TfLiteStatus status = hexagonInterpreter_->SetNumThreads(num_threads);
                        return status;
                    }
                }
            }
        }
    }
}
