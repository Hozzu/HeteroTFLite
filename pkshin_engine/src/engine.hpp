#include <cstring>
#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>

#include <opencv2/opencv.hpp>

#include <tensorflow/lite/delegates/hexagon/hexagon_delegate.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>

#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>

#include <maccel/maccel.h>
#include <hailo/hailort.hpp>

namespace pkshin{
    static int mode_ = 0; // 0 for tflite, 1 for maccel, 2 for hailo, 3 for tflite+maccel, 4 for tflite+hailo
    static char filename_[500];
    static char tflite_filename_[500];

    static std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> * hailoVstreams_;
    static std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> * hailoVstreams2_;
    static std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> * hailoVstreams3_;

    static mobilint::Model * mobilintModel_;

    static ::tflite::Interpreter * gpuInterpreter_ = nullptr;
    static ::tflite::Interpreter * hexagonInterpreter_ = nullptr;

    static std::vector<int> inputs_;
    static std::vector<TfLiteIntArray *> input_dims_;
    static std::vector<char *> input_names_;
    static std::vector<TfLiteTensor *> input_tensors_;
    static std::vector<void *> input_datas_;

    static std::vector<int> outputs_;
    static std::vector<TfLiteIntArray *> output_dims_;
    static std::vector<char *> output_names_;
    static std::vector<TfLiteTensor *> output_tensors_;
    static std::vector<void *> output_datas_;

    static int batch_sizes_ = 1;
    static std::vector<int> batch_run_ = {0};
    static std::vector<std::mutex> batch_mutex_ = std::vector<std::mutex>(1);

    static std::thread gpu_thread_;
    static std::mutex gpu_thread_mutex_;
    static std::condition_variable gpu_thread_set_cv_;
    static std::condition_variable gpu_thread_start_cv_;
    static std::condition_variable gpu_thread_end_cv_;

    static std::vector<float> perfs_ = {1, 1, 1, 1, 1};
    static std::vector<float> ori_score_thrs_ = {-1, -1};
    static std::vector<float> new_score_thrs_ = {-1, -1};
    static std::vector<float> cur_score_thrs_ = {0.001, 0.001};

    static auto invoke_start_ = std::chrono::high_resolution_clock::now();
    static std::vector<double> turnaround_ = {0};
    static double max_turnaround_ = 0;
    static double sum_turnaround_ = 0;
    static std::mutex turnaround_mutex_;

    static std::vector<int> gpu_queue_;
    static std::vector<int> hexagon_queue_;
    static std::vector<int> maccel_queue_;
    static std::vector<int> hailo_queue_;
    static std::vector<int> hailo_queue2_;
    static std::vector<int> hailo_queue3_;
}

namespace tflite{
    namespace pkshin{
        class FlatBufferModel : public ::tflite::FlatBufferModel {
            public:
            FlatBufferModel();

            static std::unique_ptr<FlatBufferModel> BuildFromFile(const char* filename, ::tflite::ErrorReporter* error_reporter = ::tflite::DefaultErrorReporter());
        };
    }

    namespace ops {
        namespace builtin {
            namespace pkshin{
                class BuiltinOpResolver : public ::tflite::ops::builtin::BuiltinOpResolver {
                    public:
                    BuiltinOpResolver();
                };

                class BuiltinOpResolverWithoutDefaultDelegates : public ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates {
                    public:
                    BuiltinOpResolverWithoutDefaultDelegates();
                };
            }
        }
    }

    namespace pkshin{
        class Interpreter : public ::tflite::Interpreter {
            public:
            Interpreter(::tflite::ErrorReporter* error_reporter = ::tflite::DefaultErrorReporter());

            ~Interpreter();

            std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> * get_hailo_vstreams();

            mobilint::Model * get_mobilint_model();

            bool is_tflite_model();

            bool is_hailo_output(int batch_id = 0);

            bool is_maccel_output(int batch_id = 0);

            bool is_tflite_output(int batch_id = 0);

            TfLiteStatus ModifyGraphWithDelegate(TfLiteDelegate* delegate);

            TfLiteStatus AllocateTensors();

            TfLiteStatus ResizeInputTensor(int tensor_index, const std::vector<int>& dims);

            const std::vector<int>& inputs();

            const std::vector<int>& outputs();

            const char* GetInputName(int index);

            const char* GetOutputName(int index);

            TfLiteTensor* input_tensor(size_t index);

            TfLiteTensor* output_tensor(size_t index);

            void * get_input_data(int index);

            void * get_output_data(int index);

            template <class T>
            T* typed_input_tensor(int index){
                if(is_tflite_model()){
                    return ::tflite::Interpreter::typed_input_tensor<T>(index);
                }
                else{
                    return (T*)get_input_data(index);
                }
            }

            TfLiteStatus SetSchedulerParams(std::vector<float> perfs);

            TfLiteStatus SetPostProcessParams(std::vector<float> ori_score_thrs, std::vector<float> new_score_thrs);

            std::vector<float> GetPostProcessParams();

            double GetSumTurnAroundTime();

            double GetMaxTurnAroundTime();

            TfLiteStatus Invoke();

            template <class T>
            T* typed_output_tensor(int index){
                if(is_tflite_model()){
                    return ::tflite::Interpreter::typed_output_tensor<T>(index);
                }
                else{
                    return (T*)get_output_data(index);
                }
            }
        };

        class InterpreterBuilder : public ::tflite::InterpreterBuilder {
            public:
            InterpreterBuilder(const ::tflite::FlatBufferModel& model, const ::tflite::OpResolver& op_resolver, const ::tflite::InterpreterOptions* options_experimental = nullptr);

            TfLiteStatus operator()(std::unique_ptr<Interpreter>* interpreter);

            TfLiteStatus SetNumThreads(int num_threads);
        };
    }
}