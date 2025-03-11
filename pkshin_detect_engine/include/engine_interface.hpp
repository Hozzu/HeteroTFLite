#ifndef _ENGINE_INTERFACE_
#define _ENGINE_INTERFACE_

#include <tensorflow/lite/delegates/hexagon/hexagon_delegate.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>

#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>

#include <maccel/maccel.h>
#include <hailo/hailort.hpp>

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


#define FlatBufferModel pkshin::FlatBufferModel
#define BuiltinOpResolver pkshin::BuiltinOpResolver
#define BuiltinOpResolverWithoutDefaultDelegates pkshin::BuiltinOpResolverWithoutDefaultDelegates
#define Interpreter pkshin::Interpreter
#define InterpreterBuilder pkshin::InterpreterBuilder

#endif //_ENGINE_INTERFACE_