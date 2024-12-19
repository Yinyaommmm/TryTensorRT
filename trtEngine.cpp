
#include "stdafx.h"
# include "trtEngine.h"
using namespace nvinfer1;
class Logger :public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << msg << std::endl;
        }
    }
};

void createDenoisePLEngineDynamicInputSize(string onnxName,int height ,int width) {
    Logger logger;
    IBuilder* builder = createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    std::string onnx_file = onnxName + ".onnx";
    auto parser = nvonnxparser::createParser(*network, logger);
    parser->parseFromFile(onnx_file.c_str(), static_cast<int>(ILogger::Severity::kINFO));
    network->getInput(0)->setName("input");
    network->getOutput(0)->setName("output");

    //--------设置转化成Engine的config
    auto config = builder->createBuilderConfig();
    if (builder->platformHasFastFp16())
        cout << "createDenoisePLEngine ： 设备支持FP16但现在先不搞" << endl;
        //config->setFlag(BuilderFlag::kFP16); // 若设备支持FP16推理，则使用FP16模式
    size_t free, total;
    cudaMemGetInfo(&free, &total);  // 获取设备显存信息
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, free); // 将所有空余显存用于推理

   
    //--------设置好engine文件的最优配置，因为存在batch这个动态变化量
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    int minBatch = 1,maxBatch = 4;
    Dims4 inputDimsSmall(minBatch, 1, height, width);  // 最小也是最优尺寸
    Dims4 inputDimsBig(maxBatch, 1, height, width);  // 最大尺寸
    profile->setDimensions("input", OptProfileSelector::kMIN, inputDimsSmall);  // 最小尺寸
    profile->setDimensions("input", OptProfileSelector::kOPT, inputDimsSmall);  // 最优尺寸
    profile->setDimensions("input", OptProfileSelector::kMAX, inputDimsBig);  // 最大尺寸
    config->addOptimizationProfile(profile);

    print("Build Engine With Config!!");
    auto engine = builder->buildEngineWithConfig(*network, *config);
    print("Build Stop!");
    IHostMemory* serializedModel = engine->serialize();
    std::ofstream engineFile(onnxName + ".engine", std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    engineFile.close();


    delete config;
    delete parser;
    delete network;
    delete builder;
    delete engine;
   
}

void mockInferDenoisePLEngine(string engineName ,int inputBatch ,int inputChannel, int inputHeight , int inputWidth) {
    Logger logger;
    print("函数内创建的Logger可能已经被销毁");
    std::ifstream engineFile(engineName+".engine", std::ios::binary);
    std::string engineData((std::istreambuf_iterator<char>(engineFile)), std::istreambuf_iterator<char>());
    engineFile.close();

    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());

    // 模拟输入数据
    unsigned long long inputSize = static_cast<unsigned long long>(inputBatch) * inputChannel * inputHeight * inputWidth;
    float* inputData = new float[inputSize]();
    void* buffers[2];  // 存放输入输出数据

    cudaMalloc(&buffers[0], sizeof(inputData));  // 输入缓冲区
    cudaMalloc(&buffers[1], sizeof(float) * inputSize);  // 输出缓冲区
    cudaMemcpy(buffers[0], inputData, sizeof(inputData), cudaMemcpyHostToDevice);

    IExecutionContext* context = engine->createExecutionContext();
    Dims4 inputDims(inputBatch, inputChannel, inputHeight, inputWidth);
    context->setInputShape("input", inputDims);
    context->executeV2(buffers);

    // 处理输出数据
    float* outputData = new float[inputSize]();
    cudaMemcpy(outputData, buffers[1], sizeof(float) * inputSize, cudaMemcpyDeviceToHost);
    cout << "Mock Output: " << outputData[0] << ' ' << outputData[1] << endl;

    // 释放资源
    delete[] outputData;
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    delete context;
    delete runtime;
    delete engine;
}


// 用了智能指针
void createDenoisePLEngine(string onnxName,int size ) {
    Logger logger;
    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(logger));
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch));


    std::string onnx_file = onnxName + ".onnx";
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    parser->parseFromFile(onnx_file.c_str(), static_cast<int>(ILogger::Severity::kINFO));
    network->getInput(0)->setName("input");
    network->getOutput(0)->setName("output");

    //--------设置转化成Engine的config
    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (builder->platformHasFastFp16()) {
        //cout << "createDenoisePLEngine ： 设备支持FP16但现在先不搞" << endl;
        cout << "导出为float16精度" << endl;
        config->setFlag(BuilderFlag::kFP16); // 若设备支持FP16推理，则使用FP16模式
    }
    else {
        cout << "导出为float32精度" << endl;
    }
    size_t free, total;
    cudaMemGetInfo(&free, &total);  // 获取设备显存信息
    size_t use = free;
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, use); // 将所有空余显存用于推理
    cout << "[Wqq] : use gpu mem: " << use << endl;

    //--------设置好engine文件的最优配置，因为存在batch这个动态变化量
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    int minBatch = 1, maxBatch = 4;
    Dims4 inputDimsSmall(minBatch, 1, size, size);  // 最小尺寸，,暂且当作最优尺寸
    Dims4 inputDimsBig(maxBatch, 1, size, size);  // 最大尺寸
    profile->setDimensions("input", OptProfileSelector::kMIN, inputDimsSmall);  // 最小尺寸
    profile->setDimensions("input", OptProfileSelector::kOPT, inputDimsSmall);  // 最优尺寸
    profile->setDimensions("input", OptProfileSelector::kMAX, inputDimsBig);  // 最大尺寸
    config->addOptimizationProfile(profile);
    ;
    print("Build Engine With Config!!");
    auto engine = std::unique_ptr<ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    print("Build Stop!");
    IHostMemory* serializedModel = engine->serialize();
    std::ofstream engineFile(onnxName + ".engine", std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    engineFile.close();

}


void inferenceDenoisePL(torch::Tensor td_norm) {
    int nGPU = 1;
    int batchStep = 4 * nGPU;
    for (int i = 0; static_cast<long long>(i) * batchStep < td_norm.sizes()[1]; i++) {
        // 确定好当前Batch的起始和结束channel
        int start_channel = i * batchStep;
        int end_channel = std::min(static_cast<int64_t>((i + 1) * batchStep), td_norm.size(1));
        auto td_batch = td_norm.slice(1, start_channel, end_channel);  // Shape: [1, batch_size, 1024, 1024]
        cout << td_batch.sizes() << endl;
    }

}