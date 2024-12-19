
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

    //--------����ת����Engine��config
    auto config = builder->createBuilderConfig();
    if (builder->platformHasFastFp16())
        cout << "createDenoisePLEngine �� �豸֧��FP16�������Ȳ���" << endl;
        //config->setFlag(BuilderFlag::kFP16); // ���豸֧��FP16������ʹ��FP16ģʽ
    size_t free, total;
    cudaMemGetInfo(&free, &total);  // ��ȡ�豸�Դ���Ϣ
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, free); // �����п����Դ���������

   
    //--------���ú�engine�ļ����������ã���Ϊ����batch�����̬�仯��
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    int minBatch = 1,maxBatch = 4;
    Dims4 inputDimsSmall(minBatch, 1, height, width);  // ��СҲ�����ųߴ�
    Dims4 inputDimsBig(maxBatch, 1, height, width);  // ���ߴ�
    profile->setDimensions("input", OptProfileSelector::kMIN, inputDimsSmall);  // ��С�ߴ�
    profile->setDimensions("input", OptProfileSelector::kOPT, inputDimsSmall);  // ���ųߴ�
    profile->setDimensions("input", OptProfileSelector::kMAX, inputDimsBig);  // ���ߴ�
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
    print("�����ڴ�����Logger�����Ѿ�������");
    std::ifstream engineFile(engineName+".engine", std::ios::binary);
    std::string engineData((std::istreambuf_iterator<char>(engineFile)), std::istreambuf_iterator<char>());
    engineFile.close();

    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());

    // ģ����������
    unsigned long long inputSize = static_cast<unsigned long long>(inputBatch) * inputChannel * inputHeight * inputWidth;
    float* inputData = new float[inputSize]();
    void* buffers[2];  // ��������������

    cudaMalloc(&buffers[0], sizeof(inputData));  // ���뻺����
    cudaMalloc(&buffers[1], sizeof(float) * inputSize);  // ���������
    cudaMemcpy(buffers[0], inputData, sizeof(inputData), cudaMemcpyHostToDevice);

    IExecutionContext* context = engine->createExecutionContext();
    Dims4 inputDims(inputBatch, inputChannel, inputHeight, inputWidth);
    context->setInputShape("input", inputDims);
    context->executeV2(buffers);

    // �����������
    float* outputData = new float[inputSize]();
    cudaMemcpy(outputData, buffers[1], sizeof(float) * inputSize, cudaMemcpyDeviceToHost);
    cout << "Mock Output: " << outputData[0] << ' ' << outputData[1] << endl;

    // �ͷ���Դ
    delete[] outputData;
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    delete context;
    delete runtime;
    delete engine;
}


// ��������ָ��
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

    //--------����ת����Engine��config
    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (builder->platformHasFastFp16()) {
        //cout << "createDenoisePLEngine �� �豸֧��FP16�������Ȳ���" << endl;
        cout << "����Ϊfloat16����" << endl;
        config->setFlag(BuilderFlag::kFP16); // ���豸֧��FP16������ʹ��FP16ģʽ
    }
    else {
        cout << "����Ϊfloat32����" << endl;
    }
    size_t free, total;
    cudaMemGetInfo(&free, &total);  // ��ȡ�豸�Դ���Ϣ
    size_t use = free;
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, use); // �����п����Դ���������
    cout << "[Wqq] : use gpu mem: " << use << endl;

    //--------���ú�engine�ļ����������ã���Ϊ����batch�����̬�仯��
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    int minBatch = 1, maxBatch = 4;
    Dims4 inputDimsSmall(minBatch, 1, size, size);  // ��С�ߴ磬,���ҵ������ųߴ�
    Dims4 inputDimsBig(maxBatch, 1, size, size);  // ���ߴ�
    profile->setDimensions("input", OptProfileSelector::kMIN, inputDimsSmall);  // ��С�ߴ�
    profile->setDimensions("input", OptProfileSelector::kOPT, inputDimsSmall);  // ���ųߴ�
    profile->setDimensions("input", OptProfileSelector::kMAX, inputDimsBig);  // ���ߴ�
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
        // ȷ���õ�ǰBatch����ʼ�ͽ���channel
        int start_channel = i * batchStep;
        int end_channel = std::min(static_cast<int64_t>((i + 1) * batchStep), td_norm.size(1));
        auto td_batch = td_norm.slice(1, start_channel, end_channel);  // Shape: [1, batch_size, 1024, 1024]
        cout << td_batch.sizes() << endl;
    }

}