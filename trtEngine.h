#pragma once

#include <NvOnnxParser.h>
#include <iostream>
#include <fstream>
#include <string>
#include "wqq.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
class Logger;

void createDenoisePLEngineDynamicInputSize(string onnxName, int height, int width);

void mockInferDenoisePLEngine(string engineName, int inputBatch, int inputChannel, int inputHeight, int inputWidth);

void createDenoisePLEngine(string onnxName,int size);

void inferenceDenoisePL(torch::Tensor td_norm);