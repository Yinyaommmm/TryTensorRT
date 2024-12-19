#include "stdafx.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>

#include "wqq.h"
#include "trtEngine.h"
#include "preprocess.h"



int main() {
	std::cout << "Cuda available: " << torch::cuda::is_available() << std::endl; // 测试CUDA是否可用
	std::cout << "CuDNN available : " << torch::cuda::cudnn_is_available() << std::endl; // 测试CUDNN是否可用
	
	
	//createDenoisePLEngine("./onnx/Denoising_Planaria_1_256_256_float16",256);
	//mockInferDenoisePLEngine("./onnx/Denoising_Planaria_1_256_256_float16",1,1,256,256);
	 
	//mockInferDenoisePLEngine("./onnx/Denoising_Planaria_1_1024_1024_float16",4,1,1024,1024);

	// 读取tif并进行推理
	//TIFF* tif = readTif("Planaria_C1.tif");
	//TiffData image = convertTif2FlatVec(tif);
	//auto td_norm = preprocessDenoisePL(image);
	//inferenceDenoisePL(td_norm);

 //   auto image = torch::randn({ 1, 3, 1024, 1024 });
	//auto f = torch::nn::functional::InterpolateFuncOptions();

	//f.mode(torch::kBilinear);
	//f.align_corners(true);
	//f.size(std::vector<int64_t>({ 512, 512 }));

	//auto resized_image = torch::nn::functional::interpolate(
	//	image,
	//	f
	//);
	//cout << resized_image.sizes() << endl;
	

}