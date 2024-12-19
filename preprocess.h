#include <opencv2/opencv.hpp>
#include <iostream>
#include <tiffio.h>
#include <vector>

using std::vector;
using std::endl;
typedef uint16_t tifpixel;
class TiffData {
public:
	vector<tifpixel> v;
	vector<size_t> dims;
	TiffData();
	TiffData(vector<tifpixel>& pixel_data, size_t c, size_t h, size_t w);
	size_t channel();
	size_t height();
	size_t width();
	tifpixel at(size_t c, size_t h, size_t w);
};

void printImageShape(cv::Mat image);
void printTiffShape(TIFF* tif);

TIFF* readTif(std::string filePath);
TiffData convertTif2FlatVec(TIFF* tif);
torch::Tensor myQuantile(const torch::Tensor t, const torch::Tensor q);
torch::Tensor preprocessDenoisePL(TiffData& td);
torch::Tensor normalizeAtBefore(torch::Tensor t);