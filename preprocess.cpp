#include "stdafx.h"
#include "preprocess.h"

using std::cout;

size_t TiffData::channel() {
    return this->dims[0];
}
size_t TiffData::height() {
    return this->dims[1];
}
size_t TiffData::width() {
    return this->dims[2];
}
TiffData::TiffData() {
    this->v = vector<tifpixel>();
    this->dims = { 0,0,0 };
}
TiffData::TiffData(std::vector<tifpixel>& pixel_data, size_t c, size_t h, size_t w) {
    this->v = pixel_data;
    this->dims = { c, h, w };
}
tifpixel TiffData::at(size_t c, size_t h, size_t w) {
    size_t first_base = c * (this->height() * this->width());
    size_t second_base = h * this->width();
    return this->v[first_base + second_base + w];
}


std::tuple<int, int, int> getTiffShape(TIFF* tif) {
    // 打开 TIFF 文件
    if (!tif) {
        throw std::runtime_error("getTiffShape: Nullptr");
    }

    int width = 0;
    int height = 0;
    int nTotalFrame = TIFFNumberOfDirectories(tif);  // 获取帧数

    // 获取宽度和高度
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

    return  std::make_tuple(nTotalFrame,width, height );
}
void printImageShape(cv::Mat image) {
    // 打印图像的尺寸和通道数
    int height = image.rows;   // 图像高度（行数）
    int width = image.cols;    // 图像宽度（列数）
    int channels = image.channels(); // 图像通道数
    cout << "Image shape: (" << channels << ", "<< height << ", " << width   << ")" << std::endl;
}
void printTiffShape(TIFF* tif) {
    if (!tif) {
        throw std::runtime_error("getTiffShape: Nullptr");
    }
    auto shape = getTiffShape(tif);
    printf("TiffShape : (%d,%d,%d)\n", std::get<0>(shape), std::get<1>(shape), std::get<2>(shape));
}
TIFF* readTif(std::string filePath) {
    // 使用 OpenCV 的 imread 函数读取 TIFF 文件
    //cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR); // 以彩色读取
    TIFF* tif = TIFFOpen(filePath.c_str(), "r");
	if (tif == nullptr)
	{
		cout << "读入图像路径错误,请重新确认";
		return nullptr;
	}
    auto shape = getTiffShape(tif);
    printTiffShape(tif);
	return tif;
}
TiffData convertTif2FlatVec(TIFF* tif) {
    if (!tif) {
        throw std::runtime_error("converTif2Array: Nullptr");
        return TiffData();
    }
    cout << "convertTif2FlatVec将tiff数据转化成uint16" << endl;
    auto shape = getTiffShape(tif);
    int numPages = std::get<0>(shape);
    int height = std::get<1>(shape);
    int width = std::get<2>(shape);
    int bytePerPixel = 2; // 根据python读取得知
    uint32_t rowSize = width * bytePerPixel; 
    uint32_t totalPageSize = rowSize * height;
    size_t totalSize = numPages * totalPageSize;

    // 使用 vector 分配内存
    vector<tifpixel> allImageData(numPages*height*width);
    // 遍历每一页
    for (int page = 0; page < numPages; ++page) {
        TIFFSetDirectory(tif, page);
        // 读取每一行的像素数据
        size_t base = page * height * width;
        for (int row = 0; row < height; ++row) {
            size_t offset = static_cast<size_t>(row * width);
            size_t pos = static_cast<size_t>(base + offset);
            if (TIFFReadScanline(tif, &allImageData[pos],row) < 0) {
                std::cerr << "读取扫描线失败，页 " << page << "，行 " << row << endl;
                TIFFClose(tif);
                return TiffData();
            }
        }
    }
    return TiffData(allImageData, numPages, height, width);
}
torch::Tensor myQuantile(const torch::Tensor t, const torch::Tensor q) {
    assert(t.dtype().name() == "float");
    assert(q.dtype().name() == "float");
    if (!torch::equal(q, std::get<0>(q.sort()))) {
        throw std::runtime_error("quantiles q are not sorted");
    }

    auto tmp = t.clone().flatten();
    auto res = torch::empty_like(q);

    auto start = tmp.data_ptr<float>();
    auto end = tmp.data_ptr<float>() + tmp.size(0);

    for (int i = 0; i < q.size(0); i++) {
        auto m = tmp.data_ptr<float>() + static_cast<size_t>((tmp.size(0) - 1) * q[i].item<float>());
        std::nth_element(start, m, end);
        res[i] = *m;
        start = m;
    }

    return res;
}
torch::Tensor normalizeAtBefore(torch::Tensor t) {
 
    if (t.sizes().size() != 4) {
        throw std::runtime_error("normalizeAtBeofre: Must be CZYX style tensor");
    }
    if (t.dtype() != torch::kFloat32 && t.dtype() != torch::kFloat16) {
        throw std::runtime_error("normalizeAtBeofre: Not Float Tensor");
    }
   
    auto minPercent = torch::tensor({ 0.02 }, torch::kFloat32);
    auto maxPercent = torch::tensor({ 0.998 }, torch::kFloat32);
    torch::Tensor minV = myQuantile(t, minPercent);
    torch::Tensor maxV = myQuantile(t, maxPercent);
    t = (t - minV) / (maxV - minV + 1e-20);
    cout << "At Normalized Before: minV " << minV.item() << " maxV: " << maxV.item() << endl;
    return t;
}
torch::Tensor preprocessDenoisePL(TiffData& td) {
    auto c = static_cast<long long>(td.channel());
    auto h = static_cast<long long>(td.height());
    auto w = static_cast<long long>(td.width());
    auto td_4dim_f32 = torch::from_blob(td.v.data(), { c,h,w }, torch::kUInt16)
        .unsqueeze(0)
        .to(torch::kFloat32);
    return  normalizeAtBefore(td_4dim_f32);
}