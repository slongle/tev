// This file was developed by Junchen Deng <junchendeng@gmail.com>.
// It is published under the BSD 3-Clause License within the LICENSE file.

#include <tev/imageio/NpyImageLoader.h>
#include <tev/ThreadPool.h>

#include <stb_image.h>

#include <regex>

using namespace nanogui;
using namespace std;

TEV_NAMESPACE_BEGIN

class Float16 {
public:
    Float16() {}
    Float16(const uint16_t& v) :bits(v) {}

    float ToFloat32() {
        IEEESingle sng;
        sng.IEEE.Sign = IEEE.Sign;

        if (!IEEE.Exp)
        {
            if (!IEEE.Frac)
            {
                sng.IEEE.Frac = 0;
                sng.IEEE.Exp = 0;
            }
            else
            {
                const float half_denorm = (1.0f / 16384.0f);
                float mantissa = ((float)(IEEE.Frac)) / 1024.0f;
                float sgn = (IEEE.Sign) ? -1.0f : 1.0f;
                sng.Float = sgn * mantissa * half_denorm;
            }
        }
        else if (31 == IEEE.Exp)
        {
            sng.IEEE.Exp = 0xff;
            sng.IEEE.Frac = (IEEE.Frac != 0) ? 1 : 0;
        }
        else
        {
            sng.IEEE.Exp = IEEE.Exp + 112;
            sng.IEEE.Frac = (IEEE.Frac << 13);
        }
        return sng.Float;
    }

    union
    {
        uint16_t bits;			// All bits
        struct
        {
            uint16_t Frac : 10;	// mantissa
            uint16_t Exp : 5;		// exponent
            uint16_t Sign : 1;		// sign
        } IEEE;
    };

    union IEEESingle
    {
        float Float;
        struct
        {
            uint32_t Frac : 23;
            uint32_t Exp : 8;
            uint32_t Sign : 1;
        } IEEE;
    };
};

bool NpyImageLoader::canLoadFile(istream& iStream) const {
    // Pretend you can load any file and throw exception on failure.
    std::string header;
    std::getline(iStream, header);
    if (header.length() < 10 || header.substr(1, 5) != "NUMPY") {
        return false;
    }
    return true;
}

Task<vector<ImageData>> NpyImageLoader::load(istream& iStream, const fs::path&, const string& channelSelector, int priority) const {    
    vector<ImageData> result(1);
    ImageData& resultData = result.front();

    std::string header;
    std::getline(iStream, header);
    if (header.length() < 10 || header.substr(1, 5) != "NUMPY") {
        throw invalid_argument{ "Not npy format." };
    }

    // Format
    auto pos = header.find("descr");
    if (pos == std::string::npos) {
        throw invalid_argument{ "" };
    }
    pos += 9;
    bool littleEndian = (header[pos] == '<' || header[pos] == '|' ? true : false);
    if (!littleEndian) {
        throw invalid_argument{ "Only supports little endian." };
    }
    char type = header[pos + 1];
    int byteSize = atoi(header.substr(pos + 2, 1).c_str()); // assume size <= 8 bytes
    if (type != 'f' && type != 'u' && byteSize > 4) {
        throw invalid_argument{ "Byte size > 4." };
    }

    // Order
    pos = header.find("fortran_order");
    if (pos == std::string::npos) {
        throw invalid_argument{ "" };
    }
    pos += 16;
    bool fortranOrder = header.substr(pos, 4) == "True" ? true : false;

    if (fortranOrder) { // Only supports C order.
        throw invalid_argument{ "Only support C order." };
    }

    // Shape
    auto offset = header.find("(") + 1;
    auto shapeString = header.substr(offset, header.find(")") - offset);
    std::regex regex("[0-9][0-9]*");
    std::smatch match;
    std::vector<int> shape;
    while (std::regex_search(shapeString, match, regex)) {
        shape.push_back(std::stoi(match[0].str()));
        shapeString = match.suffix().str();
    }
    int w = 0, h = 0, ch = 1;
    if (shape.size() < 2 || shape.size() > 4) { // support 2/3/4
        throw invalid_argument{ "Only support 2/3/4." };
    }
    if (shape.size() == 2) {
        h = shape[0];
        w = shape[1];
        ch = 1;
    }
    else if (shape.size() == 3) {
        h = shape[0];
        w = shape[1];
        ch = shape[2];
    }
    else if (shape.size() == 4) {
        //if (shape[0] > 1) // single image only
        //	return false;
        h = shape[1];
        w = shape[2];
        ch = shape[3];
    }

    if (ch > 4) { // at most 4 channel
        throw invalid_argument{ "Only support at most 4 channels." };
    }

    std::vector<char> data;
    data.resize(w * h * ch * byteSize);
    if (!iStream.read(data.data(), w * h * ch * byteSize)) {
        throw invalid_argument{ "No enough data." };
    }

    if (type != 'f' || (byteSize != 2 && byteSize != 4 && byteSize != 8)) {
        throw invalid_argument{ "Only support float16, float32 and float64." };
    }

    Vector2i size(w, h);
    if (size.x() == 0 || size.y() == 0) {
        throw invalid_argument{ "Image has zero pixels." };
    }

    int numChannels = ch;
    resultData.channels = makeNChannels(numChannels, size);

    if (byteSize == 2) {
        uint16_t* typedData = reinterpret_cast<uint16_t*>(data.data());
        co_await ThreadPool::global().parallelForAsync(0, size.y(), [&](int y) {
            for (int x = 0; x < size.x(); ++x) {
                int baseIdx = (y * size.x() + x) * numChannels;
                for (int c = 0; c < numChannels; ++c) {
                    float val = Float16(typedData[baseIdx + c]).ToFloat32();
                    resultData.channels[c].at({ x, y }) = val;
                }
            }
            }, priority);
    }
    else if (byteSize == 4) {
        float* typedData = reinterpret_cast<float*>(data.data());
        co_await ThreadPool::global().parallelForAsync(0, size.y(), [&](int y) {
            for (int x = 0; x < size.x(); ++x) {
                int baseIdx = (y * size.x() + x) * numChannels;
                for (int c = 0; c < numChannels; ++c) {
                    float val = typedData[baseIdx + c];
                    resultData.channels[c].at({ x, y }) = val;
                }
            }
            }, priority);
    }
    else if (byteSize == 8) {
        double* typedData = reinterpret_cast<double*>(data.data());
        co_await ThreadPool::global().parallelForAsync(0, size.y(), [&](int y) {
            for (int x = 0; x < size.x(); ++x) {
                int baseIdx = (y * size.x() + x) * numChannels;
                for (int c = 0; c < numChannels; ++c) {
                    float val = typedData[baseIdx + c];
                    resultData.channels[c].at({ x, y }) = val;
                }
            }
            }, priority);
    }

    resultData.hasPremultipliedAlpha = false;

    co_return result;
}

TEV_NAMESPACE_END
