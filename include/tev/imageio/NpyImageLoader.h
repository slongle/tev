// This file was developed by Junchen Deng <junchendeng@gmail.com>.
// It is published under the BSD 3-Clause License within the LICENSE file.

#pragma once

#include <tev/Image.h>
#include <tev/imageio/ImageLoader.h>

#include <istream>

TEV_NAMESPACE_BEGIN

class NpyImageLoader : public ImageLoader {
public:
    bool canLoadFile(std::istream& iStream) const override;
    Task<std::vector<ImageData>> load(std::istream& iStream, const fs::path& path, const std::string& channelSelector, int priority) const override;

    std::string name() const override {
        return "NPY";
    }
};

TEV_NAMESPACE_END
