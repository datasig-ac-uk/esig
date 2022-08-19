//
// Created by sam on 19/08/22.
//


#include <esig/paths/path.h>


using namespace esig;


paths::path paths::concatenate(std::vector<paths::path> paths)
{
    std::vector<std::unique_ptr<const paths::path_interface>> data;
    data.reserve(paths.size());

    for (auto p : paths) {
        data.emplace_back(paths::path_base_access::take(std::move(p)));
    }


}
