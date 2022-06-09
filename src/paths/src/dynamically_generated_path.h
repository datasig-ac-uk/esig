//
// Created by user on 05/04/2022.
//

#ifndef ESIG_PATHS_SRC_PATHS_SRC_DYNAMICALLY_GENERATED_PATH_H_
#define ESIG_PATHS_SRC_PATHS_SRC_DYNAMICALLY_GENERATED_PATH_H_

#include <esig/implementation_types.h>
#include <esig/paths/path.h>

#include <functional>

namespace esig {
namespace paths {

namespace dtl {

class function_increment_iterator : public algebra::data_iterator
{
    const char* m_begin;
    const char* m_end;
    bool state;

public:

    function_increment_iterator(const char* begin, const char* end);

    const char *dense_begin() override;
    const char *dense_end() override;
    bool next_sparse() override;
    const void *sparse_kv_pair() override;
    bool advance() override;
    bool finished() const override;
};



} // namespace dtl



} // namespace paths
} // namespace esig

#endif//ESIG_PATHS_SRC_PATHS_SRC_DYNAMICALLY_GENERATED_PATH_H_
