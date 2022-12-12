//
// Created by user on 24/05/22.
//

#ifndef ESIG_SRC_PATHS_SRC_TICK_PATH_H_
#define ESIG_SRC_PATHS_SRC_TICK_PATH_H_

#include <esig/implementation_types.h>
#include <esig/paths/path.h>

#include <boost/container/flat_map.hpp>

namespace esig {
namespace paths {

struct tick_entry {
    const char* ptr;
    dimn_t count;
};


class tick_increment_iterator : public algebra::data_iterator
{
    using iterator_type = typename boost::container::flat_map<param_t, tick_entry>::const_iterator;

    iterator_type m_current;
    iterator_type m_end;
    dimn_t m_index;
    dimn_t m_pair_size;

public:

    tick_increment_iterator(iterator_type b, iterator_type e, dimn_t pair_size);

    const char *dense_begin() override;
    const char *dense_end() override;
    bool next_sparse() override;
    const void *sparse_kv_pair() override;
    bool advance() override;
    bool finished() const override;
};

/**
 * @brief Path constructed from an ordered stream of events.
 *
 * A tick path is a stream of Lie increment events over some parameter interval.
 * An event indicates some change in the state - represented as a sparse Lie
 * increment - of the world, marked according to the "time" at which the event
 * occurs.
 *
 * Internally the stream of data is represented in two parts: the data, held in a
 * flat data buffer; and the tick information, held in flat-map indexed by parameter
 * stamps and pointing to tick events. A tick event is a pair containing a pointer
 * into the data buffer and a count, indicating the number of Hall-word and
 * scalar pairs are associated with that event.
 *
 * This path type implements a dyadic caching layer to help speed up the computation
 * of signatures and log-signatures.
 */
class tick_path : public dyadic_caching_layer
{
    algebra::allocating_data_buffer m_data;
    boost::container::flat_map<param_t, tick_entry> m_index;

public:

    tick_path(path_metadata&& md,
              std::vector<std::pair<param_t, dimn_t>>&& index_data,
              algebra::allocating_data_buffer&& data);
    algebra::lie log_signature(const interval &domain, const algebra::context &ctx) const override;
};

}// namespace paths
}// namespace esig

#endif//ESIG_SRC_PATHS_SRC_TICK_PATH_H_
