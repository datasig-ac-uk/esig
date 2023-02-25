//
// Created by user on 24/05/22.
//

#ifndef ESIG_SRC_PATHS_SRC_TICK_PATH_H_
#define ESIG_SRC_PATHS_SRC_TICK_PATH_H_

#include "esig/implementation_types.h"
#include "path.h"

#include <boost/container/flat_map.hpp>

namespace esig {
namespace paths {

struct tick_entry {
    scalars::scalar_array data_ptr;
    const key_type* key_ptr;
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
    scalars::key_scalar_array m_data;
    boost::container::flat_map<param_t, tick_entry> m_index;

public:

    tick_path(path_metadata&& md,
              std::vector<std::pair<param_t, std::vector<key_type>>>&& index_data,
              scalars::owned_scalar_array&& data);
    algebra::lie log_signature(const interval &domain, const algebra::context &ctx) const override;
};

}// namespace paths
}// namespace esig

#endif//ESIG_SRC_PATHS_SRC_TICK_PATH_H_
