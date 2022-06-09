//
// Created by user on 06/03/2022.
//

#ifndef ESIG_PATHS_SRC_ALGEBRA_NON_TEMPLATED_IMPL_HALL_SET_H_
#define ESIG_PATHS_SRC_ALGEBRA_NON_TEMPLATED_IMPL_HALL_SET_H_

#include <esig/implementation_types.h>
#include <vector>
#include <map>
#include <utility>
#include <string>
#include <boost/container/flat_map.hpp>


namespace esig {
namespace algebra {



class hall_set
{
public:
    using letter_type = unsigned;
    using degree_type = unsigned;
    using key_type = std::size_t; // replace me later
    using size_type = std::size_t;
    using parent_type = std::pair<key_type, key_type>;
private:
    using data_type = std::vector<parent_type>;
    using reverse_map_type = boost::container::flat_map<parent_type, key_type>;
    using l2k_map_type = std::vector<key_type>;
    using size_vector_type = std::vector<size_type>;
    using degree_range_map_type = std::vector<std::pair<size_type, size_type>>;

    degree_type width;
    degree_type current_degree;

    std::vector<letter_type> letters;
    data_type data;
    reverse_map_type reverse_map;
    l2k_map_type l2k;
    degree_range_map_type degree_ranges;
    size_vector_type sizes;

public:
    static constexpr key_type root_element { 0 };
    static constexpr parent_type root_parent { 0, 0 };


    explicit hall_set(degree_type width, degree_type depth=1);
    void grow_up(degree_type new_depth);

    key_type key_of_letter(let_t) const noexcept;
    size_type size(deg_t) const noexcept;
    bool letter(const key_type&) const noexcept;
    degree_type degree(const key_type&) const noexcept;

    const parent_type& operator[](const key_type&) const noexcept;
    const key_type& operator[](const parent_type&) const;
};


//constexpr typename hall_set::key_type hall_set::root_element;
//constexpr typename hall_set::parent_type hall_set::root_parent;

} // namespace algebra
} // namespace paths-old

#endif//ESIG_PATHS_SRC_ALGEBRA_NON_TEMPLATED_IMPL_HALL_SET_H_
