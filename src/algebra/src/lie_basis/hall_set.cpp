//
// Created by user on 06/03/2022.
//

#include "hall_set.h"
#include <stdexcept>
#include <algorithm>

namespace esig {
namespace algebra {


hall_set::hall_set(hall_set::degree_type width, hall_set::degree_type depth)
    : width(width), current_degree(0)
{
    data.reserve(1 + width);
    letters.reserve(width);
    sizes.reserve(2);
    l2k.reserve(width);
    degree_ranges.reserve(2);

    data.push_back({0, 0});
    degree_ranges.push_back({0, 1});
    sizes.push_back(0);

    for (letter_type l = 1; l <= width; ++l) {
        parent_type parents{0, l};
        letters.push_back(l);
        data.push_back(parents);
        reverse_map.insert(std::pair<parent_type, key_type>(parents, data.size()));
        l2k.push_back(l);
    }

    std::pair<size_type, size_type> range {
            degree_ranges[current_degree].second,
            data.size()
    };

    degree_ranges.push_back(range);
    sizes.push_back(width);
    ++current_degree;

    if (depth > 1) {
        grow_up(depth);
    }
}

void hall_set::grow_up(hall_set::degree_type new_depth)
{

    for (degree_type d = current_degree + 1; d <= new_depth; ++d) {
        for (degree_type e = 1; 2 * e <= d; ++e) {
            letter_type i_lower, i_upper, j_lower, j_upper;
            i_lower = degree_ranges[e].first;
            i_upper = degree_ranges[e].second;
            j_lower = degree_ranges[d - e].first;
            j_upper = degree_ranges[d - e].second;

            for (letter_type i = i_lower; i < i_upper; ++i) {
                for (letter_type j = std::max(j_lower, i + 1); j < j_upper; ++j) {
                    if (data[j].first <= i) {
                        parent_type parents(i, j);
                        data.push_back(parents);
                        reverse_map[parents] = data.size() - 1;
                    }
                }
            }
        }

        std::pair<size_type, size_type> range;
        range.first = degree_ranges[current_degree].second;
        range.second = data.size();
        degree_ranges.push_back(range);
        // The hall set contains an entry for the "god element" 0,
        // so subtract one from the size.
        sizes.push_back(data.size() - 1);

        ++current_degree;
    }
}
hall_set::degree_type hall_set::degree(const hall_set::key_type &) const noexcept
{
    return 0;
}
hall_set::key_type hall_set::key_of_letter(let_t let) const noexcept
{
    return letters[let];
}
hall_set::size_type hall_set::size(deg_t deg) const noexcept
{
    if (deg < sizes.size()) {
        return sizes[deg];
    }
    return sizes.back();
}
bool hall_set::letter(const hall_set::key_type &key) const noexcept
{
    return std::find(letters.begin(), letters.end(), key) != letters.end();
}
const hall_set::parent_type &hall_set::operator[](const hall_set::key_type &key) const noexcept
{
    return data[key];
}
const hall_set::key_type &hall_set::operator[](const hall_set::parent_type &parent) const
{
    auto found = reverse_map.find(parent);
    if (found != reverse_map.end()) {
        return found->second;
    }
    return root_element;
}

constexpr typename hall_set::key_type hall_set::root_element;
constexpr typename hall_set::parent_type hall_set::root_parent;


} // namespace algebra_old
} // namespace esig_paths
