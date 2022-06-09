//
// Created by sam on 09/05/22.
//
#include <esig/algebra/context.h>


namespace esig {
namespace algebra {


vector_construction_data::vector_construction_data(coefficient_type ctype, vector_type vtype)
        : m_data_begin(nullptr),
          m_data_end(nullptr),
          m_coeffs(ctype),
          m_vect_type(vtype),
          m_data_type(input_data_type::value_array),
          m_item_size(size_of(ctype))
{
}
vector_construction_data::vector_construction_data(const char *data_begin, const char *data_end, coefficient_type ctype, vector_type vtype, input_data_type idtype, dimn_t itemsize)
        : m_data_begin(data_begin),
          m_data_end(data_end),
          m_coeffs(ctype),
          m_vect_type(vtype),
          m_data_type(idtype),
          m_item_size(itemsize)
{
}
vector_construction_data::vector_construction_data(const coefficient *data_begin, const coefficient *data_end, vector_type vtype)
    : m_data_begin(reinterpret_cast<const char*>(data_begin)),
      m_data_end(reinterpret_cast<const char*>(data_end)),
      m_coeffs(data_begin->ctype()),
      m_vect_type(vtype),
      m_data_type(input_data_type::coeff_array),
      m_item_size(sizeof(coefficient))
{
}

vector_construction_data::vector_construction_data(const std::pair<key_type, coefficient> *data_begin, const std::pair<key_type, coefficient> *data_end, vector_type vtype)
    : m_data_begin(reinterpret_cast<const char*>(data_begin)),
      m_data_end(reinterpret_cast<const char*>(data_end)),
      m_coeffs(data_begin->second.ctype()),
      m_vect_type(vtype),
      m_data_type(input_data_type::key_coeff_array),
      m_item_size(sizeof(std::pair<key_type, coefficient>))
{}


const char *vector_construction_data::begin() const
{
    return m_data_begin;
}
const char *vector_construction_data::end() const
{
    return m_data_end;
}
coefficient_type vector_construction_data::ctype() const
{
    return m_coeffs;
}
vector_type vector_construction_data::vtype() const
{
    return m_vect_type;
}
input_data_type vector_construction_data::input_type() const
{
    return m_data_type;
}
dimn_t vector_construction_data::item_size() const
{
    return m_item_size;
}



}// namespace algebra
}// namespace esig
