//
// Created by user on 09/12/22.
//

#include "py_fmt_to_esig_fmt.h"

#include <esig/scalars.h>

using namespace esig;
using namespace esig::scalars;

std::string esig::python::py_fmt_to_esig_fmt(const std::string &fmt) {
    char python_format = 0;
    for (const auto &chr : fmt) {
        switch (chr) {
            case '<':// little-endian
#if BOOST_ENDIAN_BIG_BYTE || BOOST_ENDIAN_BIG_WORD
                throw std::runtime_error("non-native byte ordering not supported");
#endif
            case '>':// big-endian
#if BOOST_ENDIAN_LITTLE_BYTE || BOOST_ENDIAN_LITTLE_WORD
                throw std::runtime_error("non-native byte ordering not supported");
#endif
            case '@':// native
            case '=':// native
#if BOOST_ENDIAN_LITTLE_BYTE || BOOST_ENDIAN_LITTLE_WORD
                break;
#endif
            case '!':// network ( = big-endian )
#if BOOST_ENDIAN_LITTLE_BYTE || BOOST_ENDIAN_LITTLE_WORD
                throw std::runtime_error("non-native byte ordering not supported");
#else
                break;
#endif
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                throw std::runtime_error("format must be a single letter");
            default:
                python_format = chr;
                goto after_loop;
        }
    }

after_loop:
    std::string format;
    switch (python_format) {
        case 'd':
            format = "f64";
            break;
        case 'f':
            format = "f32";
            break;
        case 'l':
        case 'q':
            format = "i64";
            break;
        case 'L':
        case 'Q':
            format = "u64";
            break;
        case 'i':
            format = "i32";
            break;
        case 'I':
            format = "u32";
            break;
        case 'n':
            format = "isize";
            break;
        case 'N':
            format = "usize";
            break;
        case 'h':
            format = "i16";
            break;
        case 'H':
            format = "u16";
            break;
        case 'b':
        case 'c':
            format = "i8";
            break;
        case 'B':
            format = "u8";
            break;
        default:
            throw std::runtime_error("Unrecognised data format");
    }

    return format;
}
