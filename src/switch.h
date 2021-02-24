switch (width) {
    case 2 :
    switch (depth) {
        case 2 :
        return TemplatedFn(2, 2);
        break;

        case 3 :
        return TemplatedFn(2, 3);
        break;

        case 4 :
        return TemplatedFn(2, 4);
        break;

        case 5 :
        return TemplatedFn(2, 5);
        break;

        case 6 :
        return TemplatedFn(2, 6);
        break;

        case 7 :
        return TemplatedFn(2, 7);
        break;

        case 8 :
        return TemplatedFn(2, 8);
        break;

        case 9 :
        return TemplatedFn(2, 9);
        break;

        case 10 :
        return TemplatedFn(2, 10);
        break;

        case 11 :
        return TemplatedFn(2, 11);
        break;

        case 12 :
        return TemplatedFn(2, 12);
        break;

        case 13 :
        return TemplatedFn(2, 13);
        break;

        case 14 :
        return TemplatedFn(2, 14);
        break;

        case 15 :
        return TemplatedFn(2, 15);
        break;

        case 16 :
        return TemplatedFn(2, 16);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->16 for records with width 2 exceeds limit" );
    }
    break;

    case 3 :
    switch (depth) {
        case 2 :
        return TemplatedFn(3, 2);
        break;

        case 3 :
        return TemplatedFn(3, 3);
        break;

        case 4 :
        return TemplatedFn(3, 4);
        break;

        case 5 :
        return TemplatedFn(3, 5);
        break;

        case 6 :
        return TemplatedFn(3, 6);
        break;

        case 7 :
        return TemplatedFn(3, 7);
        break;

        case 8 :
        return TemplatedFn(3, 8);
        break;

        case 9 :
        return TemplatedFn(3, 9);
        break;

        case 10 :
        return TemplatedFn(3, 10);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->10 for records with width 3 exceeds limit" );
    }
    break;

    case 4 :
    switch (depth) {
        case 2 :
        return TemplatedFn(4, 2);
        break;

        case 3 :
        return TemplatedFn(4, 3);
        break;

        case 4 :
        return TemplatedFn(4, 4);
        break;

        case 5 :
        return TemplatedFn(4, 5);
        break;

        case 6 :
        return TemplatedFn(4, 6);
        break;

        case 7 :
        return TemplatedFn(4, 7);
        break;

        case 8 :
        return TemplatedFn(4, 8);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->8 for records with width 4 exceeds limit" );
    }
    break;

    case 5 :
    switch (depth) {
        case 2 :
        return TemplatedFn(5, 2);
        break;

        case 3 :
        return TemplatedFn(5, 3);
        break;

        case 4 :
        return TemplatedFn(5, 4);
        break;

        case 5 :
        return TemplatedFn(5, 5);
        break;

        case 6 :
        return TemplatedFn(5, 6);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->6 for records with width 5 exceeds limit" );
    }
    break;

    case 6 :
    switch (depth) {
        case 2 :
        return TemplatedFn(6, 2);
        break;

        case 3 :
        return TemplatedFn(6, 3);
        break;

        case 4 :
        return TemplatedFn(6, 4);
        break;

        case 5 :
        return TemplatedFn(6, 5);
        break;

        case 6 :
        return TemplatedFn(6, 6);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->6 for records with width 6 exceeds limit" );
    }
    break;

    case 7 :
    switch (depth) {
        case 2 :
        return TemplatedFn(7, 2);
        break;

        case 3 :
        return TemplatedFn(7, 3);
        break;

        case 4 :
        return TemplatedFn(7, 4);
        break;

        case 5 :
        return TemplatedFn(7, 5);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->5 for records with width 7 exceeds limit" );
    }
    break;

    case 8 :
    switch (depth) {
        case 2 :
        return TemplatedFn(8, 2);
        break;

        case 3 :
        return TemplatedFn(8, 3);
        break;

        case 4 :
        return TemplatedFn(8, 4);
        break;

        case 5 :
        return TemplatedFn(8, 5);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->5 for records with width 8 exceeds limit" );
    }
    break;

    case 9 :
    switch (depth) {
        case 2 :
        return TemplatedFn(9, 2);
        break;

        case 3 :
        return TemplatedFn(9, 3);
        break;

        case 4 :
        return TemplatedFn(9, 4);
        break;

        case 5 :
        return TemplatedFn(9, 5);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->5 for records with width 9 exceeds limit" );
    }
    break;

    case 10 :
    switch (depth) {
        case 2 :
        return TemplatedFn(10, 2);
        break;

        case 3 :
        return TemplatedFn(10, 3);
        break;

        case 4 :
        return TemplatedFn(10, 4);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->4 for records with width 10 exceeds limit" );
    }
    break;

    case 11 :
    switch (depth) {
        case 2 :
        return TemplatedFn(11, 2);
        break;

        case 3 :
        return TemplatedFn(11, 3);
        break;

        case 4 :
        return TemplatedFn(11, 4);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->4 for records with width 11 exceeds limit" );
    }
    break;

    case 12 :
    switch (depth) {
        case 2 :
        return TemplatedFn(12, 2);
        break;

        case 3 :
        return TemplatedFn(12, 3);
        break;

        case 4 :
        return TemplatedFn(12, 4);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->4 for records with width 12 exceeds limit" );
    }
    break;

    case 13 :
    switch (depth) {
        case 2 :
        return TemplatedFn(13, 2);
        break;

        case 3 :
        return TemplatedFn(13, 3);
        break;

        case 4 :
        return TemplatedFn(13, 4);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->4 for records with width 13 exceeds limit" );
    }
    break;

    case 14 :
    switch (depth) {
        case 2 :
        return TemplatedFn(14, 2);
        break;

        case 3 :
        return TemplatedFn(14, 3);
        break;

        case 4 :
        return TemplatedFn(14, 4);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->4 for records with width 14 exceeds limit" );
    }
    break;

    case 15 :
    switch (depth) {
        case 2 :
        return TemplatedFn(15, 2);
        break;

        case 3 :
        return TemplatedFn(15, 3);
        break;

        case 4 :
        return TemplatedFn(15, 4);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->4 for records with width 15 exceeds limit" );
    }
    break;

    case 16 :
    switch (depth) {
        case 2 :
        return TemplatedFn(16, 2);
        break;

        case 3 :
        return TemplatedFn(16, 3);
        break;

        case 4 :
        return TemplatedFn(16, 4);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->4 for records with width 16 exceeds limit" );
    }
    break;

    case 17 :
    switch (depth) {
        case 2 :
        return TemplatedFn(17, 2);
        break;

        case 3 :
        return TemplatedFn(17, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 17 exceeds limit" );
    }
    break;

    case 18 :
    switch (depth) {
        case 2 :
        return TemplatedFn(18, 2);
        break;

        case 3 :
        return TemplatedFn(18, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 18 exceeds limit" );
    }
    break;

    case 19 :
    switch (depth) {
        case 2 :
        return TemplatedFn(19, 2);
        break;

        case 3 :
        return TemplatedFn(19, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 19 exceeds limit" );
    }
    break;

    case 20 :
    switch (depth) {
        case 2 :
        return TemplatedFn(20, 2);
        break;

        case 3 :
        return TemplatedFn(20, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 20 exceeds limit" );
    }
    break;

    case 21 :
    switch (depth) {
        case 2 :
        return TemplatedFn(21, 2);
        break;

        case 3 :
        return TemplatedFn(21, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 21 exceeds limit" );
    }
    break;

    case 22 :
    switch (depth) {
        case 2 :
        return TemplatedFn(22, 2);
        break;

        case 3 :
        return TemplatedFn(22, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 22 exceeds limit" );
    }
    break;

    case 23 :
    switch (depth) {
        case 2 :
        return TemplatedFn(23, 2);
        break;

        case 3 :
        return TemplatedFn(23, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 23 exceeds limit" );
    }
    break;

    case 24 :
    switch (depth) {
        case 2 :
        return TemplatedFn(24, 2);
        break;

        case 3 :
        return TemplatedFn(24, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 24 exceeds limit" );
    }
    break;

    case 25 :
    switch (depth) {
        case 2 :
        return TemplatedFn(25, 2);
        break;

        case 3 :
        return TemplatedFn(25, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 25 exceeds limit" );
    }
    break;

    case 26 :
    switch (depth) {
        case 2 :
        return TemplatedFn(26, 2);
        break;

        case 3 :
        return TemplatedFn(26, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 26 exceeds limit" );
    }
    break;

    case 27 :
    switch (depth) {
        case 2 :
        return TemplatedFn(27, 2);
        break;

        case 3 :
        return TemplatedFn(27, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 27 exceeds limit" );
    }
    break;

    case 28 :
    switch (depth) {
        case 2 :
        return TemplatedFn(28, 2);
        break;

        case 3 :
        return TemplatedFn(28, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 28 exceeds limit" );
    }
    break;

    case 29 :
    switch (depth) {
        case 2 :
        return TemplatedFn(29, 2);
        break;

        case 3 :
        return TemplatedFn(29, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 29 exceeds limit" );
    }
    break;

    case 30 :
    switch (depth) {
        case 2 :
        return TemplatedFn(30, 2);
        break;

        case 3 :
        return TemplatedFn(30, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 30 exceeds limit" );
    }
    break;

    case 31 :
    switch (depth) {
        case 2 :
        return TemplatedFn(31, 2);
        break;

        case 3 :
        return TemplatedFn(31, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 31 exceeds limit" );
    }
    break;

    case 32 :
    switch (depth) {
        case 2 :
        return TemplatedFn(32, 2);
        break;

        case 3 :
        return TemplatedFn(32, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 32 exceeds limit" );
    }
    break;

    case 33 :
    switch (depth) {
        case 2 :
        return TemplatedFn(33, 2);
        break;

        case 3 :
        return TemplatedFn(33, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 33 exceeds limit" );
    }
    break;

    case 34 :
    switch (depth) {
        case 2 :
        return TemplatedFn(34, 2);
        break;

        case 3 :
        return TemplatedFn(34, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 34 exceeds limit" );
    }
    break;

    case 35 :
    switch (depth) {
        case 2 :
        return TemplatedFn(35, 2);
        break;

        case 3 :
        return TemplatedFn(35, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 35 exceeds limit" );
    }
    break;

    case 36 :
    switch (depth) {
        case 2 :
        return TemplatedFn(36, 2);
        break;

        case 3 :
        return TemplatedFn(36, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 36 exceeds limit" );
    }
    break;

    case 37 :
    switch (depth) {
        case 2 :
        return TemplatedFn(37, 2);
        break;

        case 3 :
        return TemplatedFn(37, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 37 exceeds limit" );
    }
    break;

    case 38 :
    switch (depth) {
        case 2 :
        return TemplatedFn(38, 2);
        break;

        case 3 :
        return TemplatedFn(38, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 38 exceeds limit" );
    }
    break;

    case 39 :
    switch (depth) {
        case 2 :
        return TemplatedFn(39, 2);
        break;

        case 3 :
        return TemplatedFn(39, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 39 exceeds limit" );
    }
    break;

    case 40 :
    switch (depth) {
        case 2 :
        return TemplatedFn(40, 2);
        break;

        case 3 :
        return TemplatedFn(40, 3);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 40 exceeds limit" );
    }
    break;

    default :
    throw std::runtime_error ( "Legitimate width 2 <-> 256 exceeded" );
}
