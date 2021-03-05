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

    default :
    throw std::runtime_error ( "Legitimate width 2 <-> 256 exceeded" );
}
