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

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->6 for records with width 2 exceeds limit" );
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

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->4 for records with width 3 exceeds limit" );
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

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->4 for records with width 4 exceeds limit" );
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

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 5 exceeds limit" );
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

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->3 for records with width 6 exceeds limit" );
    }
    break;

    default :
    throw std::runtime_error ( "Legitimate width 2 <-> 256 exceeded" );
}
