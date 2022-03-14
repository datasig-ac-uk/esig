switch (width) {
    case 2 :
    switch (depth) {
        case 2 :
        return TemplatedFn(2, 2);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->2 for records with width 2 exceeds limit" );
    }
    break;

    case 3 :
    switch (depth) {
        case 2 :
        return TemplatedFn(3, 2);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->2 for records with width 3 exceeds limit" );
    }
    break;

    case 4 :
    switch (depth) {
        case 2 :
        return TemplatedFn(4, 2);
        break;

        default :
        throw std::runtime_error ( "Legitimate depth of 2<->2 for records with width 4 exceeds limit" );
    }
    break;

    default :
    throw std::runtime_error ( "Legitimate width 2 <-> 256 exceeded" );
}
