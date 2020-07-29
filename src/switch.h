// MakeSwitchStatement.cpp : Defines the entry point for the console application.
////
//
//#include "stdafx.h"
//#include <iostream>
//#include <map>
//
///* enforced bounds and in all cases it will fit in a float
//	depth 	no of letters
//	16	2-2
//	10	3-3
//	8	4-4
//	6	5-6
//	5	7-9
//	4	10-16
//	3	17-40
//	2	41-256
//	*/
//	int _tmain(int argc, _TCHAR* argv[])
//{
//	std::map<unsigned int, unsigned short> range;
//	range[2] = 16;
//	range[3] = 10;
//	range[4] = 6;
//	range[6] = 6;
//	range[9] = 5;
//	range[16] = 4;
//	range[40] = 3;
//	range[256] = 2;
//	std::cout <<"switch (width) {\n";
//	unsigned int w = 2;
//	for (auto it = begin(range); it != end(range); ++it)
//	{
//		for ( ; w <= (it->first) ; ++w) 
//		{
//			std::cout <<"    case " << w << " :\n";
//			std::cout <<"    switch (depth) {\n";
//			for (int d = 2 ; d <= it->second ; ++d) {
//				std::cout << "        case " << d << " :\n";
//				std::cout << "        return TemplatedFn("<< w << "," << d << "); \n";
//				std::cout << "        break;\n\n";
//			}
//			std::cout << "        default: \n        std::cout << \"Legitimate depth of 2<->"<< it->second << " for records with width "<< w <<" exceeds limit\\n\";\n";
//			std::cout << "    }\n    break;\n\n";
//		}
//	}
//	std::cout <<"    default: \n    std::cout << \"Legitimate width 2 <-> 256 exceeded\\n\";\n";
//	std::cout << "}\n\n";
//	return 0;
//}
//

switch (width) {
#ifndef DUMMYSWITCHCONTENT
    case 2 :
    switch (depth) {
        case 2 :
        return TemplatedFn(2,2); 
        break;

        case 3 :
        return TemplatedFn(2,3); 
        break;

        case 4 :
        return TemplatedFn(2,4); 
        break;

        case 5 :
        return TemplatedFn(2,5); 
        break;

        case 6 :
        return TemplatedFn(2,6); 
        break;

        case 7 :
        return TemplatedFn(2,7); 
        break;

        case 8 :
        return TemplatedFn(2,8); 
        break;

        case 9 :
        return TemplatedFn(2,9); 
        break;

        case 10 :
        return TemplatedFn(2,10); 
        break;

        case 11 :
        return TemplatedFn(2,11); 
        break;

        case 12 :
        return TemplatedFn(2,12); 
        break;

        case 13 :
        return TemplatedFn(2,13); 
        break;

        case 14 :
        return TemplatedFn(2,14); 
        break;

        case 15 :
        return TemplatedFn(2,15); 
        break;

        case 16 :
        return TemplatedFn(2,16); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->16 for records with width 2 exceeds limit\n";
    }
    break;

    case 3 :
    switch (depth) {
        case 2 :
        return TemplatedFn(3,2); 
        break;

        case 3 :
        return TemplatedFn(3,3); 
        break;

        case 4 :
        return TemplatedFn(3,4); 
        break;

        case 5 :
        return TemplatedFn(3,5); 
        break;

        case 6 :
        return TemplatedFn(3,6); 
        break;

        case 7 :
        return TemplatedFn(3,7); 
        break;

        case 8 :
        return TemplatedFn(3,8); 
        break;

        case 9 :
        return TemplatedFn(3,9); 
        break;

        case 10 :
        return TemplatedFn(3,10); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->10 for records with width 3 exceeds limit\n";
    }
    break;

    case 4 :
    switch (depth) {
        case 2 :
        return TemplatedFn(4,2); 
        break;

        case 3 :
        return TemplatedFn(4,3); 
        break;

        case 4 :
        return TemplatedFn(4,4); 
        break;

        case 5 :
        return TemplatedFn(4,5); 
        break;

        case 6 :
        return TemplatedFn(4,6); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->6 for records with width 4 exceeds limit\n";
    }
    break;

    case 5 :
    switch (depth) {
        case 2 :
        return TemplatedFn(5,2); 
        break;

        case 3 :
        return TemplatedFn(5,3); 
        break;

        case 4 :
        return TemplatedFn(5,4); 
        break;

        case 5 :
        return TemplatedFn(5,5); 
        break;

        case 6 :
        return TemplatedFn(5,6); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->6 for records with width 5 exceeds limit\n";
    }
    break;

    case 6 :
    switch (depth) {
        case 2 :
        return TemplatedFn(6,2); 
        break;

        case 3 :
        return TemplatedFn(6,3); 
        break;

        case 4 :
        return TemplatedFn(6,4); 
        break;

        case 5 :
        return TemplatedFn(6,5); 
        break;

        case 6 :
        return TemplatedFn(6,6); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->6 for records with width 6 exceeds limit\n";
    }
    break;

    case 7 :
    switch (depth) {
        case 2 :
        return TemplatedFn(7,2); 
        break;

        case 3 :
        return TemplatedFn(7,3); 
        break;

        case 4 :
        return TemplatedFn(7,4); 
        break;

        case 5 :
        return TemplatedFn(7,5); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->5 for records with width 7 exceeds limit\n";
    }
    break;

    case 8 :
    switch (depth) {
        case 2 :
        return TemplatedFn(8,2); 
        break;

        case 3 :
        return TemplatedFn(8,3); 
        break;

        case 4 :
        return TemplatedFn(8,4); 
        break;

        case 5 :
        return TemplatedFn(8,5); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->5 for records with width 8 exceeds limit\n";
    }
    break;

    case 9 :
    switch (depth) {
        case 2 :
        return TemplatedFn(9,2); 
        break;

        case 3 :
        return TemplatedFn(9,3); 
        break;

        case 4 :
        return TemplatedFn(9,4); 
        break;

        case 5 :
        return TemplatedFn(9,5); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->5 for records with width 9 exceeds limit\n";
    }
    break;

    case 10 :
    switch (depth) {
        case 2 :
        return TemplatedFn(10,2); 
        break;

        case 3 :
        return TemplatedFn(10,3); 
        break;

        case 4 :
        return TemplatedFn(10,4); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->4 for records with width 10 exceeds limit\n";
    }
    break;

    case 11 :
    switch (depth) {
        case 2 :
        return TemplatedFn(11,2); 
        break;

        case 3 :
        return TemplatedFn(11,3); 
        break;

        case 4 :
        return TemplatedFn(11,4); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->4 for records with width 11 exceeds limit\n";
    }
    break;

    case 12 :
    switch (depth) {
        case 2 :
        return TemplatedFn(12,2); 
        break;

        case 3 :
        return TemplatedFn(12,3); 
        break;

        case 4 :
        return TemplatedFn(12,4); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->4 for records with width 12 exceeds limit\n";
    }
    break;

    case 13 :
    switch (depth) {
        case 2 :
        return TemplatedFn(13,2); 
        break;

        case 3 :
        return TemplatedFn(13,3); 
        break;

        case 4 :
        return TemplatedFn(13,4); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->4 for records with width 13 exceeds limit\n";
    }
    break;

    case 14 :
    switch (depth) {
        case 2 :
        return TemplatedFn(14,2); 
        break;

        case 3 :
        return TemplatedFn(14,3); 
        break;

        case 4 :
        return TemplatedFn(14,4); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->4 for records with width 14 exceeds limit\n";
    }
    break;

    case 15 :
    switch (depth) {
        case 2 :
        return TemplatedFn(15,2); 
        break;

        case 3 :
        return TemplatedFn(15,3); 
        break;

        case 4 :
        return TemplatedFn(15,4); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->4 for records with width 15 exceeds limit\n";
    }
    break;

    case 16 :
    switch (depth) {
        case 2 :
        return TemplatedFn(16,2); 
        break;

        case 3 :
        return TemplatedFn(16,3); 
        break;

        case 4 :
        return TemplatedFn(16,4); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->4 for records with width 16 exceeds limit\n";
    }
    break;

    case 17 :
    switch (depth) {
        case 2 :
        return TemplatedFn(17,2); 
        break;

        case 3 :
        return TemplatedFn(17,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 17 exceeds limit\n";
    }
    break;

    case 18 :
    switch (depth) {
        case 2 :
        return TemplatedFn(18,2); 
        break;

        case 3 :
        return TemplatedFn(18,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 18 exceeds limit\n";
    }
    break;

    case 19 :
    switch (depth) {
        case 2 :
        return TemplatedFn(19,2); 
        break;

        case 3 :
        return TemplatedFn(19,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 19 exceeds limit\n";
    }
    break;
#endif //DUMMYSWITCHCONTENT
    case 20 :
    switch (depth) {
        case 2 :
        return TemplatedFn(20,2); 
        break;

        case 3 :
        return TemplatedFn(20,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 20 exceeds limit\n";
    }
    break;

/*CommentOut
    case 21 :
    switch (depth) {
        case 2 :
        return TemplatedFn(21,2); 
        break;

        case 3 :
        return TemplatedFn(21,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 21 exceeds limit\n";
    }
    break;

    case 22 :
    switch (depth) {
        case 2 :
        return TemplatedFn(22,2); 
        break;

        case 3 :
        return TemplatedFn(22,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 22 exceeds limit\n";
    }
    break;

    case 23 :
    switch (depth) {
        case 2 :
        return TemplatedFn(23,2); 
        break;

        case 3 :
        return TemplatedFn(23,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 23 exceeds limit\n";
    }
    break;

    case 24 :
    switch (depth) {
        case 2 :
        return TemplatedFn(24,2); 
        break;

        case 3 :
        return TemplatedFn(24,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 24 exceeds limit\n";
    }
    break;

    case 25 :
    switch (depth) {
        case 2 :
        return TemplatedFn(25,2); 
        break;

        case 3 :
        return TemplatedFn(25,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 25 exceeds limit\n";
    }
    break;

    case 26 :
    switch (depth) {
        case 2 :
        return TemplatedFn(26,2); 
        break;

        case 3 :
        return TemplatedFn(26,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 26 exceeds limit\n";
    }
    break;

    case 27 :
    switch (depth) {
        case 2 :
        return TemplatedFn(27,2); 
        break;

        case 3 :
        return TemplatedFn(27,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 27 exceeds limit\n";
    }
    break;

    case 28 :
    switch (depth) {
        case 2 :
        return TemplatedFn(28,2); 
        break;

        case 3 :
        return TemplatedFn(28,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 28 exceeds limit\n";
    }
    break;

    case 29 :
    switch (depth) {
        case 2 :
        return TemplatedFn(29,2); 
        break;

        case 3 :
        return TemplatedFn(29,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 29 exceeds limit\n";
    }
    break;

    case 30 :
    switch (depth) {
        case 2 :
        return TemplatedFn(30,2); 
        break;

        case 3 :
        return TemplatedFn(30,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 30 exceeds limit\n";
    }
    break;

    case 31 :
    switch (depth) {
        case 2 :
        return TemplatedFn(31,2); 
        break;

        case 3 :
        return TemplatedFn(31,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 31 exceeds limit\n";
    }
    break;

    case 32 :
    switch (depth) {
        case 2 :
        return TemplatedFn(32,2); 
        break;

        case 3 :
        return TemplatedFn(32,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 32 exceeds limit\n";
    }
    break;

    case 33 :
    switch (depth) {
        case 2 :
        return TemplatedFn(33,2); 
        break;

        case 3 :
        return TemplatedFn(33,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 33 exceeds limit\n";
    }
    break;

    case 34 :
    switch (depth) {
        case 2 :
        return TemplatedFn(34,2); 
        break;

        case 3 :
        return TemplatedFn(34,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 34 exceeds limit\n";
    }
    break;

    case 35 :
    switch (depth) {
        case 2 :
        return TemplatedFn(35,2); 
        break;

        case 3 :
        return TemplatedFn(35,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 35 exceeds limit\n";
    }
    break;

    case 36 :
    switch (depth) {
        case 2 :
        return TemplatedFn(36,2); 
        break;

        case 3 :
        return TemplatedFn(36,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 36 exceeds limit\n";
    }
    break;

    case 37 :
    switch (depth) {
        case 2 :
        return TemplatedFn(37,2); 
        break;

        case 3 :
        return TemplatedFn(37,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 37 exceeds limit\n";
    }
    break;

    case 38 :
    switch (depth) {
        case 2 :
        return TemplatedFn(38,2); 
        break;

        case 3 :
        return TemplatedFn(38,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 38 exceeds limit\n";
    }
    break;

    case 39 :
    switch (depth) {
        case 2 :
        return TemplatedFn(39,2); 
        break;

        case 3 :
        return TemplatedFn(39,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 39 exceeds limit\n";
    }
    break;

    case 40 :
    switch (depth) {
        case 2 :
        return TemplatedFn(40,2); 
        break;

        case 3 :
        return TemplatedFn(40,3); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->3 for records with width 40 exceeds limit\n";
    }
    break;

    case 41 :
    switch (depth) {
        case 2 :
        return TemplatedFn(41,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 41 exceeds limit\n";
    }
    break;

    case 42 :
    switch (depth) {
        case 2 :
        return TemplatedFn(42,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 42 exceeds limit\n";
    }
    break;

    case 43 :
    switch (depth) {
        case 2 :
        return TemplatedFn(43,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 43 exceeds limit\n";
    }
    break;

    case 44 :
    switch (depth) {
        case 2 :
        return TemplatedFn(44,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 44 exceeds limit\n";
    }
    break;

    case 45 :
    switch (depth) {
        case 2 :
        return TemplatedFn(45,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 45 exceeds limit\n";
    }
    break;

    case 46 :
    switch (depth) {
        case 2 :
        return TemplatedFn(46,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 46 exceeds limit\n";
    }
    break;

    case 47 :
    switch (depth) {
        case 2 :
        return TemplatedFn(47,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 47 exceeds limit\n";
    }
    break;

    case 48 :
    switch (depth) {
        case 2 :
        return TemplatedFn(48,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 48 exceeds limit\n";
    }
    break;

    case 49 :
    switch (depth) {
        case 2 :
        return TemplatedFn(49,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 49 exceeds limit\n";
    }
    break;

    case 50 :
    switch (depth) {
        case 2 :
        return TemplatedFn(50,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 50 exceeds limit\n";
    }
    break;

    case 51 :
    switch (depth) {
        case 2 :
        return TemplatedFn(51,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 51 exceeds limit\n";
    }
    break;

    case 52 :
    switch (depth) {
        case 2 :
        return TemplatedFn(52,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 52 exceeds limit\n";
    }
    break;

    case 53 :
    switch (depth) {
        case 2 :
        return TemplatedFn(53,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 53 exceeds limit\n";
    }
    break;

    case 54 :
    switch (depth) {
        case 2 :
        return TemplatedFn(54,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 54 exceeds limit\n";
    }
    break;

    case 55 :
    switch (depth) {
        case 2 :
        return TemplatedFn(55,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 55 exceeds limit\n";
    }
    break;

    case 56 :
    switch (depth) {
        case 2 :
        return TemplatedFn(56,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 56 exceeds limit\n";
    }
    break;

    case 57 :
    switch (depth) {
        case 2 :
        return TemplatedFn(57,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 57 exceeds limit\n";
    }
    break;

    case 58 :
    switch (depth) {
        case 2 :
        return TemplatedFn(58,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 58 exceeds limit\n";
    }
    break;

    case 59 :
    switch (depth) {
        case 2 :
        return TemplatedFn(59,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 59 exceeds limit\n";
    }
    break;

    case 60 :
    switch (depth) {
        case 2 :
        return TemplatedFn(60,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 60 exceeds limit\n";
    }
    break;

    case 61 :
    switch (depth) {
        case 2 :
        return TemplatedFn(61,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 61 exceeds limit\n";
    }
    break;

    case 62 :
    switch (depth) {
        case 2 :
        return TemplatedFn(62,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 62 exceeds limit\n";
    }
    break;

    case 63 :
    switch (depth) {
        case 2 :
        return TemplatedFn(63,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 63 exceeds limit\n";
    }
    break;

    case 64 :
    switch (depth) {
        case 2 :
        return TemplatedFn(64,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 64 exceeds limit\n";
    }
    break;

    case 65 :
    switch (depth) {
        case 2 :
        return TemplatedFn(65,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 65 exceeds limit\n";
    }
    break;

    case 66 :
    switch (depth) {
        case 2 :
        return TemplatedFn(66,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 66 exceeds limit\n";
    }
    break;

    case 67 :
    switch (depth) {
        case 2 :
        return TemplatedFn(67,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 67 exceeds limit\n";
    }
    break;

    case 68 :
    switch (depth) {
        case 2 :
        return TemplatedFn(68,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 68 exceeds limit\n";
    }
    break;

    case 69 :
    switch (depth) {
        case 2 :
        return TemplatedFn(69,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 69 exceeds limit\n";
    }
    break;

    case 70 :
    switch (depth) {
        case 2 :
        return TemplatedFn(70,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 70 exceeds limit\n";
    }
    break;

    case 71 :
    switch (depth) {
        case 2 :
        return TemplatedFn(71,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 71 exceeds limit\n";
    }
    break;

    case 72 :
    switch (depth) {
        case 2 :
        return TemplatedFn(72,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 72 exceeds limit\n";
    }
    break;

    case 73 :
    switch (depth) {
        case 2 :
        return TemplatedFn(73,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 73 exceeds limit\n";
    }
    break;

    case 74 :
    switch (depth) {
        case 2 :
        return TemplatedFn(74,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 74 exceeds limit\n";
    }
    break;

    case 75 :
    switch (depth) {
        case 2 :
        return TemplatedFn(75,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 75 exceeds limit\n";
    }
    break;

    case 76 :
    switch (depth) {
        case 2 :
        return TemplatedFn(76,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 76 exceeds limit\n";
    }
    break;

    case 77 :
    switch (depth) {
        case 2 :
        return TemplatedFn(77,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 77 exceeds limit\n";
    }
    break;

    case 78 :
    switch (depth) {
        case 2 :
        return TemplatedFn(78,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 78 exceeds limit\n";
    }
    break;

    case 79 :
    switch (depth) {
        case 2 :
        return TemplatedFn(79,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 79 exceeds limit\n";
    }
    break;

    case 80 :
    switch (depth) {
        case 2 :
        return TemplatedFn(80,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 80 exceeds limit\n";
    }
    break;

    case 81 :
    switch (depth) {
        case 2 :
        return TemplatedFn(81,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 81 exceeds limit\n";
    }
    break;

    case 82 :
    switch (depth) {
        case 2 :
        return TemplatedFn(82,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 82 exceeds limit\n";
    }
    break;

    case 83 :
    switch (depth) {
        case 2 :
        return TemplatedFn(83,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 83 exceeds limit\n";
    }
    break;

    case 84 :
    switch (depth) {
        case 2 :
        return TemplatedFn(84,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 84 exceeds limit\n";
    }
    break;

    case 85 :
    switch (depth) {
        case 2 :
        return TemplatedFn(85,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 85 exceeds limit\n";
    }
    break;

    case 86 :
    switch (depth) {
        case 2 :
        return TemplatedFn(86,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 86 exceeds limit\n";
    }
    break;

    case 87 :
    switch (depth) {
        case 2 :
        return TemplatedFn(87,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 87 exceeds limit\n";
    }
    break;

    case 88 :
    switch (depth) {
        case 2 :
        return TemplatedFn(88,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 88 exceeds limit\n";
    }
    break;

    case 89 :
    switch (depth) {
        case 2 :
        return TemplatedFn(89,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 89 exceeds limit\n";
    }
    break;

    case 90 :
    switch (depth) {
        case 2 :
        return TemplatedFn(90,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 90 exceeds limit\n";
    }
    break;

    case 91 :
    switch (depth) {
        case 2 :
        return TemplatedFn(91,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 91 exceeds limit\n";
    }
    break;

    case 92 :
    switch (depth) {
        case 2 :
        return TemplatedFn(92,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 92 exceeds limit\n";
    }
    break;

    case 93 :
    switch (depth) {
        case 2 :
        return TemplatedFn(93,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 93 exceeds limit\n";
    }
    break;

    case 94 :
    switch (depth) {
        case 2 :
        return TemplatedFn(94,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 94 exceeds limit\n";
    }
    break;

    case 95 :
    switch (depth) {
        case 2 :
        return TemplatedFn(95,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 95 exceeds limit\n";
    }
    break;

    case 96 :
    switch (depth) {
        case 2 :
        return TemplatedFn(96,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 96 exceeds limit\n";
    }
    break;

    case 97 :
    switch (depth) {
        case 2 :
        return TemplatedFn(97,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 97 exceeds limit\n";
    }
    break;

    case 98 :
    switch (depth) {
        case 2 :
        return TemplatedFn(98,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 98 exceeds limit\n";
    }
    break;

    case 99 :
    switch (depth) {
        case 2 :
        return TemplatedFn(99,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 99 exceeds limit\n";
    }
    break;

    case 100 :
    switch (depth) {
        case 2 :
        return TemplatedFn(100,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 100 exceeds limit\n";
    }
    break;

    case 101 :
    switch (depth) {
        case 2 :
        return TemplatedFn(101,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 101 exceeds limit\n";
    }
    break;

    case 102 :
    switch (depth) {
        case 2 :
        return TemplatedFn(102,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 102 exceeds limit\n";
    }
    break;

    case 103 :
    switch (depth) {
        case 2 :
        return TemplatedFn(103,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 103 exceeds limit\n";
    }
    break;

    case 104 :
    switch (depth) {
        case 2 :
        return TemplatedFn(104,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 104 exceeds limit\n";
    }
    break;

    case 105 :
    switch (depth) {
        case 2 :
        return TemplatedFn(105,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 105 exceeds limit\n";
    }
    break;

    case 106 :
    switch (depth) {
        case 2 :
        return TemplatedFn(106,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 106 exceeds limit\n";
    }
    break;

    case 107 :
    switch (depth) {
        case 2 :
        return TemplatedFn(107,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 107 exceeds limit\n";
    }
    break;

    case 108 :
    switch (depth) {
        case 2 :
        return TemplatedFn(108,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 108 exceeds limit\n";
    }
    break;

    case 109 :
    switch (depth) {
        case 2 :
        return TemplatedFn(109,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 109 exceeds limit\n";
    }
    break;

    case 110 :
    switch (depth) {
        case 2 :
        return TemplatedFn(110,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 110 exceeds limit\n";
    }
    break;

    case 111 :
    switch (depth) {
        case 2 :
        return TemplatedFn(111,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 111 exceeds limit\n";
    }
    break;

    case 112 :
    switch (depth) {
        case 2 :
        return TemplatedFn(112,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 112 exceeds limit\n";
    }
    break;

    case 113 :
    switch (depth) {
        case 2 :
        return TemplatedFn(113,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 113 exceeds limit\n";
    }
    break;

    case 114 :
    switch (depth) {
        case 2 :
        return TemplatedFn(114,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 114 exceeds limit\n";
    }
    break;

    case 115 :
    switch (depth) {
        case 2 :
        return TemplatedFn(115,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 115 exceeds limit\n";
    }
    break;

    case 116 :
    switch (depth) {
        case 2 :
        return TemplatedFn(116,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 116 exceeds limit\n";
    }
    break;

    case 117 :
    switch (depth) {
        case 2 :
        return TemplatedFn(117,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 117 exceeds limit\n";
    }
    break;

    case 118 :
    switch (depth) {
        case 2 :
        return TemplatedFn(118,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 118 exceeds limit\n";
    }
    break;

    case 119 :
    switch (depth) {
        case 2 :
        return TemplatedFn(119,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 119 exceeds limit\n";
    }
    break;

    case 120 :
    switch (depth) {
        case 2 :
        return TemplatedFn(120,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 120 exceeds limit\n";
    }
    break;

    case 121 :
    switch (depth) {
        case 2 :
        return TemplatedFn(121,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 121 exceeds limit\n";
    }
    break;

    case 122 :
    switch (depth) {
        case 2 :
        return TemplatedFn(122,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 122 exceeds limit\n";
    }
    break;

    case 123 :
    switch (depth) {
        case 2 :
        return TemplatedFn(123,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 123 exceeds limit\n";
    }
    break;

    case 124 :
    switch (depth) {
        case 2 :
        return TemplatedFn(124,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 124 exceeds limit\n";
    }
    break;

    case 125 :
    switch (depth) {
        case 2 :
        return TemplatedFn(125,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 125 exceeds limit\n";
    }
    break;

    case 126 :
    switch (depth) {
        case 2 :
        return TemplatedFn(126,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 126 exceeds limit\n";
    }
    break;

    case 127 :
    switch (depth) {
        case 2 :
        return TemplatedFn(127,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 127 exceeds limit\n";
    }
    break;

    case 128 :
    switch (depth) {
        case 2 :
        return TemplatedFn(128,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 128 exceeds limit\n";
    }
    break;

    case 129 :
    switch (depth) {
        case 2 :
        return TemplatedFn(129,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 129 exceeds limit\n";
    }
    break;

    case 130 :
    switch (depth) {
        case 2 :
        return TemplatedFn(130,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 130 exceeds limit\n";
    }
    break;

    case 131 :
    switch (depth) {
        case 2 :
        return TemplatedFn(131,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 131 exceeds limit\n";
    }
    break;

    case 132 :
    switch (depth) {
        case 2 :
        return TemplatedFn(132,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 132 exceeds limit\n";
    }
    break;

    case 133 :
    switch (depth) {
        case 2 :
        return TemplatedFn(133,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 133 exceeds limit\n";
    }
    break;

    case 134 :
    switch (depth) {
        case 2 :
        return TemplatedFn(134,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 134 exceeds limit\n";
    }
    break;

    case 135 :
    switch (depth) {
        case 2 :
        return TemplatedFn(135,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 135 exceeds limit\n";
    }
    break;

    case 136 :
    switch (depth) {
        case 2 :
        return TemplatedFn(136,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 136 exceeds limit\n";
    }
    break;

    case 137 :
    switch (depth) {
        case 2 :
        return TemplatedFn(137,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 137 exceeds limit\n";
    }
    break;

    case 138 :
    switch (depth) {
        case 2 :
        return TemplatedFn(138,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 138 exceeds limit\n";
    }
    break;

    case 139 :
    switch (depth) {
        case 2 :
        return TemplatedFn(139,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 139 exceeds limit\n";
    }
    break;

    case 140 :
    switch (depth) {
        case 2 :
        return TemplatedFn(140,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 140 exceeds limit\n";
    }
    break;

    case 141 :
    switch (depth) {
        case 2 :
        return TemplatedFn(141,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 141 exceeds limit\n";
    }
    break;

    case 142 :
    switch (depth) {
        case 2 :
        return TemplatedFn(142,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 142 exceeds limit\n";
    }
    break;

    case 143 :
    switch (depth) {
        case 2 :
        return TemplatedFn(143,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 143 exceeds limit\n";
    }
    break;

    case 144 :
    switch (depth) {
        case 2 :
        return TemplatedFn(144,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 144 exceeds limit\n";
    }
    break;

    case 145 :
    switch (depth) {
        case 2 :
        return TemplatedFn(145,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 145 exceeds limit\n";
    }
    break;

    case 146 :
    switch (depth) {
        case 2 :
        return TemplatedFn(146,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 146 exceeds limit\n";
    }
    break;

    case 147 :
    switch (depth) {
        case 2 :
        return TemplatedFn(147,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 147 exceeds limit\n";
    }
    break;

    case 148 :
    switch (depth) {
        case 2 :
        return TemplatedFn(148,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 148 exceeds limit\n";
    }
    break;

    case 149 :
    switch (depth) {
        case 2 :
        return TemplatedFn(149,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 149 exceeds limit\n";
    }
    break;

    case 150 :
    switch (depth) {
        case 2 :
        return TemplatedFn(150,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 150 exceeds limit\n";
    }
    break;

    case 151 :
    switch (depth) {
        case 2 :
        return TemplatedFn(151,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 151 exceeds limit\n";
    }
    break;

    case 152 :
    switch (depth) {
        case 2 :
        return TemplatedFn(152,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 152 exceeds limit\n";
    }
    break;

    case 153 :
    switch (depth) {
        case 2 :
        return TemplatedFn(153,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 153 exceeds limit\n";
    }
    break;

    case 154 :
    switch (depth) {
        case 2 :
        return TemplatedFn(154,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 154 exceeds limit\n";
    }
    break;

    case 155 :
    switch (depth) {
        case 2 :
        return TemplatedFn(155,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 155 exceeds limit\n";
    }
    break;

    case 156 :
    switch (depth) {
        case 2 :
        return TemplatedFn(156,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 156 exceeds limit\n";
    }
    break;

    case 157 :
    switch (depth) {
        case 2 :
        return TemplatedFn(157,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 157 exceeds limit\n";
    }
    break;

    case 158 :
    switch (depth) {
        case 2 :
        return TemplatedFn(158,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 158 exceeds limit\n";
    }
    break;

    case 159 :
    switch (depth) {
        case 2 :
        return TemplatedFn(159,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 159 exceeds limit\n";
    }
    break;

    case 160 :
    switch (depth) {
        case 2 :
        return TemplatedFn(160,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 160 exceeds limit\n";
    }
    break;

    case 161 :
    switch (depth) {
        case 2 :
        return TemplatedFn(161,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 161 exceeds limit\n";
    }
    break;

    case 162 :
    switch (depth) {
        case 2 :
        return TemplatedFn(162,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 162 exceeds limit\n";
    }
    break;

    case 163 :
    switch (depth) {
        case 2 :
        return TemplatedFn(163,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 163 exceeds limit\n";
    }
    break;

    case 164 :
    switch (depth) {
        case 2 :
        return TemplatedFn(164,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 164 exceeds limit\n";
    }
    break;

    case 165 :
    switch (depth) {
        case 2 :
        return TemplatedFn(165,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 165 exceeds limit\n";
    }
    break;

    case 166 :
    switch (depth) {
        case 2 :
        return TemplatedFn(166,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 166 exceeds limit\n";
    }
    break;

    case 167 :
    switch (depth) {
        case 2 :
        return TemplatedFn(167,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 167 exceeds limit\n";
    }
    break;

    case 168 :
    switch (depth) {
        case 2 :
        return TemplatedFn(168,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 168 exceeds limit\n";
    }
    break;

    case 169 :
    switch (depth) {
        case 2 :
        return TemplatedFn(169,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 169 exceeds limit\n";
    }
    break;

    case 170 :
    switch (depth) {
        case 2 :
        return TemplatedFn(170,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 170 exceeds limit\n";
    }
    break;

    case 171 :
    switch (depth) {
        case 2 :
        return TemplatedFn(171,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 171 exceeds limit\n";
    }
    break;

    case 172 :
    switch (depth) {
        case 2 :
        return TemplatedFn(172,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 172 exceeds limit\n";
    }
    break;

    case 173 :
    switch (depth) {
        case 2 :
        return TemplatedFn(173,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 173 exceeds limit\n";
    }
    break;

    case 174 :
    switch (depth) {
        case 2 :
        return TemplatedFn(174,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 174 exceeds limit\n";
    }
    break;

    case 175 :
    switch (depth) {
        case 2 :
        return TemplatedFn(175,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 175 exceeds limit\n";
    }
    break;

    case 176 :
    switch (depth) {
        case 2 :
        return TemplatedFn(176,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 176 exceeds limit\n";
    }
    break;

    case 177 :
    switch (depth) {
        case 2 :
        return TemplatedFn(177,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 177 exceeds limit\n";
    }
    break;

    case 178 :
    switch (depth) {
        case 2 :
        return TemplatedFn(178,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 178 exceeds limit\n";
    }
    break;

    case 179 :
    switch (depth) {
        case 2 :
        return TemplatedFn(179,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 179 exceeds limit\n";
    }
    break;

    case 180 :
    switch (depth) {
        case 2 :
        return TemplatedFn(180,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 180 exceeds limit\n";
    }
    break;

    case 181 :
    switch (depth) {
        case 2 :
        return TemplatedFn(181,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 181 exceeds limit\n";
    }
    break;

    case 182 :
    switch (depth) {
        case 2 :
        return TemplatedFn(182,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 182 exceeds limit\n";
    }
    break;

    case 183 :
    switch (depth) {
        case 2 :
        return TemplatedFn(183,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 183 exceeds limit\n";
    }
    break;

    case 184 :
    switch (depth) {
        case 2 :
        return TemplatedFn(184,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 184 exceeds limit\n";
    }
    break;

    case 185 :
    switch (depth) {
        case 2 :
        return TemplatedFn(185,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 185 exceeds limit\n";
    }
    break;

    case 186 :
    switch (depth) {
        case 2 :
        return TemplatedFn(186,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 186 exceeds limit\n";
    }
    break;

    case 187 :
    switch (depth) {
        case 2 :
        return TemplatedFn(187,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 187 exceeds limit\n";
    }
    break;

    case 188 :
    switch (depth) {
        case 2 :
        return TemplatedFn(188,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 188 exceeds limit\n";
    }
    break;

    case 189 :
    switch (depth) {
        case 2 :
        return TemplatedFn(189,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 189 exceeds limit\n";
    }
    break;

    case 190 :
    switch (depth) {
        case 2 :
        return TemplatedFn(190,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 190 exceeds limit\n";
    }
    break;

    case 191 :
    switch (depth) {
        case 2 :
        return TemplatedFn(191,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 191 exceeds limit\n";
    }
    break;

    case 192 :
    switch (depth) {
        case 2 :
        return TemplatedFn(192,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 192 exceeds limit\n";
    }
    break;

    case 193 :
    switch (depth) {
        case 2 :
        return TemplatedFn(193,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 193 exceeds limit\n";
    }
    break;

    case 194 :
    switch (depth) {
        case 2 :
        return TemplatedFn(194,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 194 exceeds limit\n";
    }
    break;

    case 195 :
    switch (depth) {
        case 2 :
        return TemplatedFn(195,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 195 exceeds limit\n";
    }
    break;

    case 196 :
    switch (depth) {
        case 2 :
        return TemplatedFn(196,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 196 exceeds limit\n";
    }
    break;

    case 197 :
    switch (depth) {
        case 2 :
        return TemplatedFn(197,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 197 exceeds limit\n";
    }
    break;

    case 198 :
    switch (depth) {
        case 2 :
        return TemplatedFn(198,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 198 exceeds limit\n";
    }
    break;

    case 199 :
    switch (depth) {
        case 2 :
        return TemplatedFn(199,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 199 exceeds limit\n";
    }
    break;

    case 200 :
    switch (depth) {
        case 2 :
        return TemplatedFn(200,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 200 exceeds limit\n";
    }
    break;

    case 201 :
    switch (depth) {
        case 2 :
        return TemplatedFn(201,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 201 exceeds limit\n";
    }
    break;

    case 202 :
    switch (depth) {
        case 2 :
        return TemplatedFn(202,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 202 exceeds limit\n";
    }
    break;

    case 203 :
    switch (depth) {
        case 2 :
        return TemplatedFn(203,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 203 exceeds limit\n";
    }
    break;

    case 204 :
    switch (depth) {
        case 2 :
        return TemplatedFn(204,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 204 exceeds limit\n";
    }
    break;

    case 205 :
    switch (depth) {
        case 2 :
        return TemplatedFn(205,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 205 exceeds limit\n";
    }
    break;

    case 206 :
    switch (depth) {
        case 2 :
        return TemplatedFn(206,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 206 exceeds limit\n";
    }
    break;

    case 207 :
    switch (depth) {
        case 2 :
        return TemplatedFn(207,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 207 exceeds limit\n";
    }
    break;

    case 208 :
    switch (depth) {
        case 2 :
        return TemplatedFn(208,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 208 exceeds limit\n";
    }
    break;

    case 209 :
    switch (depth) {
        case 2 :
        return TemplatedFn(209,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 209 exceeds limit\n";
    }
    break;

    case 210 :
    switch (depth) {
        case 2 :
        return TemplatedFn(210,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 210 exceeds limit\n";
    }
    break;

    case 211 :
    switch (depth) {
        case 2 :
        return TemplatedFn(211,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 211 exceeds limit\n";
    }
    break;

    case 212 :
    switch (depth) {
        case 2 :
        return TemplatedFn(212,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 212 exceeds limit\n";
    }
    break;

    case 213 :
    switch (depth) {
        case 2 :
        return TemplatedFn(213,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 213 exceeds limit\n";
    }
    break;

    case 214 :
    switch (depth) {
        case 2 :
        return TemplatedFn(214,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 214 exceeds limit\n";
    }
    break;

    case 215 :
    switch (depth) {
        case 2 :
        return TemplatedFn(215,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 215 exceeds limit\n";
    }
    break;

    case 216 :
    switch (depth) {
        case 2 :
        return TemplatedFn(216,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 216 exceeds limit\n";
    }
    break;

    case 217 :
    switch (depth) {
        case 2 :
        return TemplatedFn(217,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 217 exceeds limit\n";
    }
    break;

    case 218 :
    switch (depth) {
        case 2 :
        return TemplatedFn(218,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 218 exceeds limit\n";
    }
    break;

    case 219 :
    switch (depth) {
        case 2 :
        return TemplatedFn(219,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 219 exceeds limit\n";
    }
    break;

    case 220 :
    switch (depth) {
        case 2 :
        return TemplatedFn(220,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 220 exceeds limit\n";
    }
    break;

    case 221 :
    switch (depth) {
        case 2 :
        return TemplatedFn(221,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 221 exceeds limit\n";
    }
    break;

    case 222 :
    switch (depth) {
        case 2 :
        return TemplatedFn(222,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 222 exceeds limit\n";
    }
    break;

    case 223 :
    switch (depth) {
        case 2 :
        return TemplatedFn(223,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 223 exceeds limit\n";
    }
    break;

    case 224 :
    switch (depth) {
        case 2 :
        return TemplatedFn(224,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 224 exceeds limit\n";
    }
    break;

    case 225 :
    switch (depth) {
        case 2 :
        return TemplatedFn(225,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 225 exceeds limit\n";
    }
    break;

    case 226 :
    switch (depth) {
        case 2 :
        return TemplatedFn(226,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 226 exceeds limit\n";
    }
    break;

    case 227 :
    switch (depth) {
        case 2 :
        return TemplatedFn(227,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 227 exceeds limit\n";
    }
    break;

    case 228 :
    switch (depth) {
        case 2 :
        return TemplatedFn(228,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 228 exceeds limit\n";
    }
    break;

    case 229 :
    switch (depth) {
        case 2 :
        return TemplatedFn(229,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 229 exceeds limit\n";
    }
    break;

    case 230 :
    switch (depth) {
        case 2 :
        return TemplatedFn(230,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 230 exceeds limit\n";
    }
    break;

    case 231 :
    switch (depth) {
        case 2 :
        return TemplatedFn(231,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 231 exceeds limit\n";
    }
    break;

    case 232 :
    switch (depth) {
        case 2 :
        return TemplatedFn(232,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 232 exceeds limit\n";
    }
    break;

    case 233 :
    switch (depth) {
        case 2 :
        return TemplatedFn(233,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 233 exceeds limit\n";
    }
    break;

    case 234 :
    switch (depth) {
        case 2 :
        return TemplatedFn(234,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 234 exceeds limit\n";
    }
    break;

    case 235 :
    switch (depth) {
        case 2 :
        return TemplatedFn(235,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 235 exceeds limit\n";
    }
    break;

    case 236 :
    switch (depth) {
        case 2 :
        return TemplatedFn(236,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 236 exceeds limit\n";
    }
    break;

    case 237 :
    switch (depth) {
        case 2 :
        return TemplatedFn(237,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 237 exceeds limit\n";
    }
    break;

    case 238 :
    switch (depth) {
        case 2 :
        return TemplatedFn(238,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 238 exceeds limit\n";
    }
    break;

    case 239 :
    switch (depth) {
        case 2 :
        return TemplatedFn(239,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 239 exceeds limit\n";
    }
    break;

    case 240 :
    switch (depth) {
        case 2 :
        return TemplatedFn(240,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 240 exceeds limit\n";
    }
    break;

    case 241 :
    switch (depth) {
        case 2 :
        return TemplatedFn(241,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 241 exceeds limit\n";
    }
    break;

    case 242 :
    switch (depth) {
        case 2 :
        return TemplatedFn(242,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 242 exceeds limit\n";
    }
    break;

    case 243 :
    switch (depth) {
        case 2 :
        return TemplatedFn(243,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 243 exceeds limit\n";
    }
    break;

    case 244 :
    switch (depth) {
        case 2 :
        return TemplatedFn(244,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 244 exceeds limit\n";
    }
    break;

    case 245 :
    switch (depth) {
        case 2 :
        return TemplatedFn(245,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 245 exceeds limit\n";
    }
    break;

    case 246 :
    switch (depth) {
        case 2 :
        return TemplatedFn(246,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 246 exceeds limit\n";
    }
    break;

    case 247 :
    switch (depth) {
        case 2 :
        return TemplatedFn(247,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 247 exceeds limit\n";
    }
    break;

    case 248 :
    switch (depth) {
        case 2 :
        return TemplatedFn(248,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 248 exceeds limit\n";
    }
    break;

    case 249 :
    switch (depth) {
        case 2 :
        return TemplatedFn(249,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 249 exceeds limit\n";
    }
    break;

    case 250 :
    switch (depth) {
        case 2 :
        return TemplatedFn(250,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 250 exceeds limit\n";
    }
    break;

    case 251 :
    switch (depth) {
        case 2 :
        return TemplatedFn(251,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 251 exceeds limit\n";
    }
    break;

    case 252 :
    switch (depth) {
        case 2 :
        return TemplatedFn(252,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 252 exceeds limit\n";
    }
    break;

    case 253 :
    switch (depth) {
        case 2 :
        return TemplatedFn(253,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 253 exceeds limit\n";
    }
    break;

    case 254 :
    switch (depth) {
        case 2 :
        return TemplatedFn(254,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 254 exceeds limit\n";
    }
    break;

    case 255 :
    switch (depth) {
        case 2 :
        return TemplatedFn(255,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 255 exceeds limit\n";
    }
    break;

    case 256 :
    switch (depth) {
        case 2 :
        return TemplatedFn(256,2); 
        break;

        default: 
        std::cout << "Legitimate depth of 2<->2 for records with width 256 exceeds limit\n";
    }
    break;
CommentOut*/
    default: 
#ifdef DUMMYSWITCHCONTENT
		std::cout << "TEST VERSION - WIDTH 20 DEPTH 2 ONLY\n";
#else
		std::cout << "Legitimate width 2 <-> 20 exceeded\n";
#endif //DUMMYSWITCHCONTENT
}

