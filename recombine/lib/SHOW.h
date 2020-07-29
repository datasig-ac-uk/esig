#ifndef SHOW_h__
#define SHOW_h__
// copyright terry lyons 2010 (updated 2013)
//
// a convenient macro for outputting when debugging
// SHOW(expression) outputs the value returned by the expression provided value_type is overloaded for output
// it is useful because it also gives the program name of the variable etc and returns a reference to the value
//
// ans = SHOW(2 * sin (3.)) + 5.
// prints "2 * 3. : 6 \n\n" and ans = 11.
// unless SHO_NO is defined before the header file SHOW is first included when ans = 11. 
// and no output will be produced
//
// define SHO_FILENAME to see line numbers and file name as well
// define SHO_IOS to the name of an ostream object to redirect output from cout
// eg. to std::err or a file
//
// define SHO_NO to suppress this output completely
//
#include <stdlib.h>
#include <iosfwd>
#include <iostream>
#include <string>
#pragma warning( push )
#pragma warning( disable : 4100 )
//namespace {

template <class T>
T SHO_(const char* message, T value, const char* message1, unsigned line,
	std::ostream& os = std::cout)
{
	os << message << " = " << value ;	// dont use an array so arrays can be overloaded
#ifdef SHO_FILENAME
#pragma warning( push )
#pragma warning( disable : 4996 )
		char drive[_MAX_DRIVE];
		char dir[_MAX_DIR];
		char fname[_MAX_FNAME];
		char ext[_MAX_EXT];

		_splitpath( message1, drive, dir, fname, ext ); // C4996
		// Note: _splitpath is deprecated; consider using _splitpath_s instead
		os << std::endl << "\t" << fname << ext << " Line:"<< line;
#pragma warning( pop )
#endif
	os << std::endl << std::endl;
	return value;
}
//}
#pragma warning( pop )
// Macro to give trace
#ifndef SHOW
#ifndef SHO_IOS
#define SHO_IOS std::cout
#endif
#ifndef SHO_NO
#define SHOW(arg) SHO_( #arg ,(arg), __FILE__ , __LINE__ , SHO_IOS)
#define SHOWWITHLABEL(label,arg) SHO_( (std::string(label)).c_str() ,(arg), __FILE__ , __LINE__ , SHO_IOS)
#else
#define SHOW(arg) (arg)
#define SHOWWITHLABEL(label,arg) (arg)
#endif
#endif
//#define SHOW(arg)

#endif // SHOW_h__
