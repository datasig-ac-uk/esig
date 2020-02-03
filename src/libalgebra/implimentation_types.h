
/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai, 
Greg Gyurkó and Arend Janssen. 

Distributed under the terms of the GNU General Public License, 
Version 3. (See accompanying file License.txt)

************************************************************* */


#include <stddef.h>

//  implimentation_types.h : provides definitions for basic types

#ifndef implimentation_types_h__
#define implimentation_types_h__

namespace alg {

	// moved from  libalgebra.h

	/// Used to store degrees. A value of 0 means no degree truncation.
	typedef unsigned DEG;
	/// Used to number letters. The value 0 is special.
	typedef unsigned long long LET;

}
#endif // implimetation_types_h__