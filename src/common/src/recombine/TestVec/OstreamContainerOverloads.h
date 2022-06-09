
#ifndef OstreamContainerOverloads_h__
#define OstreamContainerOverloads_h__
#include <iostream>
#include <vector>
#include <iterator>
#include <valarray>
#include <array>


	// output a vector if one can output the elements
	template <class T> 
	inline	std::ostream & operator << (std::ostream & out, const std::vector < T > & arg)
	{
		out << '{';
		for (typename std::vector < T > ::const_iterator it(arg.begin()); it != arg.end(); ++it)
		{
			out << *it;
			if (it + 1 != arg.end())
				out << ", ";
		}		out << '}';
		return out;
	}

	// output a valarray if one can output the elements
	template <class T> 
	inline	std::ostream & operator << (std::ostream & out, const std::valarray < T > & arg)
	{
		out << '{';

		std::copy(arg.begin(), arg.end(), std::ostream_iterator < T > (out, " "));
		out << '}';
		return out;
	}

	template <class T, size_t N>
	inline std::ostream& operator << (std::ostream& os, const std::array < T, N > & in)
	{
		os << "{";
		for (typename std::array < T, N >::const_iterator it(in.begin()); it != in.end(); ++it)
		{
			os << *it;
			if (it + 1 != in.end())
				os << ", ";
		}
		os << "}";
		return os;
	}

#endif // OstreamContainerOverloads_h__
