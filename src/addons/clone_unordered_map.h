#pragma once
// derived from http://en.cppreference.com/w/cpp/container/unordered_map
#include <unordered_map> 

template <typename... T> class
CLONE_UNORDERED_MAP : std::unordered_map<T...>
{
	typedef CLONE_UNORDERED_MAP<T...> my_type;
	typedef std::unordered_map<T...> my_base;
public:
	using my_base::key_type;
	using my_base::mapped_type;
	using my_base::value_type;
	using my_base::size_type;
	using my_base::difference_type;
	using my_base::hasher;
	using my_base::key_equal;
	using my_base::allocator_type;
	using my_base::/*std::allocator_traits<allocator_type>::*/pointer;
	using my_base::/*std::allocator_traits<allocator_type>::*/const_pointer;
	using my_base::iterator;
	using my_base::const_iterator;
	using my_base::local_iterator;
	using my_base::const_local_iterator;
	//	using my_base::node_type;
	//  using my_base::insert_return_type;
	using  my_base::value_compare;

	/// constructors
	CLONE_UNORDERED_MAP(const CLONE_UNORDERED_MAP& v) : my_base((const my_base&) v) {}
	CLONE_UNORDERED_MAP(CLONE_UNORDERED_MAP&& v) : my_base((my_base&&) v) {}

	template<class ...T>
	CLONE_UNORDERED_MAP(T... args) : my_base(args...) {}
	//	  constructs the unordered_map
	//	  (public member function)

	~CLONE_UNORDERED_MAP() {}
	//destructs the unordered_map
	//(public member function)

	my_type& operator=(const my_type& rhs)
	{
		my_base::operator=(rhs);
		return (*this);
	}

	my_type& operator=(my_type&& rhs)
	{
		my_base::operator=(std::move(rhs));
		return (*this);
	}

	my_type& operator=(std::initializer_list<value_type> init_list)
	{
		clear();
		insert(init_list);
		return (*this);
	}

	//assigns values to the container
	//(public member function)

	using  my_base::get_allocator;
	//returns the associated allocator
	//(public member function)

///		  Capacity

	using  my_base::empty;
	//checks whether the container is empty
	//(public member function)

	using  my_base::size;
	//returns the number of elements
	//(public member function)

	using  my_base::max_size;
	//returns the maximum possible number of elements
	//(public member function)

///		  Modifiers

	using  my_base::clear;
	//clears the contents
	//(public member function)

	using  my_base::insert;
	//inserts elements or nodes(since C++17)
	//(public member function)

	using  my_base::insert_or_assign;
	//(C++17)
	//inserts an element or assigns to the current element if the key already exists
	//(public member function)

	using  my_base::emplace;
	//(C++11)
	//constructs element in - place
	//(public member function)

	using  my_base::emplace_hint;
	//(C++11)
	//constructs elements in - place using a hint
	//(public member function)

	using  my_base::try_emplace;
	//(C++17)
	//inserts in - place if the key does not exist, does nothing if the key exists
	//(public member function)

	using  my_base::erase;
	//erases elements
	//(public member function)

	void swap(my_type& arg)
	{
		my_base::swap((my_base&) arg);//template constructors produce wrong types
	}
	//using  my_base::swap;
	//swaps the contents
	//(public member function)

//	using  my_base::extract;
	//(C++17)
	//extracts nodes from the container
	//(public member function)

//	using  my_base::merge;
	//(C++17)
	//splices nodes from another container
	//(public member function)

///		  Lookup

	using  my_base::at;
	//(C++11)
	//access specified element with bounds checking
	//(public member function)

	using  my_base::operator[];
	//access specified element
	//(public member function)

	using  my_base::count;
	//returns the number of elements matching specific key
	//(public member function)

	using  my_base::find;
	//finds element with specific key
	//(public member function)

	using  my_base::equal_range;
	//returns range of elements matching a specific key
	//(public member function)

///      Bucket Interface

	using  my_base::begin;
	using  my_base::cbegin;
	//returns an iterator to the beginning of the specified bucket
	//(public member function)

	using  my_base::end;
	using  my_base::cend;
	//returns an iterator to the end of the specified bucket
	//(public member function)

	using  my_base::bucket_count;
	//returns the number of buckets
	//(public member function)

	using  my_base::max_bucket_count;
	//returns the maximum number of buckets
	//(public member function)

	using  my_base::bucket_size;
	//returns the number of elements in specific bucket
	//(public member function)

	using  my_base::bucket;
	//returns the bucket for specific key
	//(public member function)

///		Hash policy

	using  my_base::load_factor;
	//returns average number of elements per bucket
	//(public member function)

	using  my_base::max_load_factor;
	//manages maximum average number of elements per bucket
	//(public member function)

	using  my_base::rehash;
	//reserves at least the specified number of buckets.
	//	This regenerates the hash table.
	//	(public member function)

	using  my_base::reserve;
	//reserves space for at least the specified number of elements.
	//This regenerates the hash table.
	//(public member function)

///		  Observers

	using  my_base::hash_function;
	//returns function used to hash the keys 
	//(public member function)

	using  my_base::key_eq;
	//returns the function that compares keys
	//(public member function)

	///		  Non - member functions

#define BOOLOP(xopx) \
	friend 	bool operator xopx (const CLONE_UNORDERED_MAP<T...>& lhs,\
                                const CLONE_UNORDERED_MAP<T...>& rhs) \
	{return CLONE_UNORDERED_MAP< T...>::my_base::operator xopx (static_cast<const CLONE_UNORDERED_MAP<T...>::my_base&>(lhs),\
                           static_cast<const CLONE_UNORDERED_MAP<T...>::my_base&>(rhs));\
    } \

	BOOLOP(== );
	BOOLOP(!= );
	/*
	BOOLOP(< );
	BOOLOP(<= );
	BOOLOP(> );
	BOOLOP(>= );
	*/
#undef BOOLOP

	friend void swap(
		const CLONE_UNORDERED_MAP< T...>& lhs,
		const CLONE_UNORDERED_MAP< T...>& rhs
	)
	{
		CLONE_UNORDERED_MAP< T...>::my_base::swap(
			static_cast<const CLONE_UNORDERED_MAP< T...>::my_base&>(lhs),
			static_cast<const CLONE_UNORDERED_MAP< T...>::my_base&>(rhs)
		);
	}
	//specializes the std::swap algorithm
	//(freestanding binary function)
};
