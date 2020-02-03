/* *************************************************************

Copyright 2010 Terry Lyons, Stephen Buckley, Djalil Chafai, 
Greg Gyurkó and Arend Janssen. 

Distributed under the terms of the GNU General Public License, 
Version 3. (See accompanying file License.txt)

************************************************************* */



#pragma once
#ifndef basis_traits_h__
#define basis_traits_h__
/// Basis traits

/// Without_degree: an algebra with a product but no meaningful degree function defined on basis KEYs.
/// With_Degree: an algebra with degree(KEY) defined and with the property that 
/// degree(KEY1) + degree(KEY2) > max_degree => prod(KEY1,KEY2)=0.
/// No_Product: no product structure.
enum basis_product_trait
{
	Without_Degree,
	With_Degree,
	No_Product
};

/// The basis' properties

/// This structure stores the number of letters: NO_LETTERS, maximum degree: MAX_DEGREE 
/// and the product_trait, either Without_Degree, With_Degree or No_Product.
template <
	basis_product_trait trait = Without_Degree,
	DEG no_letters = 0,
	DEG max_degree = 0>
struct basis_traits
{

	/// The number of letters used to generate the algebra
	/// zero if there is no well defined finite generating set
	static const DEG NO_LETTERS = no_letters;	
	/// The trait of the product; either Without_Degree, With_Degree or No_Product
	static const basis_product_trait PRODUCT_TYPE = trait;
	/// The maximum degree
	static const DEG MAX_DEGREE = (trait == With_Degree)? max_degree : DEG(0);

};

#endif // basis_traits_h__
