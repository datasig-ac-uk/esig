#ifndef RdToPowers2_h__
#define RdToPowers2_h__

typedef double SCA;
typedef SCA* PSCA;

void prods(PSCA& now, SCA val, size_t k, const size_t D, const SCA* ptv, const SCA* end);

enum prodsswitch{
	Prods2 = 1,
	Prods_test = 2,
	Prods_nonrecursive3 = 3,
	Prods_nonrecursive2 = 4,
	Prods_nonrecursive = 5,
	Prods_wei1 = 6,
	Prods_cheb = 7,
	Prods = 8
};



#ifdef __cplusplus
extern "C"
{
#endif
void RdToPowers(void* pIn, SCA* pOut, void* vpCBufferHelper);
extern prodsswitch prodmethod;
#ifdef __cplusplus
}
#endif

#endif // RdToPowers2_h__
