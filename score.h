// Header code for exact poker winning probabilities

#ifndef __exact_h__
#define __exact_h__

#ifdef __OPENCL_VERSION__
typedef uint uint32_t;
typedef uint4 uint32_t4;
typedef ulong uint64_t;
typedef ulong4 uint64_t4;
#else
#include <stdint.h>
typedef cl_uint4 uint32_t4;
typedef cl_ulong4 uint64_t4;
#define __global
#define __kernel
#define get_global_id(i) 0
#endif

// A 52-entry bit set representing the cards in a hand, in suit-value major order
typedef uint64_t cards_t;
typedef uint64_t4 cards_t4;

// A 32-bit integer representing the value of a 7 card hand
typedef uint32_t score_t;
typedef uint32_t4 score_t4;

// To make parallelization easy, we precompute the set of 5 element subsets of 48 elements.
// These are stored as 5 6-bit values packed into a 32-bit int
typedef uint32_t five_subset_t;
typedef uint32_t4 five_subset_t4;
#define NUM_FIVE_SUBSETS 1712304

// Hand types
#define HIGH_CARD      (1<<27)
#define PAIR           (2<<27)
#define TWO_PAIR       (3<<27)
#define TRIPS          (4<<27)
#define STRAIGHT       (5<<27)
#define FLUSH          (6<<27)
#define FULL_HOUSE     (7<<27)
#define QUADS          (8<<27)
#define STRAIGHT_FLUSH (9<<27)

#define TYPE_MASK (0xffff<<27)

#define BLOCK_SIZE 256

// Extract the minimum bit, assuming a nonzero input
#define min_bit(x) ((x)&-(x))

// OpenCL whines if we don't have prototypes
inline uint64_t hash(uint64_t k);
inline uint64_t hash2(uint64_t a, uint64_t b);
inline uint64_t hash3(uint64_t a, uint64_t b, uint64_t c);
#ifdef __OPENCL_VERSION__
inline uint64_t4 hash_4(uint64_t4 k);
inline uint64_t4 hash2_4(uint64_t4 a, uint64_t4 b);
inline uint64_t4 hash3_4(uint64_t4 a, uint64_t4 b, uint64_t4 c);
#endif
inline cards_t free_set(__global const cards_t* free, five_subset_t set);

// From Thomas Wang, http://www.concentric.net/~ttwang/tech/inthash.htm
#define DEFINE_HASH(name,type) \
    inline type name(type k) { \
        k = (~k)+(k<<21); \
        k = k^(k>>24); \
        k = (k+(k<<3))+(k<<8); \
        k = k^(k>>14); \
        k = (k+(k<<2))+(k<<4); \
        k = k^(k>>28); \
        k = k+(k<<31); \
        return k; \
    }
DEFINE_HASH(hash,uint64_t)
#ifdef __OPENCL_VERSION__
DEFINE_HASH(hash_4,uint64_t4)
#endif
#undef DEFINE_HASH

// From http://burtleburtle.net/bob/c/lookup8.c
#define DEFINE_HASH(name,type) \
    inline type name(type a, type b, type c) { \
        a -= b; a -= c; a ^= c>>43; \
        b -= c; b -= a; b ^= a<<9; \
        c -= a; c -= b; c ^= b>>8; \
        a -= b; a -= c; a ^= c>>38; \
        b -= c; b -= a; b ^= a<<23; \
        c -= a; c -= b; c ^= b>>5; \
        a -= b; a -= c; a ^= c>>35; \
        b -= c; b -= a; b ^= a<<49; \
        c -= a; c -= b; c ^= b>>11; \
        a -= b; a -= c; a ^= c>>12; \
        b -= c; b -= a; b ^= a<<18; \
        c -= a; c -= b; c ^= b>>22; \
        return c; \
    }
DEFINE_HASH(hash3,uint64_t)
#ifdef __OPENCL_VERSION__
DEFINE_HASH(hash3_4,uint64_t4)
#endif
#undef DEFINE_HASH

inline uint64_t hash2(uint64_t a, uint64_t b) {
    return hash3(hash(0),a,b);
}

#ifdef __OPENCL_VERSION__
inline uint64_t4 hash2_4(uint64_t4 a, uint64_t4 b) {
    return hash3_4(hash_4(0),a,b);
}
#endif

inline cards_t free_set(__global const cards_t* free, five_subset_t set) {
    #define F(i) free[set>>(6*i)&0x3f]
    return F(0)|F(1)|F(2)|F(3)|F(4);
    #undef F
}

#endif
