// Header code for exact poker winning probabilities

#ifndef __exact_h__
#define __exact_h__

#ifdef __OPENCL_VERSION__
typedef uint uint32_t;
typedef ulong uint64_t;
#else
#include <stdint.h>
#define __global
#define __kernel
#define get_global_id(i) 0
#endif

// A 52-entry bit set representing the cards in a hand, in suit-value major order
typedef uint64_t cards_t;

// A 32-bit integer representing the value of a 7 card hand
typedef uint32_t score_t;

// To make parallelization easy, we precompute the set of 5 element subsets of 48 elements.
struct five_subset_t {
    unsigned i0 : 6;
    unsigned i1 : 6;
    unsigned i2 : 6;
    unsigned i3 : 6;
    unsigned i4 : 6;
};
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
inline cards_t free_set(__global const cards_t* free, struct five_subset_t set);

// From Thomas Wang, http://www.concentric.net/~ttwang/tech/inthash.htm
inline uint64_t hash(uint64_t k) {
    k = (~k)+(k<<21);
    k = k^(k>>24);
    k = (k+(k<<3))+(k<<8);
    k = k^(k>>14);
    k = (k+(k<<2))+(k<<4);
    k = k^(k>>28);
    k = k+(k<<31);
    return k;
}

// From http://burtleburtle.net/bob/c/lookup8.c
inline uint64_t hash3(uint64_t a, uint64_t b, uint64_t c) {
    a -= b; a -= c; a ^= c>>43;
    b -= c; b -= a; b ^= a<<9;
    c -= a; c -= b; c ^= b>>8;
    a -= b; a -= c; a ^= c>>38;
    b -= c; b -= a; b ^= a<<23;
    c -= a; c -= b; c ^= b>>5;
    a -= b; a -= c; a ^= c>>35;
    b -= c; b -= a; b ^= a<<49;
    c -= a; c -= b; c ^= b>>11;
    a -= b; a -= c; a ^= c>>12;
    b -= c; b -= a; b ^= a<<18;
    c -= a; c -= b; c ^= b>>22;
    return c;
}

inline uint64_t hash2(uint64_t a, uint64_t b) {
    return hash3(hash(0),a,b);
}

inline cards_t free_set(__global const cards_t* free, struct five_subset_t set) {
    return free[set.i0]|free[set.i1]|free[set.i2]|free[set.i3]|free[set.i4];
}

#endif
