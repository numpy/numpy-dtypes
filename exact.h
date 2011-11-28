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

// OpenCL whines if we don't have prototypes
inline cards_t free_set(__global const cards_t* free, struct five_subset_t set);

inline cards_t free_set(__global const cards_t* free, struct five_subset_t set) {
    return free[set.i0]|free[set.i1]|free[set.i2]|free[set.i3]|free[set.i4];
}

#endif
