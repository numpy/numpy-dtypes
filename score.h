// Header code for exact poker winning probabilities

#ifndef __exact_h__
#define __exact_h__

#ifdef __OPENCL_VERSION__
typedef uint uint32_t;
typedef uint4 uint32_tv;
typedef ulong uint64_t;
typedef ulong4 uint64_tv;
#else
#include <stdint.h>
#include <algorithm>
typedef uint32_t uint32_tv;
typedef uint64_t uint64_tv;
#define __global
#define __kernel
#define get_global_id(i) 0
using std::max;
#endif

// A 52-entry bit set representing the cards in a hand, in suit-value major order
typedef uint64_t cards_t;
typedef uint64_tv cards_tv;

// A 32-bit integer representing the value of a 7 card hand
typedef uint32_t score_t;
typedef uint32_tv score_tv;

// To make parallelization easy, we precompute the set of 5 element subsets of 48 elements.
// These are stored as 5 6-bit values packed into a 32-bit int
typedef uint32_t five_subset_t;
typedef uint32_tv five_subset_tv;
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

// Extract the minimum bit, assuming a nonzero input (2 operations)
#define min_bit(x) ((x)&-(x))

// OpenCL whines if we don't have prototypes
inline uint64_t hash(uint64_t k);
inline uint64_t hash2(uint64_t a, uint64_t b);
inline uint64_t hash3(uint64_t a, uint64_t b, uint64_t c);
#ifdef __OPENCL_VERSION__
inline uint64_tv hashv(uint64_tv k);
inline uint64_tv hash2v(uint64_tv a, uint64_tv b);
inline uint64_tv hash3v(uint64_tv a, uint64_tv b, uint64_tv c);
#endif
inline score_tv drop_bit(score_tv x);
inline score_tv drop_two_bits(score_tv x);
inline cards_tv count_suits(cards_tv cards);
inline score_tv cards_with_suit(cards_tv cards, cards_tv suits);
inline score_tv all_straights(score_tv unique);
inline score_tv max_bit(score_tv x);
score_tv score_hand(cards_tv cards);
inline uint64_tv compare_cards(cards_t alice_cards, cards_t bob_cards, __global const cards_t* free, five_subset_tv set);
inline cards_tv mostly_random_set(uint64_tv r);
inline cards_t free_set(__global const cards_t* free, five_subset_t set);
inline cards_tv free_sets(__global const cards_t* free, five_subset_tv set);

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
DEFINE_HASH(hashv,uint64_tv)
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
DEFINE_HASH(hash3v,uint64_tv)
#endif
#undef DEFINE_HASH

inline uint64_t hash2(uint64_t a, uint64_t b) {
    return hash3(hash(0),a,b);
}

#ifdef __OPENCL_VERSION__
inline uint64_tv hash2v(uint64_tv a, uint64_tv b) {
    return hash3v(hash(0),a,b);
}
#endif

// Drop the lowest bit (3 operations)
inline score_tv drop_bit(score_tv x) {
    return x-min_bit(x);
}

// Drop the two lowest bits (6 operations)
inline score_tv drop_two_bits(score_tv x) {
    return drop_bit(drop_bit(x));
}

// Count the number of cards in each suit in parallel (15 operations)
inline cards_tv count_suits(cards_tv cards) {
    const cards_t suits = 1+((cards_t)1<<13)+((cards_t)1<<26)+((cards_t)1<<39);
    cards_tv s = cards; // initially, each suit has 13 single bit chunks
    s = (s&suits*0x1555)+(s>>1&suits*0x0555); // reduce each suit to 1 single bit and 6 2-bit chunks
    s = (s&suits*0x1333)+(s>>2&suits*0x0333); // reduce each suit to 1 single bit and 3 4-bit chunks
    s = (s&suits*0x0f0f)+(s>>4&suits*0x010f); // reduce each suit to 2 8-bit chunks
    s = (s+(s>>8))&suits*0xf; // reduce each suit to 1 16-bit count (only 4 bits of which can be nonzero)
    return s;
}

#ifdef __OPENCL_VERSION__
#define convert_score convert_uint4
#define convert_cards convert_ulong4
#else
#define convert_score(c) ((score_tv)(c))
#define convert_cards(c) ((cards_tv)(c))
#define isequal(a,b) ((a)==(b))
#define isnotequal(a,b) ((a)!=(b))
#define isgreater(a,b) ((a)>(b))
#define isgreaterequal(a,b) ((a)>=(b))
#define select(a,b,c) ((c)?(b):(a))
#endif

// Given a set of cards and a set of suits, find the set of cards with that suit (7 operations)
inline score_tv cards_with_suit(cards_tv cards, cards_tv suits) {
    cards_tv c = cards&suits*0x1fff;
    c |= c>>13;
    c |= c>>26;
    return convert_score(c)&0x1fff;
}

// Non-branching ternary operators.  All the 0* stuff is to make overload resolution work.  It should disappear at compile time.
// I'm counting each of these as two operations.
#define DEFINE_IFS(suffix,type) \
    inline type if_nz##suffix(type c, type a, type b); \
    inline type if_eq##suffix(type x, type y, type a, type b); \
    inline type if_ne##suffix(type x, type y, type a, type b); \
    inline type if_gt##suffix(type x, type y, type a, type b); \
    inline type if_ge##suffix(type x, type y, type a, type b); \
    inline type if_nz1##suffix(type c, type a); \
    inline type if_eq1##suffix(type x, type y, type a); \
    inline type if_ne1##suffix(type x, type y, type a); \
    inline type if_nz##suffix(type c, type a, type b) { return select(a,b,isequal(c,0)); } \
    inline type if_eq##suffix(type x, type y, type a, type b) { return select(b,a,isequal(x,y)); } \
    inline type if_ne##suffix(type x, type y, type a, type b) { return select(b,a,isnotequal(x,y)); } \
    inline type if_gt##suffix(type x, type y, type a, type b) { return select(b,a,isgreater(x,y)); } \
    inline type if_ge##suffix(type x, type y, type a, type b) { return select(b,a,isgreaterequal(x,y)); } \
    inline type if_nz1##suffix(type c, type a) { return if_nz##suffix(c,a,0); } \
    inline type if_eq1##suffix(type x, type y, type a) { return if_eq##suffix(x,y,a,0); } \
    inline type if_ne1##suffix(type x, type y, type a) { return if_ne##suffix(x,y,a,0); }
DEFINE_IFS(,uint32_tv)
DEFINE_IFS(l,uint64_tv)
#undef DEFINE_IFS

// Find all straights in a (suited) set of cards, assuming cards == cards&0x1111111111111 (8 operations)
inline score_tv all_straights(score_tv unique) {
    const score_tv u = unique&(unique<<1|unique>>12); // the ace wraps around to the bottom
    return u&u>>2&unique>>3;
}

#ifndef __OPENCL_VERSION__
#define clz __builtin_clz
#endif

// Find the maximum bit set of x, assuming x is nonzero (2 operations)
inline score_tv max_bit(score_tv x) {
    return ((score_t)1<<31)>>clz(x);
}

// Determine the best possible five card hand out of a bit set of seven cards (40+19+26+30+16+13+26+4 = 174 operations)
score_tv score_hand(cards_tv cards) {
    #define SCORE(type,c0,c1) ((type)|((c0)<<14)|(c1)) // 3 operations
    const score_t each_card = 0x1fff;
    const cards_t each_suit = 1+((cards_t)1<<13)+((cards_t)1<<26)+((cards_t)1<<39);

    // Check for straight flushes (15+5+8+7+3+2 = 40 operations)
    const cards_tv suits = count_suits(cards);
    const cards_tv flushes = each_suit&(suits>>2)&(suits>>1|suits); // Detect suits with at least 5 cards
    const score_tv straight_flushes = all_straights(cards_with_suit(cards,flushes));
    score_tv score = if_nz1(straight_flushes,SCORE(STRAIGHT_FLUSH,0,max_bit(straight_flushes)));

    // Check for four of a kind (2+3+2+3+1+2+3+2+1 = 19 operations)
    const score_tv cand = convert_score(cards&cards>>26);
    const score_tv cor  = convert_score(cards|cards>>26)&each_card*(1+(1<<13));
    const score_tv quads = cand&cand>>13;
    const score_tv unique = each_card&(cor|cor>>13);
    score = max(score,if_nz1(quads,SCORE(QUADS,quads,max_bit(unique-quads))));

    // Check for a full house (5+4+7+1+1+3+2+3 = 26 operations)
    const score_tv all_trips = (cand&cor>>13)|(cor&cand>>13);
    const score_tv trips = if_nz1(all_trips,max_bit(all_trips));
    const score_tv pairs_and_trips = each_card&(cand|cand>>13|(cor&cor>>13));
    const score_tv pairs = pairs_and_trips-trips;
    score = max(score,select((score_tv)0,SCORE(FULL_HOUSE,trips,max_bit(pairs)),(pairs!=0)&(trips!=0)));

    // Check for flushes (7+7+2*(2+1+2)+1+2+3 = 30 operations)
    const score_tv suit_count = cards_with_suit(suits,flushes);
    score_tv suited = cards_with_suit(cards,flushes);
    suited = if_gt(suit_count,5u,suited-min_bit(suited),suited);
    suited = if_gt(suit_count,6u,suited-min_bit(suited),suited);
    score = max(score,if_nz1(suited,SCORE(FLUSH,0,suited)));

    // Check for straights (8+1+2+3+2 = 16 operations)
    const score_tv straights = all_straights(unique);
    score = max(score,if_nz1(straights,SCORE(STRAIGHT,0,max_bit(straights))));

    // Check for three of a kind (7+1+2+3 = 13 operations)
    const score_tv kickers = drop_two_bits(unique-pairs_and_trips);
    score = max(score,if_nz1(trips,SCORE(TRIPS,trips,kickers)));

    // Check for pair or two pair (3+1+2+2+2+3+2+2+3+3+2+1 = 26 operations)
    const score_tv high_pairs = drop_bit(pairs);
    score = max(score,if_nz1(pairs,
        if_eq(pairs,min_bit(pairs),SCORE(PAIR,pairs,kickers),
        if_eq(high_pairs,min_bit(high_pairs),SCORE(TWO_PAIR,pairs,kickers),
        SCORE(TWO_PAIR,high_pairs,max_bit(unique-high_pairs))))));

    // Nothing interesting happened, so high cards win (1+3 = 4 operations)
    score = max(score,SCORE(HIGH_CARD,0,kickers));
    return score;
    #undef SCORE
}

inline cards_t free_set(__global const cards_t* free, five_subset_t set) {
    #define F(i) free[set>>(6*i)&0x3f]
    return F(0)|F(1)|F(2)|F(3)|F(4);
    #undef F
}

inline cards_tv free_sets(__global const cards_t* free, five_subset_tv set) {
#ifdef __OPENCL_VERSION__
    return (cards_tv)(free_set(free,set.s0),free_set(free,set.s1),free_set(free,set.s2),free_set(free,set.s3));
#else
    return free_set(free,set);
#endif
}

// Evaluate a full set of hands and shared cards
inline uint64_tv compare_cards(cards_t alice_cards, cards_t bob_cards, __global const cards_t* free, five_subset_tv set) {
    const cards_tv shared_cards = free_sets(free,set);
    const cards_tv alice_score = convert_cards(score_hand(shared_cards|alice_cards)),
                   bob_score   = convert_cards(score_hand(shared_cards|bob_cards));
    return if_gtl(alice_score,bob_score,(uint64_tv)1<<32,
           if_gtl(bob_score,alice_score,(uint64_tv)1,(uint64_tv)0));
}

#endif
