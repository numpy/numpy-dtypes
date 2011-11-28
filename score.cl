// Kernel code for exact poker winning probabilities

#include "score.h"

// OpenCL whines if we don't have prototypes
inline score_t4 drop_bit(score_t4 x);
inline score_t4 drop_two_bits(score_t4 x);
inline cards_t4 count_suits(cards_t4 cards);
inline score_t4 cards_with_suit(cards_t4 cards, cards_t4 suits);
inline score_t4 all_straights(score_t4 unique);
inline score_t4 max_bit2(score_t4 x);
inline score_t4 max_bit3(score_t4 x);
score_t4 score_hand(cards_t4 cards);
inline uint64_t4 compare_cards(cards_t alice_cards, cards_t bob_cards, __global const cards_t* free, five_subset_t4 set);
inline cards_t4 mostly_random_set(uint64_t4 r);

// Drop the lowest bit, assuming a nonzero input
inline score_t4 drop_bit(score_t4 x) {
    return x-min_bit(x);
}

// Drop the two lowest bits, assuming at least two bits set
inline score_t4 drop_two_bits(score_t4 x) {
    return drop_bit(drop_bit(x));
}

// Count the number of cards in each suit in parallel
inline cards_t4 count_suits(cards_t4 cards) {
    const cards_t suits = 1+((cards_t)1<<13)+((cards_t)1<<26)+((cards_t)1<<39);
    cards_t4 s = cards; // initially, each suit has 13 single bit chunks
    s = (s&suits*0x1555)+(s>>1&suits*0x0555); // reduce each suit to 1 single bit and 6 2-bit chunks
    s = (s&suits*0x1333)+(s>>2&suits*0x0333); // reduce each suit to 1 single bit and 3 4-bit chunks
    s = (s&suits*0x0f0f)+(s>>4&suits*0x010f); // reduce each suit to 2 8-bit chunks
    s = (s+(s>>8))&suits*0xf; // reduce each suit to 1 16-bit count (only 4 bits of which can be nonzero)
    return s;
}

// Given a set of cards and a set of suits, find the set of cards with that suit
inline score_t4 cards_with_suit(cards_t4 cards, cards_t4 suits) {
    cards_t4 c = cards&suits*0x1fff;
    c |= c>>13;
    c |= c>>26;
    return convert_uint4(c)&0x1fff;
}

// Non-branching ternary operators.  All the 0* stuff is to make overload resolution work.  It should disappear at compile time.
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
DEFINE_IFS(,uint32_t4)
DEFINE_IFS(l,uint64_t4)
#undef DEFINE_IFS

// Find all straights in a (suited) set of cards, assuming cards == cards&0x1111111111111
inline score_t4 all_straights(score_t4 unique) {
    const score_t wheel = (1<<12)|1|2|4|8;
    const score_t4 u = unique&unique<<1;
    return (u&u>>2&unique>>3)|if_eq1(unique&wheel,wheel,(score_t4)1);
}

// Find the maximum bit set of x, assuming x has at most 2 bit sets
inline score_t4 max_bit2(score_t4 x) {
    score_t4 m = min_bit(x);
    return if_eq(x,m,x,x-m);
}

// Find the maximum bit set of x, assuming x has at most 3 bit sets
inline score_t4 max_bit3(score_t4 x) {
    return max_bit2(max_bit2(x));
}

// Determine the best possible five card hand out of a bit set of seven cards
score_t4 score_hand(cards_t4 cards) {
    #define SCORE(type,c0,c1) ((type)|((c0)<<14)|(c1))
    const score_t each_card = 0x1fff;
    const cards_t each_suit = 1+((cards_t)1<<13)+((cards_t)1<<26)+((cards_t)1<<39);

    // Check for straight flushes
    const cards_t4 suits = count_suits(cards);
    const cards_t4 flushes = each_suit&(suits>>2)&(suits>>1|suits); // Detect suits with at least 5 cards
    const score_t4 straight_flushes = all_straights(cards_with_suit(cards,flushes));
    score_t4 score = if_nz1(straight_flushes,SCORE(STRAIGHT_FLUSH,0,max_bit3(straight_flushes)));

    // Check for four of a kind
    const score_t4 cand = convert_uint4(cards&cards>>26);
    const score_t4 cor  = convert_uint4(cards|cards>>26)&each_card*(1+(1<<13));
    const score_t4 quads = cand&cand>>13;
    const score_t4 unique = each_card&(cor|cor>>13);
    score = max(score,if_nz1(quads,SCORE(QUADS,quads,max_bit3(unique-quads))));

    // Check for a full house
    const score_t4 trips = (cand&cor>>13)|(cor&cand>>13);
    const score_t4 pairs = each_card&~trips&(cand|cand>>13|(cor&cor>>13));
    const score_t4 min_trips = min_bit(trips);
    score = max(score,if_nz1(trips,
        if_nz(pairs,SCORE(FULL_HOUSE,trips,max_bit2(pairs)), // If there are pairs, there can't be two kinds of trips
        if_ne1(trips,min_trips,SCORE(FULL_HOUSE,trips-min_trips,min_trips))))); // Two kind of trips: use only two of the lower one

    // Check for flushes
    const score_t4 suit_count = cards_with_suit(suits,flushes);
    score_t4 suited = cards_with_suit(cards,flushes);
    suited = if_gt(suit_count,5u,suited-min_bit(suited),suited);
    suited = if_gt(suit_count,6u,suited-min_bit(suited),suited);
    score = max(score,if_nz1(suited,SCORE(FLUSH,0,suited)));

    // Check for straights
    const score_t4 straights = all_straights(unique);
    score = max(score,if_nz1(straights,SCORE(STRAIGHT,0,max_bit3(straights))));

    // Check for three of a kind
    score = max(score,if_nz1(trips,SCORE(TRIPS,trips,drop_two_bits(unique-trips))));

    // Check for pair or two pair
    const score_t4 high_pairs = drop_bit(pairs);
    score = max(score,if_nz1(pairs,
        if_eq(pairs,min_bit(pairs),SCORE(PAIR,pairs,drop_two_bits(unique-pairs)),
        if_eq(high_pairs,min_bit(high_pairs),SCORE(TWO_PAIR,pairs,drop_two_bits(unique-pairs)),
        SCORE(TWO_PAIR,high_pairs,drop_bit(unique-high_pairs))))));

    // Nothing interesting happened, so high cards win
    score = max(score,SCORE(HIGH_CARD,0,drop_two_bits(unique)));
    return score;
    #undef SCORE
}

// Evaluate a full set of hands and shared cards
inline uint64_t4 compare_cards(cards_t alice_cards, cards_t bob_cards, __global const cards_t* free, five_subset_t4 set) {
    const cards_t4 shared_cards = (cards_t4)(free_set(free,set.s0),free_set(free,set.s1),free_set(free,set.s2),free_set(free,set.s3));
    const cards_t4 alice_score = convert_ulong4(score_hand(shared_cards|alice_cards)),
                    bob_score  = convert_ulong4(score_hand(shared_cards|bob_cards));
    return if_gtl(alice_score,bob_score,(uint64_t4)1<<32,
           if_gtl(bob_score,alice_score,(uint64_t4)1,(uint64_t4)0));
}

// Score a bunch of hands
__kernel void score_hands_kernel(__global const cards_t* cards, __global score_t* results) {
    const int id = get_global_id(0);
    vstore4(score_hand(vload4(0,cards+4*id)),0,results+4*id);
}

// Given Alice's and Bob's hands, determine outcomes for one block of shared cards
__kernel void compare_cards_kernel(__global const five_subset_t* five_subsets, __global const cards_t* free, __global uint64_t* results, const cards_t alice_cards, const cards_t bob_cards) {
    const int id = get_global_id(0);
    const int offset = id*BLOCK_SIZE;
    //const int bound = min(BLOCK_SIZE,NUM_FIVE_SUBSETS-offset);
    uint64_t4 sum = 0;
    for (int i = 0; i < BLOCK_SIZE/4; i++)
        sum += compare_cards(alice_cards,bob_cards,free,vload4(0,five_subsets+offset+4*i));
    results[id] = sum.s0+sum.s1+sum.s2+sum.s3;
}

inline cards_t4 mostly_random_set(uint64_t4 r) {
    cards_t4 cards = 0;
    #define ADD(a) \
        uint64_t4 i##a = r>>(6*a)&0x3f; \
        i##a = if_gel(i##a,52,i##a-52,i##a); \
        const cards_t4 b##a = (cards_t)1<<i##a; \
        cards |= if_nzl(cards&b##a,min_bit(~cards),b##a);
    ADD(0) ADD(1) ADD(2) ADD(3) ADD(4) ADD(5) ADD(6)
    #undef ADD
    return cards;
}

// Hash a bunch of mostly random hand scores
__kernel void hash_scores_kernel(__global uint64_t* results, const uint64_t offset) {
    const int id = get_global_id(0), i = offset+id;
    uint64_t h = 0;
    for (int j = 0; j < 1024/4; j++) {
        const score_t4 score = score_hand(mostly_random_set(hash2_4(i,(uint64_t4)(4*j,4*j+1,4*j+2,4*j+3))));
        h = hash2(hash2(hash2(hash2(h,score.s0),score.s1),score.s2),score.s3);
    }
    results[id] = h;
}
