// Kernel code for exact poker winning probabilities

#include "score.h"

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
    uint64_tv sum = 0;
    for (int i = 0; i < BLOCK_SIZE/4; i++)
        sum += compare_cards(alice_cards,bob_cards,free,vload4(0,five_subsets+offset+4*i));
    results[id] = sum.s0+sum.s1+sum.s2+sum.s3;
}

inline cards_tv mostly_random_set(uint64_tv r) {
    cards_tv cards = 0;
    #define ADD(a) \
        uint64_tv i##a = r>>(6*a)&0x3f; \
        i##a = if_gel(i##a,52,i##a-52,i##a); \
        const cards_tv b##a = (cards_t)1<<i##a; \
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
        const score_tv score = score_hand(mostly_random_set(hash2v(i,(uint64_tv)(4*j,4*j+1,4*j+2,4*j+3))));
        h = hash2(hash2(hash2(hash2(h,score.s0),score.s1),score.s2),score.s3);
    }
    results[id] = h;
}
