// Kernel code for exact poker winning probabilities

#include "exact.h"

// OpenCL whines if we don't have prototypes
inline score_t drop_bit(score_t x);
inline score_t drop_two_bits(score_t x);
inline cards_t count_suits(cards_t cards);
inline uint32_t cards_with_suit(cards_t cards, cards_t suits);
inline score_t all_straights(score_t unique);
inline score_t max_bit(score_t x, int n);
score_t score_hand(cards_t cards);
inline uint64_t compare_cards(cards_t alice_cards, cards_t bob_cards, __global const cards_t* free, struct five_subset_t set);
inline cards_t mostly_random_set(uint64_t r);

// Drop the lowest bit, assuming a nonzero input
inline score_t drop_bit(score_t x) {
    return x-min_bit(x);
}

// Drop the two lowest bits, assuming at least two bits set
inline score_t drop_two_bits(score_t x) {
    return drop_bit(drop_bit(x));
}

// Count the number of cards in each suit in parallel
inline cards_t count_suits(cards_t cards) {
    const cards_t suits = 1+((cards_t)1<<13)+((cards_t)1<<26)+((cards_t)1<<39);
    cards_t s = cards; // initially, each suit has 13 single bit chunks
    s = (s&suits*0x1555)+(s>>1&suits*0x0555); // reduce each suit to 1 single bit and 6 2-bit chunks
    s = (s&suits*0x1333)+(s>>2&suits*0x0333); // reduce each suit to 1 single bit and 3 4-bit chunks
    s = (s&suits*0x0f0f)+(s>>4&suits*0x010f); // reduce each suit to 2 8-bit chunks
    s = (s+(s>>8))&suits*0xf; // reduce each suit to 1 16-bit count (only 4 bits of which can be nonzero)
    return s;
}

// Given a set of cards and a set of suits, find the set of cards with that suit
inline uint32_t cards_with_suit(cards_t cards, cards_t suits) {
    cards_t c = cards&suits*0x1fff;
    c |= c>>13;
    c |= c>>26;
    return c&0x1fff;
}

// Find all straights in a (suited) set of cards, assuming cards == cards&0x1111111111111
inline score_t all_straights(score_t unique) {
    const score_t wheel = (1<<12)|1|2|4|8;
    const score_t u = unique&unique<<1;
    return (u&u>>2&unique>>3)|((unique&wheel)==wheel);
}

// Find the maximum bit set of x, assuming x has at most n bits set, where n <= 3
inline score_t max_bit(score_t x, int n) {
    if (n>1 && x!=min_bit(x)) x -= min_bit(x);
    if (n>2 && x!=min_bit(x)) x -= min_bit(x);
    return x;
}

// Determine the best possible five card hand out of a bit set of seven cards
score_t score_hand(cards_t cards) {
    #define SCORE(type,c0,c1) ((type)|((c0)<<14)|(c1))
    const score_t each_card = 0x1fff;
    const cards_t each_suit = 1+((cards_t)1<<13)+((cards_t)1<<26)+((cards_t)1<<39);

    // Check for straight flushes
    const cards_t suits = count_suits(cards);
    const cards_t flushes = each_suit&(suits>>2)&(suits>>1|suits); // Detect suits with at least 5 cards
    if (flushes) {
        const score_t straight_flushes = all_straights(cards_with_suit(cards,flushes));
        if (straight_flushes)
            return SCORE(STRAIGHT_FLUSH,0,max_bit(straight_flushes,3));
    }

    // Check for four of a kind
    const score_t cand = cards&cards>>26;
    const score_t cor  = (cards|cards>>26)&each_card*(1+(1<<13));
    const score_t quads = cand&cand>>13;
    const score_t unique = each_card&(cor|cor>>13);
    if (quads)
        return SCORE(QUADS,quads,max_bit(unique-quads,3));

    // Check for a full house
    const score_t trips = (cand&cor>>13)|(cor&cand>>13);
    const score_t pairs = each_card&~trips&(cand|cand>>13|(cor&cor>>13));
    if (trips) {
        if (pairs) // If there are pairs, there can't be two kinds of trips
            return SCORE(FULL_HOUSE,trips,max_bit(pairs,2));
        else if (trips!=min_bit(trips)) // Two kind of trips: use only two of the lower one
            return SCORE(FULL_HOUSE,trips-min_bit(trips),min_bit(trips));
    }

    // Check for flushes
    if (flushes) {
        const int count = cards_with_suit(suits,flushes);
        score_t best = cards_with_suit(cards,flushes);
        if (count>5) best -= min_bit(best);
        if (count>6) best -= min_bit(best);
        return SCORE(FLUSH,0,best);
    }

    // Check for straights
    const score_t straights = all_straights(unique);
    if (straights)
        return SCORE(STRAIGHT,0,max_bit(straights,3));

    // Check for three of a kind
    if (trips)
        return SCORE(TRIPS,trips,drop_two_bits(unique-trips));

    // Check for pair or two pair
    if (pairs) {
        if (pairs==min_bit(pairs))
            return SCORE(PAIR,pairs,drop_two_bits(unique-pairs));
        const cards_t high_pairs = drop_bit(pairs);
        if (high_pairs==min_bit(high_pairs))
            return SCORE(TWO_PAIR,pairs,drop_two_bits(unique-pairs));
        return SCORE(TWO_PAIR,high_pairs,drop_bit(unique-high_pairs));
    }

    // Nothing interesting happened, so high cards win
    return SCORE(HIGH_CARD,0,drop_two_bits(unique));
    #undef SCORE
}

// Evaluate a full set of hands and shared cards
inline uint64_t compare_cards(cards_t alice_cards, cards_t bob_cards, __global const cards_t* free, struct five_subset_t set) {
    const cards_t shared_cards = free_set(free,set);
    const score_t alice_score = score_hand(shared_cards|alice_cards),
                  bob_score   = score_hand(shared_cards|bob_cards);
    return alice_score>bob_score?(uint64_t)1<<32:alice_score<bob_score?1u:0;
}

// Score a bunch of hands
__kernel void score_hands_kernel(__global const cards_t* cards, __global score_t* results) {
    const int id = get_global_id(0);
    results[id] = score_hand(cards[id]);
}

// Given Alice's and Bob's hands, determine outcomes for one block of shared cards
__kernel void compare_cards_kernel(__global const struct five_subset_t* five_subsets, __global const cards_t* free, __global uint64_t* results, const cards_t alice_cards, const cards_t bob_cards) {
    const int id = get_global_id(0);
    const int offset = id*BLOCK_SIZE;
    //const int bound = min(BLOCK_SIZE,NUM_FIVE_SUBSETS-offset);
    uint64_t sum = 0;
    for (int i = 0; i < BLOCK_SIZE; i++)
        sum += compare_cards(alice_cards,bob_cards,free,five_subsets[offset+i]);
    results[id] = sum;
}

inline cards_t mostly_random_set(uint64_t r) {
    cards_t cards = 0;
    #define ADD(a) \
        int i##a = (r>>(6*a)&0x3f)%52; \
        cards_t b##a = (cards_t)1<<i##a; \
        cards |= cards&b##a?min_bit(~cards):b##a;
    ADD(0) ADD(1) ADD(2) ADD(3) ADD(4) ADD(5) ADD(6)
    #undef ADD
    return cards;
}

// Hash a bunch of mostly random hand scores
__kernel void hash_scores_kernel(__global uint64_t* results, const uint64_t offset) {
    const int id = get_global_id(0), i = offset+id;
    uint64_t h = 0;
    for (int j = 0; j < 1024; j++)
        h = hash2(h,score_hand(mostly_random_set(hash2(i,j))));
    results[id] = h;
}
