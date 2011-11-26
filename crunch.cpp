// Compute exact winning probabilities for all preflop holdem matchups

#include <stdint.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

namespace {

// Enough bits to count all possible 9 card poker hands (2+2+5)
typedef uint64_t uint;

const char card_str[13+1] = "23456789TJQKA";

// cards_t consists of 13 3 bit chunks, counting the occurrences of each card without suit
typedef uint64_t cards_t;

struct hand_t {
    unsigned c0 : 4;
    unsigned c1 : 4;
    unsigned s : 1;

    hand_t(int c0,int c1,bool s)
        :c0(c0),c1(c1),s(s) {}

    cards_t cards() const {
        return (1L<<3*c0)+(1L<<3*c1);
    }

    friend ostream& operator<<(ostream& out, hand_t hand) {
        out<<card_str[hand.c0]<<card_str[hand.c1];
        if (hand.c0!=hand.c1)
            out<<(hand.s?'s':'o');
        return out;
    }
};

inline uint bit_stack(int b0, int b1, int b2, int b3) {
    return b0|(b1<<1)|(b2<<2)|(b3<<3);
}

// From Thomas Wang, http://www.concentric.net/~ttwang/tech/inthash.htm
inline uint hash(uint k) {
  k = (~k)+(k<<21);
  k = k^(k>>24);
  k = (k+(k<<3))+(k<<8);
  k = k^(k>>14);
  k = (k+(k<<2))+(k<<4);
  k = k^(k>>28);
  k = k+(k<<31);
  return k;
}

const uint one_of_each = 1L+(1L<<3)+(1L<<6)+(1L<<9)+(1L<<12)+(1L<<15)+(1L<<18)+(1L<<21)+(1L<<24)+(1L<<27)+(1L<<30)+(1L<<33)+(1L<<36);

// Extract the minimum bit, assuming a nonzero input
inline uint min_bit(uint x) {
    return x&-x;
}

// Drop the lowest card of a hand, assuming a nonempty hand
inline cards_t drop_card(cards_t cards) {
    uint low = min_bit(cards); 
    return cards-(one_of_each&(low|(low>>1)|(low>>2)));
}

// Drop the two lowest cards of a hand, assuming a hand with at least two cards
inline cards_t drop_two_cards(cards_t cards) {
    return drop_card(drop_card(cards));
}

inline uint popcount(uint x) {
    return __builtin_popcountl(x);
}

typedef __uint128_t score_t;

enum type_t {HIGH_CARD,PAIR_OR_TWO,TRIPS,STRAIGHT,FLUSH,FULL_HOUSE,QUADS,STRAIGHT_FLUSH};

score_t score_hand(cards_t cards) {
    #define SCORE(type,c0,c1) ((score_t(type)<<(64+39+3))|(score_t(c0)<<64)|(c1))
    // TODO: Check for straight flushes

    // Check for four of a kind
    uint quads = cards&(one_of_each<<2);
    if (quads)
        return SCORE(QUADS,quads,drop_two_cards(cards-quads));

    // Check for a full house
    uint trips = cards&(cards>>1)&one_of_each;
    uint pairs = ~trips&(cards>>1)&one_of_each;
    if (trips && pairs) {
        if (popcount(pairs)>1) pairs = drop_card(pairs);
        return SCORE(FULL_HOUSE,trips,pairs);
    }

    // TODO: Check for flushes

    // Check for straights
    cards_t unique = one_of_each&(cards|(cards>>1)|(cards>>2));
    cards_t wheel = (1L<<36)|1|(1L<<3)|(1L<<6)|(1L<<9);
    cards_t straights = ((unique<<3)&unique&(unique>>3)&(unique>>6)&(unique>>9))|((unique&wheel)==wheel);
    if (straights) {
        if (popcount(straights)>1) straights = drop_card(straights);
        if (popcount(straights)>1) straights = drop_card(straights);
        assert(popcount(straights)==1);
        return SCORE(STRAIGHT,0,straights);
    }

    // Check for three of a kind
    if (trips) {
        if (popcount(trips)>1) trips = drop_card(trips);
        return SCORE(TRIPS,trips,drop_two_cards(cards-3*trips));
    }

    // Check for pair or two pair
    if (pairs) {
        if (popcount(pairs)>2) pairs = drop_card(pairs);
        return SCORE(PAIR_OR_TWO,pairs,drop_two_cards(cards-2*pairs));
    }

    // Nothing interesting happened, so high card wins
    return SCORE(HIGH_CARD,0,drop_two_cards(cards));
    #undef SCORE
}

struct outcomes_t {
    uint alice,bob,tie;

    outcomes_t()
        :alice(0),bob(0),tie(0) {}

    outcomes_t& operator+=(outcomes_t o) {
        alice+=o.alice;bob+=o.bob;tie+=o.tie;
        return *this;
    }

    uint total() const {
        return alice+bob+tie;
    }
};

// Precomputed number of nontrivial permutations of 5 cards for each possible equality configuration
int interesting_permutations[16];

void compute_interesting_permutations() {
    for (int d0 = 0; d0 < 2; d0++)
        for (int d1 = 0; d1 < 2; d1++)
            for (int d2 = 0; d2 < 2; d2++)
                for (int d3 = 0; d3 < 2; d3++) {
                    int i0 = 0, i1 = i0+d0, i2 = i1+d1, i3 = i2+d2, i4 = i3+d3;
                    int i[5] = {i0,i1,i2,i3,i4};
                    int stabilizing = 0;
                    for (int j0 = 0; j0 < 5; j0++)
                        for (int j1 = 0; j1 < 5; j1++) if (j1 != j0)
                            for (int j2 = 0; j2 < 5; j2++) if (j2 != j0 && j2 != j1)
                                for (int j3 = 0; j3 < 5; j3++) if (j3 != j0 && j3 != j1 && j3 != j2) {
                                    int j4 = 0+1+2+3+4-j0-j1-j2-j3;
                                    int j[5] = {j0,j1,j2,j3,j4};
                                    for (int k = 0; k < 5; k++)
                                        if (i[j[k]] != i[k])
                                            goto nope;
                                    stabilizing++;
                                    nope:;
                                }
                    assert(120/stabilizing*stabilizing==120);
                    interesting_permutations[bit_stack(d0,d1,d2,d3)] = 120/stabilizing;
                }
}

// Consider all possible post-flop cards.  For speed, we consider hands only in decreasing order
// of card value, and account for the multiple ways such an ordered hand can occur via the above
// permutations array.
outcomes_t compare_hands(cards_t alice, cards_t bob) {
    const cards_t cards0 = alice + bob;
    const uint ways0 = 1;
    const int threads = omp_get_max_threads();
    outcomes_t outcomes[threads*threads];
    #define LOOP(bound,i,j) \
        for (int c##i = 0; c##i <= bound; c##i++) \
            if (uint ways##j = ways##i*(4-((cards##i>>3*c##i)&7))) \
                if (const cards_t cards##j = cards##i+(1L<<3*c##i))
    #pragma omp parallel for
    LOOP(12,0,1) {
        int t0 = omp_get_thread_num();
        #pragma omp parallel for
        LOOP(c0,1,2) {
            int t = t0*threads+omp_get_thread_num();
            LOOP(c1,2,3)
                LOOP(c2,3,4)
                    LOOP(c3,4,5) {
                        uint ways = ways5*interesting_permutations[bit_stack(c0!=c1,c1!=c2,c2!=c3,c3!=c4)];
                        score_t alice_score = score_hand(cards5-bob),
                                bob_score   = score_hand(cards5-alice);
                        if (alice_score>bob_score) outcomes[t].alice += ways;
                        else if (alice_score<bob_score) outcomes[t].bob += ways;
                        else outcomes[t].tie += ways;
                    }
        }
    }
    #undef LOOP

    // Sum outcomes from different threads
    outcomes_t total;
    for (int t = 0; t < threads*threads; t++)
        total += outcomes[t];
    return total;
}

void show_comparison(hand_t h0, hand_t h1,outcomes_t o) {
    cout<<h0<<" vs. "<<h1<<':'<<endl;
    cout<<"  Alice: "<<o.alice<<"/"<<o.total()<<" = "<<(double)o.alice/o.total()<<endl;
    cout<<"  Bob:   "<<o.bob<<"/"<<o.total()<<" = "<<(double)o.bob/o.total()<<endl;
    cout<<"  Tie:   "<<o.tie<<"/"<<o.total()<<" = "<<(double)o.tie/o.total()<<endl;
}

} // unnamed namespace

int main() {
    // Initialize
    compute_interesting_permutations();

    // Collect all hands
    vector<hand_t> hands;
    for (int c0 = 0; c0 < 13; c0++) {
        hands.push_back(hand_t(c0,c0,0));
        for (int c1 = 0; c1 < c0; c1++)
            for (int s = 0; s < 2; s++)
                hands.push_back(hand_t(c0,c1,s));
    }
    assert(hands.size()==169);

    // Print hands
    cout<<"hands =";
    for (uint i = 0; i < hands.size(); i++)
        cout<<' '<<hands[i];
    cout<<endl;

    // Compute equities for some (mostly random) pairs of hands
    if (0) {
        uint random = 0;
        for (int i = 0; i < 10; i++) {
            hand_t h0 = hands[hash(random++)%hands.size()];
            hand_t h1 = hands[hash(random++)%hands.size()];
            //if (hash(random++)%4==0) h1 = h0;
            show_comparison(h0,h1,compare_hands(h0.cards(),h1.cards()));
        }
    }

    // Compute all hand pair equities
    #pragma omp parallel for
    for (uint i = 0; i < hands.size(); i++)
        #pragma omp parallel for
        for (uint j = 0; j <= i; j++) {
            outcomes_t o = compare_hands(hands[i].cards(),hands[j].cards());
            #pragma omp critical
            show_comparison(hands[i],hands[j],o);
        }

    return 0;
}
