// Compute exact winning probabilities for all preflop holdem matchups

#include <stdint.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <omp.h>

using std::ostream;
using std::cout;
using std::cerr;
using std::endl;
using std::flush;
using std::vector;
using std::string;

namespace {

// Enough bits to count all possible 9 card poker hands (2+2+5)
typedef uint64_t uint;

const char show_card[13+1] = "23456789TJQKA";
const char show_suit[4+1] = "shdc";

// A 52-entry bit set representing the cards in a hand
typedef uint64_t cards_t;

cards_t read_cards(const char* s) {
    uint n = strlen(s);
    assert(!(n&1));
    n /= 2;
    cards_t cards = 0;
    for (uint i = 0; i < n; i++) {
        const char* p = strchr(show_card,s[2*i]);
        assert(p);
        const char* q = strchr(show_suit,s[2*i+1]);
        assert(q);
        cards |= 1L<<(4*(p-show_card)+(q-show_suit));
    }
    return cards;
}

string show_cards(cards_t cards) {
    string r;
    for (int c = 0; c < 13; c++)
        for (int s = 0; s < 4; s++)
            if (cards&(1L<<(4*c+s))) {
                r += show_card[c];
                r += show_suit[s];
            }
    return r;
}

struct hand_t {
    unsigned card0 : 4;
    unsigned card1 : 4;
    unsigned suited : 1;

    hand_t(int card0,int card1,bool suited)
        :card0(card0),card1(card1),suited(suited) {}

    bool operator==(const hand_t h) {
        return card0==h.card0 && card1==h.card1 && suited==h.suited;
    }

    friend ostream& operator<<(ostream& out, hand_t hand) {
        out<<show_card[hand.card0]<<show_card[hand.card1];
        if (hand.card0!=hand.card1)
            out<<(hand.suited?'s':'o');
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

// From http://burtleburtle.net/bob/c/lookup8.c
inline uint hash(uint a, uint b, uint c) {
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

inline uint hash(uint a, uint b) {
    return hash(hash(0),a,b);
}

inline uint hash(__uint128_t k) {
    return hash(k>>64,k);
}

// Extract the minimum bit, assuming a nonzero input
inline uint min_bit(uint x) {
    return x&-x;
}

// Drop the lowest bit, assuming a nonzero input
inline uint drop_bit(uint x) {
    return x-min_bit(x);
}

// Drop the two lowest bits, assuming at least two bits set
inline uint drop_two_bits(uint x) {
    return drop_bit(drop_bit(x));
}

inline uint popcount(uint x) {
    return __builtin_popcountl(x);
}

typedef __uint128_t score_t;

const score_t HIGH_CARD      = score_t(1)<<108,
              PAIR           = score_t(1)<<109,
              TWO_PAIR       = score_t(1)<<110,
              TRIPS          = score_t(1)<<111,
              STRAIGHT       = score_t(1)<<112,
              FLUSH          = score_t(1)<<113,
              FULL_HOUSE     = score_t(1)<<114,
              QUADS          = score_t(1)<<115,
              STRAIGHT_FLUSH = score_t(1)<<116;

const score_t type_mask = score_t(0xffff)<<108;

const char* show_type(score_t type) {
    switch (type) {
        case HIGH_CARD:      return "high-card";
        case PAIR:           return "pair";
        case TWO_PAIR:       return "two-pair";
        case TRIPS:          return "trips";
        case STRAIGHT:       return "straight";
        case FLUSH:          return "flush";
        case FULL_HOUSE:     return "full-house";
        case QUADS:          return "quads";
        case STRAIGHT_FLUSH: return "straight-flush";
        default:             return "<unknown>";
    }
}

// Equivalent to popcount, but assumes cards == cards&0x1111111111111
inline int flush_popcount(cards_t cards) {
    uint c = cards;
    c += c>>(8*4);
    c += c>>(4*4);
    c += c>>(2*4);
    c += c>>(1*4);
    return c&15;
}

// If there's a flush, return all cards of that suit (i.e., possibly more than 5 cards),
// but shifted down to be independent of suit.
inline cards_t all_flushes(cards_t cards) {
    for (int s = 0; s < 4; s++) {
        uint cs = (cards>>s)&0x1111111111111;
        if (flush_popcount(cs)>=5)
            return cs;
    }
    return 0;
}

// Find all straights in a (suited) set of cards, assuming cards == cards&0x1111111111111
inline uint all_straights(cards_t unique) {
    const cards_t wheel = (1L<<48)|1|(1L<<4)|(1L<<8)|(1L<<12);
    return ((unique<<4)&unique&(unique>>4)&(unique>>8)&(unique>>12))|((unique&wheel)==wheel);
}

// Find the maximum bit set of x, assuming x has at most n bits set, where n <= 3
inline uint max_bit(uint x, int n) {
    if (n>1 && x!=min_bit(x)) x -= min_bit(x);
    if (n>2 && x!=min_bit(x)) x -= min_bit(x);
    return x;
}

// Determine the best possible five card hand out of a bit set of seven cards
score_t score_hand(cards_t cards) {
    #define SCORE(type,c0,c1) (type|(score_t(c0)<<(52+4))|(c1))
    const cards_t each = 0x1111111111111;

    // Check for straight flushes
    const cards_t flushes = all_flushes(cards);
    if (flushes) {
        const uint straight_flushes = all_straights(flushes);
        if (straight_flushes)
            return SCORE(STRAIGHT_FLUSH,0,max_bit(straight_flushes,3));
    }

    // Check for four of a kind
    const uint quads = each&cards&(cards>>1)&(cards>>2)&(cards>>3);
    const cards_t unique = each&(cards|(cards>>1)|(cards>>2)|(cards>>3));
    if (quads)
        return SCORE(QUADS,quads,max_bit(unique-quads,3));

    // Check for a full house
    const uint counts = (cards&each)+((cards>>1)&each)+((cards>>2)&each)+((cards>>3)&each);
    const uint trips = counts&(counts>>1)&each;
    const uint pairs = ~trips&(counts>>1)&each;
    if (trips) {
        if (pairs) // If there are pairs, there can't be two kinds of trips
            return SCORE(FULL_HOUSE,trips,max_bit(pairs,2));
        else if (trips!=min_bit(trips)) // Two kind of trips: use only two of the lower one
            return SCORE(FULL_HOUSE,trips-min_bit(trips),min_bit(trips));
    }

    // Check for flushes
    if (flushes) {
        const int count = flush_popcount(flushes);
        cards_t best = flushes;
        if (count>5) best -= min_bit(best);
        if (count>6) best -= min_bit(best);
        return SCORE(FLUSH,0,best);
    }

    // Check for straights
    const cards_t straights = all_straights(unique);
    if (straights)
        return SCORE(STRAIGHT,0,max_bit(straights,3));

    // Check for three of a kind
    if (trips)
        return SCORE(TRIPS,trips,drop_two_bits(unique-trips));

    // Check for pair or two pair
    if (pairs) {
        if (pairs==min_bit(pairs))
            return SCORE(PAIR,pairs,drop_two_bits(unique-pairs));
        const uint high_pairs = drop_bit(pairs);
        if (high_pairs==min_bit(high_pairs))
            return SCORE(TWO_PAIR,pairs,drop_two_bits(unique-pairs));
        return SCORE(TWO_PAIR,high_pairs,drop_bit(unique-high_pairs));
    }

    // Nothing interesting happened, so high cards win
    return SCORE(HIGH_CARD,0,drop_two_bits(unique));
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

// To make parallelization easy, we precompute the set of 5 element subsets of 48 elements 
struct five_subset_t {
    unsigned i0 : 6;
    unsigned i1 : 6;
    unsigned i2 : 6;
    unsigned i3 : 6;
    unsigned i4 : 6;
};
five_subset_t five_subsets[1712304];

void compute_five_subsets() {
    int n = 0;
    for (int i0 = 0; i0 < 48; i0++)
        for (int i1 = 0; i1 < i0; i1++)
            for (int i2 = 0; i2 < i1; i2++)
                for (int i3 = 0; i3 < i2; i3++)
                    for (int i4 = 0; i4 < i3; i4++) {
                        five_subsets[n].i0 = i0;
                        five_subsets[n].i1 = i1;
                        five_subsets[n].i2 = i2;
                        five_subsets[n].i3 = i3;
                        five_subsets[n].i4 = i4;
                        n++;
                    }
}

// Consider all possible sets of shared cards to determine the probabilities of wins, losses, and ties.
// For efficiency, the set of shared cards is generated in decreasing order (this saves a factor of 5! = 120).
outcomes_t compare_hands(hand_t alice, hand_t bob) {
    const int threads = omp_get_max_threads();
    outcomes_t outcomes[threads];
    // We fix the suits of Alice's cards
    const int sa0 = 0, sa1 = !alice.suited;
    const cards_t alice_cards = (1L<<(4*alice.card0+sa0))|(1L<<(4*alice.card1+sa1));
    // Consider all compatible suits of Bob's cards
    for (int sb0 = 0; sb0 < 4; sb0++)
        for (int sb1 = 0; sb1 < 4; sb1++)
            if ((sb0==sb1)==bob.suited) {
                const cards_t bob_cards = (1L<<(4*bob.card0+sb0))|(1L<<(4*bob.card1+sb1));
                const cards_t hand_cards = alice_cards|bob_cards;
                // Make sure we don't use the same card twice
                if (popcount(hand_cards)<4) continue;
                // Make a list of the cards we're allowed to use
                cards_t free[48];
                for (int c = 0, i = 0; c < 52; c++)
                    if (!((1L<<c)&hand_cards))
                        free[i++] = 1L<<c;
                // Consider all possible sets of shared cards
                #pragma omp parallel for
                for (int n = 0; n < 1712304; n++) {
                    const int t = omp_get_thread_num();
                    const five_subset_t set = five_subsets[n];
                    const cards_t shared_cards = free[set.i0]|free[set.i1]|free[set.i2]|free[set.i3]|free[set.i4];
                    const score_t alice_score = score_hand(shared_cards|alice_cards),
                                  bob_score   = score_hand(shared_cards|bob_cards);
                    if (alice_score>bob_score) outcomes[t].alice++;
                    else if (alice_score<bob_score) outcomes[t].bob++;
                    else outcomes[t].tie++;
                }
            }

    // Sum outcomes from different threads
    outcomes_t total;
    for (int t = 0; t < threads; t++)
        total += outcomes[t];
    return total;
}

void show_comparison(hand_t alice, hand_t bob,outcomes_t o) {
    cout<<alice<<" vs. "<<bob<<':'<<endl;
    cout<<"  Alice: "<<o.alice<<"/"<<o.total()<<" = "<<(double)o.alice/o.total()<<endl;
    cout<<"  Bob:   "<<o.bob<<"/"<<o.total()<<" = "<<(double)o.bob/o.total()<<endl;
    cout<<"  Tie:   "<<o.tie<<"/"<<o.total()<<" = "<<(double)o.tie/o.total()<<endl;
    if (alice==bob && o.alice!=o.bob) {
        cout<<"  Error: Identical hands should win equally often"<<endl;
        exit(1);
    }
}

void test_score_hand() {
    const char *Alice = "Alice", *tie = "tie", *Bob = "Bob";
    struct test_t {
        const char *alice,*bob,*shared;
        score_t alice_type,bob_type;
        const char* result;
    };
    const test_t tests[] = {
        {"As2d","KsTc","Qh3h7h9d4c",HIGH_CARD,HIGH_CARD,Alice},         // high card wins
        {"Ks2d","AsTc","Qh3h7h9d4c",HIGH_CARD,HIGH_CARD,Bob},           // high card wins
        {"4s2d","5s3c","QhAh7h9dTc",HIGH_CARD,HIGH_CARD,tie},           // only five cards matter
        {"4s3d","5s3c","QhAh7h9d2c",HIGH_CARD,HIGH_CARD,Bob},           // the fifth card matters
        {"4s3d","4d3c","QhAh7h9d2c",HIGH_CARD,HIGH_CARD,tie},           // suits don't matter
        {"As2d","KsTc","Qh3h7h9d2c",PAIR,HIGH_CARD,Alice},              // pair beats high card
        {"Ks2d","AsTc","Qh3h7h9d2c",PAIR,HIGH_CARD,Alice},              // pair beats high card
        {"Ks2d","AsTc","KhAh7h9d3c",PAIR,PAIR,Bob},                     // higher pair wins
        {"Ks2d","KdTc","KhAh7h9d3c",PAIR,PAIR,Bob},                     // pair + higher kicker wins
        {"KsTd","Kd2c","KhAh7h9d3c",PAIR,PAIR,Alice},                   // pair + higher kicker wins
        {"Ks3d","Kd2c","KhAh7h9d6c",PAIR,PAIR,tie},                     // given a pair, only three other cards matter
        {"7s6d","5d4c","KhKdJh9d8c",PAIR,PAIR,tie},                     // given a pair, only three other cards matter
        {"7s6d","5d4c","7d5h4hAdKc",PAIR,TWO_PAIR,Bob},                 // two pair beats higher pair
        {"2s6d","5d4c","2d5h4hAdKc",PAIR,TWO_PAIR,Bob},                 // two pair beats lower pair
        {"7s2d","5d4c","2h5h4h7dKc",TWO_PAIR,TWO_PAIR,Alice},           // the higher pair matters
        {"7s2d","7d2c","2h5h4h7hKc",TWO_PAIR,TWO_PAIR,tie},             // two pairs can tie
        {"7sAd","7dQc","Kh5h4h7hKc",TWO_PAIR,TWO_PAIR,Alice},           // two pair + higher kicker wins
        {"KsAd","QdAc","JhJcThTc2c",TWO_PAIR,TWO_PAIR,tie},             // only one kicker matters with two pair
        {"JsAd","QdAc","AhJcKhKc2c",TWO_PAIR,TWO_PAIR,Bob},             // three pair doesn't matter
        {"JsAd","QdKc","JhJcQhKs2c",TRIPS,TWO_PAIR,Alice},              // trips beat two pair
        {"JsAd","QdKc","ThTcTs3s2c",TRIPS,TRIPS,Alice},                 // trips + highest kicker wins
        {"9s8d","7d6c","ThTcTsAsKc",TRIPS,TRIPS,tie},                   // only two kickers matter with trips
        {"Ts8d","QdJc","ThTc2sAsKc",TRIPS,STRAIGHT,Bob},                // straight beats trips
        {"Ts8d","QdJc","2h3c4s5s6c",STRAIGHT,STRAIGHT,tie},             // kickers don't matter with straights
        {"Ah5c","Tc2h","6d7h8c9dAs",STRAIGHT,STRAIGHT,Bob},             // highest straight wins
        {"AhJc","5cKh","2d3h4c5d5h",STRAIGHT,TRIPS,Alice},              // aces can be low in straights
        {"AhJc","6cKh","2d3h4c5d5h",STRAIGHT,STRAIGHT,Bob},             // the wheel is the lowest straight
        {"AhJc","6c2d","Th3h4h5d5h",FLUSH,STRAIGHT,Alice},              // flush beats straight
        {"AhJc","6h2d","Th3h4h5d5h",FLUSH,FLUSH,Alice},                 // highest flush wins
        {"7h6c","6h2d","AhKhQh9h8h",FLUSH,FLUSH,tie},                   // only five cards matter in a flush
        {"7h6h","5h2h","AhKhQh9h8h",FLUSH,FLUSH,tie},                   // only five cards matter in a flush
        {"7d6d","5h2h","7h7c6hTh8h",FULL_HOUSE,FLUSH,Alice},            // full house beats flush
        {"7d6d","6c6s","7h7c6h9h8h",FULL_HOUSE,FULL_HOUSE,Alice},       // with two full houses, higher trips win
        {"7d7s","6c6s","7h2c6h9h9s",FULL_HOUSE,FULL_HOUSE,Alice},       // with two full houses, higher trips win
        {"7d7s","6c6s","9c2c6h9h9s",FULL_HOUSE,FULL_HOUSE,Alice},       // if the trips match, higher pairs win
        {"AdKd","QcJs","9c6c6h9h9s",FULL_HOUSE,FULL_HOUSE,tie},         // there are no kickers in full houses
        {"AdKd","AcQs","AsAhQhQdKs",FULL_HOUSE,FULL_HOUSE,Alice},       // two trips don't matter
        {"2d2c","AcQs","AsAhQh2h2s",QUADS,FULL_HOUSE,Alice},            // quads beat a full house
        {"2d2c","3c3s","3d3hQh2h2s",QUADS,QUADS,Bob},                   // higher quads win
        {"Ad7c","Qc3s","2d2cQh2h2s",QUADS,QUADS,Alice},                 // quads + higher kicker wins
        {"AdKc","AcQs","2d2cQh2h2s",QUADS,QUADS,tie},                   // only one kicker matters with quads
        {"2d3d","AcAs","AdAh4d5d6d",STRAIGHT_FLUSH,QUADS,Alice},        // straight flush beats quads
        {"Ts8s","QsJs","2s3s4s5s6s",STRAIGHT_FLUSH,STRAIGHT_FLUSH,tie}, // kickers don't matter with straight flushes
        {"Ah5c","Tc2h","6c7c8c9cKh",STRAIGHT_FLUSH,STRAIGHT_FLUSH,Bob}, // highest straight flush wins
        {"AhJc","5c5s","2h3h4h5d5h",STRAIGHT_FLUSH,QUADS,Alice},        // aces can be low in straight flushes
        {"AhJc","6hKh","2h3h4h5h5d",STRAIGHT_FLUSH,STRAIGHT_FLUSH,Bob}, // the steel wheel is the lowest straight flush
        {"7d8h","7h2c","2h3h4h5h6h",STRAIGHT_FLUSH,STRAIGHT_FLUSH,Bob}, // higher straight flush beats higher flush and straight
    };

    for (uint i = 0; i < sizeof(tests)/sizeof(test_t); i++) {
        const cards_t alice  = read_cards(tests[i].alice),
                      bob    = read_cards(tests[i].bob),
                      shared = read_cards(tests[i].shared);
        if (popcount(alice|bob|shared)!=9) {
            cout<<"test "<<tests[i].alice<<' '<<tests[i].bob<<' '<<tests[i].shared<<" has duplicated cards"<<endl;
            exit(1);
        }
        const score_t alice_score = score_hand(alice|shared),
                      bob_score   = score_hand(bob|shared),
                      alice_type  = alice_score&type_mask,
                      bob_type    = bob_score&type_mask;
        const char* result = alice_score>bob_score?Alice:alice_score<bob_score?Bob:tie;
        if (alice_type!=tests[i].alice_type || bob_type!=tests[i].bob_type || result!=tests[i].result) {
            cout<<"test "<<tests[i].alice<<' '<<tests[i].bob<<' '<<tests[i].shared
                <<": expected "<<show_type(tests[i].alice_type)<<' '<<show_type(tests[i].bob_type)<<' '<<tests[i].result
                <<", got "<<show_type(alice_type)<<' '<<show_type(bob_type)<<' '<<result<<endl;
            exit(1);
        }
    }
}

void regression_test_score_hand() {
    // Score a large number of hands
    const uint m = 1<<17, n = 1<<10;
    vector<uint> hashes(m);
    #pragma omp parallel for
    for (uint i = 0; i < m; i++) {
        for (uint j = 0; j < n; j++) {
            cards_t cards = 0;
            for (int k = 0; k < 7; k++)
                cards |= 1L<<hash(i,j,k)%52;
            if (popcount(cards)<7)
                continue;
            hashes[i] = hash(hashes[i],hash(score_hand(cards)));
        }
        if (i%n==0) {
            #pragma omp critical
            cout<<'.'<<flush;
        }
    }
    cout << endl;

    // Merge hashes
    uint merged = 0;
    for (uint i = 0; i < m; i++)
        merged = hash(merged,hashes[i]);

    const uint expected = 0x87e12072088a7eaa;
    if (merged!=expected) {
        cout<<"regression test: expected 0x"<<std::hex<<expected<<", got 0x"<<merged<<std::dec<<endl;
        exit(1);
    } else
        cout<<"regression test passed!"<<endl;
}

void usage(const char** argv) {
    cerr<<"usage: "<<argv[0]<<" hands|test|some|all"<<endl;
}

} // unnamed namespace

int main(int argc, const char** argv) {
    if (argc!=2) {
        usage(argv);
        return 1;
    }
    string cmd = argv[1];

    // Initialize
    compute_five_subsets();

    // Collect all hands
    vector<hand_t> hands;
    for (int c0 = 0; c0 < 13; c0++) {
        hands.push_back(hand_t(c0,c0,0));
        for (int c1 = 0; c1 < c0; c1++)
            for (int s = 0; s < 2; s++)
                hands.push_back(hand_t(c0,c1,s));
    }
    assert(hands.size()==169);

    // Run a few tests
    test_score_hand();

    // Print hands
    if (cmd=="hands") {
        cout<<"hands =";
        for (uint i = 0; i < hands.size(); i++)
            cout<<' '<<hands[i];
        cout<<endl;
    }

    // Run more expensive tests
    else if (cmd=="test")
        regression_test_score_hand();

    // Compute equities for some (mostly random) pairs of hands
    else if (cmd=="some") {
        uint random = 0;
        for (int i = 0; i < 10; i++) {
            hand_t h0 = hands[hash(random++)%hands.size()];
            hand_t h1 = hands[hash(random++)%hands.size()];
            show_comparison(h0,h1,compare_hands(h0,h1));
        }
    }

    // Compute all hand pair equities
    else if (cmd=="all")
        for (uint i = 0; i < hands.size(); i++)
            for (uint j = 0; j <= i; j++)
                show_comparison(hands[i],hands[j],compare_hands(hands[i],hands[j]));

    // Didn't understand command
    else {
        usage(argv);
        cerr<<"unknown command: "<<cmd<<endl;
        return 1;
    }

    return 0;
}
