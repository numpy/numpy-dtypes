Poker analysis tools
====================

This code consists of two parts: an exact win probability calculator written
in C++ and OpenCL, and Python code analyzing Nash equilibria.  Here are
details about each one:

Exact win/loss probabilities
----------------------------

The `exact` program computes the exact rational number probability of a win,
loss, or tie given any pair of preflop hold'em hands.  The code is written in
a combination of C++ and OpenCL (with a bit of OpenMP to take advantage of
multiple OpenCL devices).  The core routine is a vectorized, branch free
function that takes a 7 card hand represented as a bit set and uses bit
twiddling to check for the various poker hands (straights, flushes, pairs, etc.).

Computing the entire table of win probabilities for all pairs of preflop hands
takes around 5 minutes on a recent machine.  For convenience (and regression
testing) the results are included as exact.txt.

### Usage

To build `exact`, run

    make

Currently the Makefile is specific to Mac OS X, but this can easily be fixed.
The only dependencies are OpenCL and OpenMP.  On Mac this means 10.6 or later
is required.

To rebuild the table of exact probabilities from scratch, run

    make exact.txt

which does

    time ./exact all > exact.txt

Other ways to invoke exact include

    ./exact           # print usage information
    ./exact hands     # print the list of two card hold'em hands
    ./exact test      # run regression tests
    ./exact some 100  # compute win/loss/tie probabilities for 100 random pairs of hands

Nash equilibria
---------------

The `heads-up` Python script computes Nash equilibria for an extremely simplified
hold'em game with the following rules:

1. There are two players, Alice and Bob.
2. Bob posts a $1 blind.
3. Alice either folds or raises a fixed amount b, for a total bet of $1+b.
4. Bob either calls or folds.

To emphasize, the bet amount b is frozen in advance; Alice does not get to choose it.

### Dependencies

`heads-up` has the following dependencies

* [python](http://python.org)
* [numpy](http://numpy.scipy.org)
* [scipy](http://www.scipy.org)
* [matplotlib](http://matplotlib.sourceforge.net)
* [cvxopt](http://abel.ee.ucla.edu/cvxopt)

On a Mac, these can be obtained through [MacPorts](http://www.macports.org) via

    sudo port install py26-numpy py26-scipy py26-matplotlib py26-cvxopt

### Usage

Here are some example ways to use `heads-up`:

    ./heads-up -h                    # print usage information
    ./heads-up -b 2                  # determine the Nash equilibrium for bet level 2
    ./heads-up --plot -b 10 -n 100   # plot Alice's equity for b = 0 to 10
    ./heads-up --max -b 10           # find the bet level that maximizes Alice's equity

Contributing
------------

Want to contribute?  Great!  The Nash equilibrium computation could be generalized in
many different ways, and `exact` (which is already almost entirely an exercise in
pointless optimization) could always be optimized further.  If you have changes you
want to push back, either send pull requests or email to ask for commit access.
