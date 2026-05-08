#pragma once
#include "StackJoiner.h"
#include <nupack/model/Move.h>
#include <nupack/model/ModelVariants.h>
#include <nupack/iteration/Search.h>
#include <nupack/algorithms/Utility.h>

namespace nupack::kmc {

/******************************************************************************************/

struct NoLoopJoiner {
    Mat<real> const &join_propensities() const {NUPACK_ERROR("joining not enabled");}

    void update(Ignore, Ignore, Ignore, Ignore, Ignore) {}

    vec<JoinLoc> all_join_locations(Ignore, Ignore, Ignore, Ignore, Ignore, Ignore, Ignore) const {return {};}

    JoinLoc choose_loc(Ignore, Ignore, Ignore, Ignore, Ignore, Ignore, Ignore) const {NUPACK_ERROR("joining not enabled");}
};

/******************************************************************************************/

struct QuadraticLoopJoiner {
    Mat<real> const &join_propensities() const {NUPACK_ERROR("join propensities not enabled");}

    void update(Ignore, Ignore, Ignore, Ignore, Ignore) {}

    template <class RF>
    vec<JoinLoc> all_join_locations(SubsequenceList const &v, uint p, uint, Base b1, Ignore, Ignore, RF const &rf) const {
        vec<JoinLoc> locs;
        for (auto s : iterators(v))  // iterate through the sequences
            for (auto b : iterators(*s).offset(1, -1)) if (*b == b1 && p == rf.bimolecular_half_contexts(b, *s).first) locs.emplace_back(s-v.begin(), b);
        return locs;
    }

    JoinLoc choose_loc(Ignore, Ignore, Ignore, Ignore, Ignore, Ignore, Ignore) const {NUPACK_ERROR("join choice not enabled based on only one loop");}
};

/******************************************************************************************/

struct ProductLoopJoiner {
    using Active = True;

    NUPACK_REFLECT(ProductLoopJoiner, propensities);

    Mat<real> propensities;
    ProductLoopJoiner() = default;

    /**************************************************************************************/

    Mat<real> const &join_propensities() const {return propensities;};

    template <class State, class EM, class RF>
    void update(State const &, SubsequenceList const &, int, EM const &, RF const &);

    /// Iterate over all locations in the loop with their partial dG and partial rate constant
    template <class EM, class RF, class F>
    void for_each_join_location(SubsequenceList const &, uint, uint, Base, Base, EM const &, RF const &, F &&) const;

    template <class State, class EM, class RF>
    JoinLoc choose_loc(SubsequenceList const &, double &, uint index, bool which, State const &, EM const &, RF const &rf) const;

    /// Return all locations that can form the specified base pair and pair of contexts
    template <class EM, class RF>
    vec<JoinLoc> all_join_locations(SubsequenceList const &, uint, uint, Base, Base, EM const &, RF const &) const;
};

/******************************************************************************************/

struct StackAccum {
    real *data;
    uint n, n1, nbp; // number of bases and base pairs
    uint mb, ml, mp;
    StackAccum(real *data, uint n, uint nbp) : data(data), n(n), n1(n+1), nbp(nbp), 
        mb(n1 * n1 * n), ml(mb + 2 * n), mp(ml + 4 * n * n1) {}

    void skip() const {}; // {++data;}
    void skip(std::size_t n) const {}; // {data += n;}
    uint index(Base b) const {return b == Base::null() ? n : +b;}

    void both_dangle(Base j, Base j0, Base j1, real t) const {data[index(j) * sq(n1) + index(j0) * n1 + index(j1)] += t;}

    void coaxial_both_left(Base j, real t) const {data[mb + index(j)] += t;}
    void coaxial_both_right(Base bi, real t) const {data[mb + n + index(bi)] += t;}
    
    void coaxial_top_left(Base j, Base j1, real t) const {data[ml + index(j) * n1 + index(j1)] += t;}
    void coaxial_bottom_left(Base j, Base j0, real t) const {data[ml + n * n1 + index(j) * n1 + index(j0)] += t;}

    void coaxial_top_right(Base bi, Base bi0, real t) const {data[ml + 2 * n * n1 + index(bi) * n1 + index(bi0)] += t;}
    void coaxial_bottom_right(Base bi, Base bi1, real t) const {data[ml + 3 * n * n1 + index(bi) * n1 + index(bi1)] += t;}

    void coaxial_top_left_bottom_right(uint bp, real t) const {data[mp + bp] += t;}
    void coaxial_bottom_left_top_right(uint bp, real t) const {data[mp + nbp + bp] += t;}
};

/******************************************************************************************/

// Remember that ProductLoopJoiner updates are replicated in BOTH for_each_join_location and update
template <class State, class EM, class RF>
void ProductLoopJoiner::update(State const &state, SubsequenceList const &v, int n, EM const &em, RF const &rf) {
    auto const &joiner = state.joiner.product_joiner();
    propensities.zeros(joiner.propensity_vector_size(em, rf), 2);
    NUPACK_ASSERT(rf.has_bimolecular_propensity(em.ensemble), "ProductJoiner not compatible with this rate function and ensemble", rf, em.ensemble);

    if (has_subensemble(em.ensemble)) {
        NUPACK_REQUIRE(rf.number_of_bimolecular_half_contexts(), ==, 1, "not implemented yet");
        auto const nick = find_nick(v);
        real * const l = propensities.memptr(), * const r = l + propensities.n_rows;
        StackEnvironment<real> const env(v, nick, em);
        for (auto s : indices(v)) for (auto i : indices(v[s]).offset(+1, -1)) {
            accumulate_left_stacks(StackAccum(l, em.alphabet().length(), len(joiner.base_pairs)), env, em, joiner.base_pairs, v, nick, s, i);
            accumulate_right_stacks(StackAccum(r, em.alphabet().length(), len(joiner.base_pairs)), env, em, joiner.base_pairs, v, nick, s, i);
        }
        propensities *= em.boltz(0.5 * em.join_penalty() - em.loop_energy(v, find_nick(v)));
    } else {
        uint const nc = rf.number_of_bimolecular_half_contexts();
        em.dangle_switch([&](auto const &dangle) {
            for (auto s : iterators(v)) {
                auto const old5 = safe_dangle5(dangle, v, s); // at beginning
                auto const old3 = safe_dangle3(dangle, v, s); // at end

                for (auto b : iterators(*s).offset(1, -1)) {
                    izip(joiner.base_pairs, [&](auto index, auto const &bp) {
                        if (*b != bp.first) return;
                        Base const c = bp.second;
                        real new3 = (b > begin_of(*s) + 1) ? dangle.energy3(b[-1], b[0], c) : real(); // at beginning
                        real new5 = (b < end_of(*s) - 2) ? dangle.energy5(c, b[0], b[1]) : real();    // at end

                        // Beginning of sequence
                        real dG = 0.5 * (em.join_penalty() + em.terminal_penalty(*b, c));
                        if (b == begin_of(*s) + 1) dG += new3 - old5;
                        else dG += dangle.combine(new3, old5, 2 == b - begin_of(*s)) - old5;

                        // End of sequence
                        if (b == end_of(*s) - 2) dG += new5 - old3;
                        else dG += dangle.combine(new5, old3, 3 == end_of(*s) - b) - old3;

                        auto const [i, j] = rf.bimolecular_half_contexts(b, *s);
                        for (auto bj :range(nc)) 
                            propensities(index * sq(nc) + i * nc + bj, 0) += rf.bimolecular_propensity(i, bj, -em.beta * dG);
                    });
                }
            }
        });
        izip(joiner.base_pairs, [&](auto ibp, auto const &p1) {
            izip(joiner.base_pairs, [&](auto jbp, auto const &p2) {
                if (p1.first == p2.second && p1.second == p2.first) {
                    for (auto i : range(nc)) for (auto j :range(nc))
                        propensities(jbp * sq(nc) + j * nc + i, 1) = propensities(ibp * sq(nc) + i * nc + j, 0);
                }
            });
        });   
    }
}

/******************************************************************************************/

// ProductLoopJoiner::choose_loc could be more efficient to just calculate at that index, right now just programmed to calculate the whole vector
template <class State, class EM, class RF>
JoinLoc ProductLoopJoiner::choose_loc(SubsequenceList const &v, double &r, uint index, bool which, State const &w, EM const &em, RF const &rf) const {
    auto const &joiner = w.joiner.product_joiner();
     if (has_subensemble(em.ensemble)) {
        auto const nick = find_nick(v);

        Col<real> search(joiner.propensity_vector_size(em, rf), la::fill::zeros);
        search(index) -= r * em.boltz(em.loop_energy(v, find_nick(v)) - 0.5 * em.join_penalty()); // take off constant scale factors

        StackEnvironment<real> const env(v, nick, em);
        for (auto s : indices(v)) for (auto i : indices(v[s]).offset(+1, -1)) {
            real const last = search(index);
            StackAccum const accumulator(search.memptr(), em.alphabet().length(), len(joiner.base_pairs));
            if (!which) accumulate_left_stacks(accumulator, env, em, joiner.base_pairs, v, nick, s, i);
            else accumulate_right_stacks(accumulator, env, em, joiner.base_pairs, v, nick, s, i);
            
            if (search(index) >= 0) {
                real const factor = search(index) - last;
                real const dlogp = std::log(factor);
                r = -last / factor;
                return JoinLoc(s, v[s].begin() + i, dlogp, rf.bimolecular_propensity(0, 0, dlogp));
            }
        }
        NUPACK_ERROR("Join choice failure (stacking)", v, r, index, which, w);
    } else {
        uint nc = rf.number_of_bimolecular_half_contexts(), cindex = index % sq(nc), p = cindex / nc, q = cindex % nc;
        auto [x, y] = joiner.base_pairs[index / sq(nc)];
        if (which) {swap(x, y); swap(p, q);}
        JoinLoc ret;
        for_each_join_location(v, p, q, x, y, em, rf, [&](auto i, auto b, auto dlogp, auto hrate) {
            if (!minus_divide_if(r, hrate)) return false;
            else ret = JoinLoc(i, b, dlogp, hrate); return true;
        });
        NUPACK_REQUIRE(ret.dlogp_propensity.value().second, >, 0, "Join choice failure (not stacking)", v, r, index, which, w);
        return ret;
    }
}

/******************************************************************************************/

// Iterate over all locations in the loop that can form base pair b1 to b2
// Remember that ProductLoopJoiner updates are replicated in BOTH for_each_join_location and update
template <class EM, class RF, class F>
void ProductLoopJoiner::for_each_join_location(SubsequenceList const &v, uint p, uint q, Base b1, Base b2, EM const &em, RF const &rf, F &&f) const {
    NUPACK_ASSERT(!has_subensemble(em.ensemble), "for_each_join_location should not be used in stacking ensemble");
    
    em.dangle_switch([&](auto const &dangle) { // switch on the specified type of dangle
        for (auto s : iterators(v)) { // iterate through the sequences
            auto const old5 = safe_dangle5(dangle, v, s); // old dangle contribution at beginning
            auto const old3 = safe_dangle3(dangle, v, s); // old dangle contribution at end

            for (auto const b : iterators(*s).offset(1, -1)) if (*b == b1) {
                auto const [p1, q1] = rf.bimolecular_half_contexts(b, *s);
                if (p1 != p) continue;
                real new3 = (b > begin_of(*s) + 1) ? dangle.energy3(b[-1], b[0], b2): real(); // at beginning
                real new5 = (b < end_of(*s) - 2) ? dangle.energy5(b2, b[0], b[1]) : real();   // at end

                // Split join penalty in 2 so it is properly counted
                real dG = 0.5 * (em.join_penalty() + em.terminal_penalty(b1, b2));
                // Beginning of sequence
                if (b == begin_of(*s) + 1) dG += new3 - old5;
                else dG += dangle.combine(new3, old5, 2 == b - begin_of(*s)) - old5;

                // End of sequence -- use dangle.combine to delegate none, min, all dangles
                if (b == end_of(*s) - 2) dG += new5 - old3;
                else dG += dangle.combine(new5, old3, 3 == end_of(*s) - b) - old3;
                NUPACK_ASSERT(contains_iter(*s, b));

                // Call the callback with the base location information, free energy, and partial rate constant
                // if the callback returns true, short-circuit and return early.
                real const dlogp = -em.beta * dG;
                if (f(s - begin_of(v), b, dlogp, rf.bimolecular_propensity(p, q, dlogp))) return;
            }
        }
    });
}

/******************************************************************************************/

template <class EM, class RF>
vec<JoinLoc> ProductLoopJoiner::all_join_locations(SubsequenceList const &v, uint p, uint q, Base b1, Base b2, EM const &em, RF const &rf) const {
    vec<JoinLoc> ret;
    /// Iterate over all locations in the loop with their partial dG and partial rate constant
    for_each_join_location(v, p, q, b1, b2, em, rf, [&](auto i, auto b, auto dlogp, auto hrate){
        ret.emplace_back(i, b, dlogp, hrate);
        return false; // dont short circuit out of the iteration over locations
    });
    return ret;
}

/******************************************************************************************/

using LoopJoiner = Variant<NoLoopJoiner, ProductLoopJoiner, QuadraticLoopJoiner>;

}
