#pragma once
#include <nupack/thermo/Compute.h>
#include <nupack/thermo/Strategy.h>

/******************************************************************************************/

namespace nupack {

template <class T>
struct ValueGrad {
    T value, gradient;

    NUPACK_REFLECT(ValueGrad, value, gradient);

    ValueGrad() = default;
    ValueGrad(T v) : value(v), gradient(0) {}
    ValueGrad(T v, T g) : value(v), gradient(g) {}

    friend ValueGrad operator/(ValueGrad const &a, ValueGrad const &b) {return {a.value / b.value, (b.value * a.gradient - a.value * b.gradient) / sq(b.value)};}
    friend ValueGrad operator*(ValueGrad const &a, ValueGrad const &b) {return {a.value * b.value, a.value * b.gradient + b.value * a.gradient};}
    friend ValueGrad operator+(ValueGrad const &a, ValueGrad const &b) {return {a.value + b.value, a.gradient + b.gradient};}
    friend ValueGrad operator-(ValueGrad const &a, ValueGrad const &b) {return {a.value - b.value, a.gradient - b.gradient};}

    ValueGrad operator-() const {return {-value, -gradient};}
    ValueGrad& operator+=(ValueGrad const &s) {return *this = *this + s;}

    friend bool operator<(ValueGrad const &a, ValueGrad const &b) {return a.value < b.value;}
    friend bool operator>(ValueGrad const &a, ValueGrad const &b) {return a.value > b.value;}
    friend bool operator==(ValueGrad const &a, ValueGrad const &b) {return a.value == b.value;}
    friend bool operator!=(ValueGrad const &a, ValueGrad const &b) {return a.value != b.value;}
    friend bool operator<=(ValueGrad const &a, ValueGrad const &b) {return a.value <= b.value;}
    friend bool operator>=(ValueGrad const &a, ValueGrad const &b) {return a.value >= b.value;}
};

template <class T>
struct infinity_t<ValueGrad<T>> {
    ValueGrad<T> operator()() const {return inf<T>();}
};

#define NUPACK_TMP(NAME, ...) \
template <class T> \
struct NAME##_functor_t<ValueGrad<T>> {auto operator()(ValueGrad<T> const &s) const __VA_ARGS__};

NUPACK_TMP(exp, {auto v = exp(s.value); return ValueGrad{v, v * s.gradient};})
NUPACK_TMP(log, {return ValueGrad{log(s.value), s.gradient / s.value};})
NUPACK_TMP(exp2, {auto v = exp2(s.value); return ValueGrad{v, v * s.gradient * simd::LogOf2<T>};})
NUPACK_TMP(sign, {return sign(s.value);})
NUPACK_TMP(is_finite, {return is_finite(s.value);})

#undef NUPACK_TMP

}

/******************************************************************************************/

namespace nupack::simd {

template <class T>
struct DoubleDispatch<ValueGrad<T>, ValueGrad<T>> {
    using S = ValueGrad<T>;
    static S plus(S const &t, S const &u) noexcept {return t + u;}
    static S multiplies(S const &t, S const &u) noexcept {return t * u;}
    static S min(S const &t, S const &u) noexcept {return min(t, u);}
    static S max(S const &t, S const &u) noexcept {return max(t, u);}

    static void set_logarithm(S &t, S const &u) noexcept {t = log(u);}
};


template <class T>
struct SingleDispatch<ValueGrad<T>> {
    using S = ValueGrad<T>;
    static S reciprocal(S const &t) noexcept {return S(1) / t;}
};

}

/******************************************************************************************/

namespace nupack {

template <class T>
Model<ValueGrad<T>> enthalpy_model(Ensemble e, ParameterFile const &p={}, ModelConditions const &cs={}, Optional<BasePairing> pairs={}) {
    auto const dG = load_parameter_set(ParameterInfo{.file=p, .kind="dG", .na_molarity=cs.na_molarity, .mg_molarity=cs.mg_molarity, .temperature=cs.temperature});
    auto const dGdB = load_parameter_set(ParameterInfo{.file=p, .kind="dGdB", .na_molarity=cs.na_molarity, .mg_molarity=cs.mg_molarity, .temperature=cs.temperature});
    ParameterSet<ValueGrad<T>> params(dG);
    zip(params.data, dGdB.data, [](ValueGrad<T> &x, T const &dGdB) {x.gradient = dGdB;});
    Model<ValueGrad<T>> model(e, std::move(params), cs, std::move(pairs));
    model.beta.gradient = 1;
    return model;
}

}

/******************************************************************************************/

namespace nupack::thermo {

template <class T>
struct DefaultFactory<Model<ValueGrad<T>>> {
    static ComputerList create(Model<ValueGrad<T>>);
};

template <class T>
ComputerList DefaultFactory<Model<ValueGrad<T>>>::create(Model<ValueGrad<T>> m) {
    ComputerList v;
    fork(m.ensemble_type(), [&](auto d) {
        v.emplace_back(make_computer<PFStrategy<Model<ValueGrad<T>>, decltype(d), 3>>(m));
        v.emplace_back(make_computer<PFStrategy<Model<ValueGrad<T>>, decltype(d), 3>>(m));
    });
    return v;
};

/******************************************************************************************/

template <class T>
struct Finalizer<ValueGrad<T>> {
    Result::Pairs operator()(Matrix<ValueGrad<T>> P, Job::Pairs const &p) const {
        auto const n = P.shape()[0];
        Mat<real> V(n, n, la::fill::none), G(n, n, la::fill::none);
        std::transform(P.begin(), P.end(), V.begin(), [](ValueGrad<T> const &x) {return x.value;});
        std::transform(P.begin(), P.end(), G.begin(), [](ValueGrad<T> const &x) {return x.gradient;});

        if (auto const logq = V(0, 0); std::isfinite(logq)) {
            V = la::exp(V + V.t() - logq);
            V.diag().zeros();
            V.diag() = 1 - la::sum(V, 0);
            G = V % (G + G.t() - G(0, 0));
            G.diag().zeros();
            for (auto &x : G) if (std::isnan(x)) x = 0;
            G.diag() = -la::sum(G, 0);
            NUPACK_QUICK_ASSERT(V.is_finite(), V, logq);
        } else {
            V.zeros();
            V.diag().ones();
            G.zeros();
        }
        PairMatrix<real> sV(std::move(V), p.sparsity), sG;
        sG.rows = sV.rows;
        sG.cols = sV.cols;
        sG.diagonal = G.diag();
        sG.values.set_size(sV.values.size());
        zip(sG.values, sG.rows, sG.cols, [&](auto &o, auto i, auto j) {o = G.at(i, j);});
        return Result::Pairs{.matrix=std::move(sV), .gradient=std::move(sG)};
    }
    Result::Costs operator()(Matrix<ValueGrad<T>> P, Job::Costs const &p) const {
        NUPACK_ERROR("Not implemented");
    }
    Result::PF operator()(True, ValueGrad<T> result, ValueGrad<T> raw) const {
        return {.logq=result.value, .raw_logq=raw.value, .gradient=result.gradient, .raw_gradient=raw.gradient};
    }
};

/******************************************************************************************/

}