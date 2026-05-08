#include <nupack/thermo/Block.h>
#include <nupack/thermo/Action.h>
#include <nupack/thermo/Engine.h>
#include <nupack/thermo/Compute.h>
#include <nupack/thermo/Constants.h>

#include <nupack/model/Model.h>
#include <nupack/types/Matrix.h>
#include <nupack/concentration/Equilibrate.h>
#include <spdlog/spdlog.h>

namespace nupack::thermo {

/******************************************************************************************/

real MinimumChunkSize = 100.0;

uint diagonal_chunk_size(uint n_tasks, uint diagonal) noexcept {
    if (n_tasks == 0) return 1;
    uint const n_chunks = max(1, std::round((diagonal+1) * n_tasks / MinimumChunkSize));
    return std::ceil(real(n_tasks) / n_chunks);
}   

/******************************************************************************************/

vec<std::pair<real, uint>> compute_weights(mark_t const &marks, real value, DefaultRNG &rng) {
    auto w = vmap(marks, [](auto m) {return std::make_pair(real(), m);});
    random_shuffle(w, rng);
    zip(~range(1, len(w)+1), ::nupack::reversed(w), [&](real n, auto &x) {
        // somehow mt19937 sometimes gives 0 or 1 randomly, for example with seed 4154861010 for me
        real random;
        do {random = random_float(rng);} while (random == 0 || random == 1);
        real const ratio = std::pow(random, real(1) / n);
        x.first = (value *= ratio);
    });
    NUPACK_QUICK_ASSERT(std::is_sorted(w.begin(), w.end()), w, value);
    return w;
}

/******************************************************************************************/

Result::Pairs Finalizer<real>::operator()(Matrix<real> P0, Job::Pairs const &p) const {
    Mat<real> const Pm(P0.begin(), P0.shape()[0], P0.shape()[1], false, false);
    Mat<real> P(Pm.n_rows, Pm.n_cols, la::fill::none);
    if (auto const logq = Pm(0, 0); std::isfinite(logq)) {
        P = la::exp(Pm + Pm.t() - logq);
        P.diag().zeros();
        P.diag() = 1 - la::sum(P, 0);
        NUPACK_QUICK_ASSERT(P.is_finite(), P, logq);
    } else {
        P.zeros();
        P.diag().ones();
    }
    return Result::Pairs{.matrix=PairMatrix<real>(std::move(P), p.sparsity)};
}

Result::Costs Finalizer<real>::operator()(Matrix<real> P0, Job::Costs const &p) const {
    Mat<real> const Pm(P0.begin(), P0.shape()[0], P0.shape()[1], false, false);
    auto const mfe = Pm(0, 0);
    Mat<real> P = Pm + Pm.t() - mfe;
    P.diag().fill(inf<real>());
    P.diag() = la::min(P, 0);
    return Result::Costs{.matrix=std::move(P)};
}

/******************************************************************************************/

// TODO need to figure recursion names out later...

static constexpr std::array<std::string_view, 33> RecursionNames = {
    "X", "MM", "B", "Z", "D", "YA", "YB", "MD", "MC", "MCS", "MS", "CD", "CD00", "CD01", "CD10", "CD11", "S", "S00", "S01", "S10", "S11", "M", "Q", "Q00", "Q01", "Q10", "Q11", "N", "N00", "N01", "N10", "N11"
};

std::string_view recursion_name(Priority p) {
    if (p < RecursionNames.size()) return RecursionNames[p];
    return "_";
}

/******************************************************************************************/

void augment_result(Ignore, Ignore) {}

void augment_result(Result::Sample &a, Result::Sample &&b) {
    a.structures.insert(a.structures.end(), std::make_move_iterator(b.structures.begin()), std::make_move_iterator(b.structures.end()));
}

void augment_result(Result::Subopt &a, Result::Subopt &&b) {
    if (len(a.structures) < len(b.structures)) a = std::move(b);
}

/******************************************************************************************/

template <class T, NUPACK_IF(is_same<T, Job::PF, Job::MFE>)>
void call_extension(T const &j, CallbackList &callbacks, Complex x, Sink const &sink, Computer const &computer) {
    computer.extensions.at(0)(callbacks, Item::from(std::move(x)), [sink](Item res) mutable {
        auto p = res.take<std::pair<Complex, if_t<is_same<T, Job::PF>, Result::PF, Result::MFE>>>();
        sink(p.first, p.second);
    });
}

static constexpr real SuboptMinimum = 1e-6;

template <class T, NUPACK_IF(is_same<T, Job::Subopt, Job::Sample>)>
void call_extension(T j, CallbackList &callbacks, Complex x, Sink const &sink, Computer const &computer) {
    call_extension(if_t<is_same<T, Job::Sample>, Job::PF, Job::MFE>(), callbacks, x, sink, computer);
    if constexpr(std::is_same_v<T, Job::Subopt>) if (j.gap >= 0) max_eq(j.gap, SuboptMinimum);
    computer.extensions.at(2)(callbacks, Item::from(std::make_pair(std::move(j), std::move(x))), [sink](Item res) mutable {
        auto p = res.take<std::pair<Complex, if_t<is_same<T, Job::Sample>, Result::Sample, Result::Subopt>>>();
        sink(p.first, p.second);
    });
}

template <class T, NUPACK_IF(is_same<T, Job::Pairs, Job::Costs>)>
void call_extension(T j, CallbackList &callbacks, Complex x, Sink const &sink, Computer const &computer) {
    call_extension(if_t<is_same<T, Job::Pairs>, Job::PF, Job::MFE>(), callbacks, x, sink, computer);
    computer.extensions.at(1)(callbacks, Item::from(std::make_pair(std::move(j), std::move(x))), [sink](Item res) mutable {
        auto p = res.take<std::pair<Complex, if_t<is_same<T, Job::Pairs>, Result::Pairs, Result::Costs>>>();
        sink(p.first, p.second);
    });
}

/******************************************************************************************/

template <class V>
auto schedule_each(tf::Taskflow &flow, Sink const &sink, Computer const &computer, ComputeOptions const &ops, V const &jobs) {
    CallbackList callbacks;
    for (auto &j : jobs)
        std::visit([&](auto const &kind) {call_extension(kind, callbacks, j.strands, sink, computer);}, j.kind);
    return schedule_all(flow, computer, ops, std::move(callbacks));
}

/******************************************************************************************/

tf::Task schedule(tf::Taskflow &flow, Sink sink, vec<Job> jobs, Factory const &factory, ComputeOptions const &ops) {
    NUPACK_QUICK_REQUIRE(len(factory.computers), ==, 2, "computers is assumed to be {PF, MFE} right now");
    auto const it = std::partition(jobs.begin(), jobs.end(), [](auto const &j) {return j.kind.index() < 3;});

    auto tasks = vmap(indices(factory.computers), [&](std::size_t i) {
        return schedule_each(flow, sink, factory.computers[i], ops, i == 0 ? view(jobs.begin(), it) : view(it, jobs.end()));
    });
    std::size_t needed = tasks[0].max_bytes + tasks[1].max_bytes;
    flow.emplace([cache=factory.computers[0].cache, a=ops.max_bytes > needed ? ops.max_bytes - needed : 0] {cache->shrink_to_limit(a);})
        .precede(tasks[0].start, tasks[1].start);
    if (!tasks[0].max_bytes || !tasks[1].max_bytes || needed <= ops.max_bytes) { // concurrent
        return flow.emplace([]{}).succeed(tasks[0].finish, tasks[1].finish);
    } else { // one after the other
        std::size_t allowed = max(ops.max_bytes, tasks[1].max_bytes) - tasks[1].max_bytes;
        auto shrink = flow.emplace([cache=factory.computers[0].cache, allowed] {cache->shrink_to_limit(allowed);});
        tasks[0].finish.precede(shrink);
        shrink.precede(tasks[1].start);
        return tasks[1].finish;
    }
}

/******************************************************************************************/

ResultMap Future::get() {
    NUPACK_ASSERT(result, "Future::get(): result was already retrieved");
    future.get();
    // flow->dump(std::cout);
    error.rethrow_if_set();
    auto r = std::move(*result);
    result.reset();
    flow.reset();
    return r;
}

/******************************************************************************************/

void compute_callback(Sink sink, vec<Job> jobs, Factory const &factory, ComputeOptions const &ops) {
    tf::Taskflow flow;
    schedule(flow, std::move(sink), std::move(jobs), factory, ops);

    auto exec = ops.executor;
    if (!exec) exec = SharedExecutor(1);
    exec.run(flow).wait();
    ops.error.rethrow_if_set();
}

/******************************************************************************************/

Future submit(vec<Job> jobs, Factory const &factory, ComputeOptions const &ops) {
    auto map = std::make_shared<ResultMap>();
    auto flow = std::make_shared<tf::Taskflow>();

    schedule(*flow, [map, mut=std::make_shared<std::mutex>()](Complex const &s, Update u) {
        std::lock_guard lock(*mut); // need to lock to modify the map safely
        auto &v = map->try_emplace(s).first->second;
        fork(u, [&](auto &u) {
            for_each(members_of(v), [&](auto &x) {
                if constexpr(is_like<decltype(*x), decltype(u)>) {
                    if (!x) x.emplace(std::move(u));
                    else augment_result(*x, std::move(u));
                }
            });
        });
    }, std::move(jobs), factory, ops);

    auto exec = ops.executor;
    if (!exec) exec = SharedExecutor(1);

    return {
        .future=exec.run(*flow),
        .executor=exec,
        .flow=flow,
        .result=std::move(map),
        .error=ops.error
    };
}

/******************************************************************************************/

ResultMap computes(vec<Job> jobs, Factory const &factory, ComputeOptions const &ops) {
    return submit(std::move(jobs), factory, ops).get();
}

Result compute(Job job, Factory const &factory, ComputeOptions const &ops) {
    return computes({job}, factory, ops).at(job.strands);
}

/******************************************************************************************/

std::pair<PairMatrix<real>, real> sparse_pair_probability(SequenceList seqs, Factory const &factory, ComputeOptions const &ops) {
    auto r = compute(Job{std::move(seqs), Job::Pairs()}, factory, ops);
    return {std::move(r.pairs.value().matrix), r.pfunc.value().logq};
}

std::pair<Mat<real>, real> pair_probability(SequenceList seqs, Factory const &factory, ComputeOptions const &ops) {
    auto r = compute(Job{std::move(seqs), Job::Pairs()}, factory, ops);
    return {r.pairs.value().matrix.full(), r.pfunc.value().logq};
}

std::pair<Mat<real>, real> mfe_cost(SequenceList seqs, Factory const &factory, ComputeOptions const &ops) {
    auto r = compute(Job{std::move(seqs), Job::Costs()}, factory, ops);
    return {std::move(r.costs.value().matrix), r.mfe.value().energy};
}

real log_partition_function(SequenceList seqs, Factory const &factory, ComputeOptions const &ops) {
    return compute(Job{std::move(seqs), Job::PF()}, factory, ops).pfunc.value().logq;
}

real indistinguishable_logq(SequenceList seqs, Factory const &f, ComputeOptions const &ops) {
    real correction = std::log(rotational_symmetry(seqs));
    return log_partition_function(std::move(seqs), f, ops) - correction;
}

real min_free_energy(SequenceList seqs, Factory const &factory, ComputeOptions const &ops) {
    auto res = compute(Job{std::move(seqs), Job::MFE()}, factory, ops);
    return res.mfe.value().energy;
}

std::pair<vec<PairList>, real> sample(std::size_t n, SequenceList seqs, Factory const &factory, ComputeOptions const &ops, std::size_t copies) {
    auto res = compute(Job{std::move(seqs), Job::Sample{.number=n, .seeds=vmap(range(copies), [](Ignore) -> std::uint64_t {return StaticRNG();})}}, factory, ops);
    return {std::move(res.sample.value().structures), res.pfunc.value().logq};
}

vec<StructureEnergy> subopt(real gap, SequenceList seqs, Factory const &factory, ComputeOptions const &ops) {
    constexpr std::size_t max = 100000000;
    auto res = compute(Job{std::move(seqs), Job::Subopt{.gap=gap, .max_number=max}}, factory, ops);
    return res.subopt.value().structures;
}

/******************************************************************************************/

real fraction_bases_unpaired(vec<Complex> complexes, Factory const &factory, Col<real> const &mole_fractions, ComputeOptions const &ops) {
    auto const map = computes(vmap(complexes, [](auto &x) {return Job{std::move(x), Job::Pairs{{.diagonal_only=true}}};}), factory, ops);
    auto const logq = vmap<Col<real>>(map, [](auto const &r) {return r.second.pfunc->logq;});

    Col<real> lengths(1 + maximum(map, [](auto const &p) {return maximum(p.first, [](auto s) {return s.id;});}));
    for (auto const &[x, r] : map) for (auto const &s : x) lengths(s.id) = len(s);

    la::Mat<real> A(len(map), len(lengths), la::fill::zeros);
    izip(map, [&A](auto i, auto const &p) {for (auto const &s : p.first) A(i, s.id) += 1;});

    auto const output = concentration::equilibrate(A, la::log(mole_fractions), logq);

    real out = 0;
    zip(map, output.solution, [&](auto const &r, auto c) {out += c * la::accu(r.second.pairs->matrix.diagonal);});
    out /= la::dot(mole_fractions, lengths);
    return out;
}

Col<real> melt_curve(vec<Sequence> strands, Col<real> const &concentrations, uint max_size, Col<real> const &temperatures, std::function<Factory(real)> factory, ComputeOptions const &ops) {
    izip(strands, [](auto i, auto &s) {s.id = i;});
    vec<Complex> complexes;
    for (auto s : range(max_size)) compute_necklaces(vec<int>(s+1, 0), len(strands), [&](auto const &x) {
        complexes.emplace_back(vmap(x, [&](auto i) {return strands[i];}));
    });
    return vmap<Col<real>>(temperatures, [&](auto T) {
        return fraction_bases_unpaired(complexes, factory(T), concentrations / water_molarity(T), ops);
    });
}

/******************************************************************************************/

}
