#pragma once
#include "Schedule.h"
#include "Action.h"
#include "Common.h"
#include "Cache.h"
#include "../model/Model.h"
#include "../standard/Map.h"
#include "../standard/Function.h"
#include "../types/PairList.h"
#include "../types/Sequence.h"
#include "../math/Sparse.h"

#include <atomic>

namespace nupack::thermo {

/******************************************************************************************/

/// Result type holding a secondary structure, its structure free energy, and its stacking state free energy
struct StructureEnergy {
    PairList structure;
    real energy;
    real stack_energy;

    NUPACK_REFLECT(StructureEnergy, structure, energy, stack_energy);
};

/// Result type holding all of the calculated results for each complex
struct Result {
    // Partition function result holding log partition function, and the same unadjusted for join penalty
    struct PF     {real logq, raw_logq, gradient, raw_gradient; NUPACK_REFLECT(PF, logq, raw_logq, gradient, raw_gradient);};
    // MFE result holding MFE, and the same unadjusted for join penalty
    struct MFE    {real energy, raw_energy; NUPACK_REFLECT(MFE, energy, raw_energy);};
    // Pair probability result
    struct Pairs  {PairMatrix<real> matrix, gradient; NUPACK_REFLECT(Pairs, matrix, gradient);};
    // Free energy cost matrix result
    struct Costs  {Mat<real> matrix, gradient; NUPACK_REFLECT(Costs, matrix, gradient);};
    // Boltzmann sampled structure results
    struct Sample {vec<PairList> structures; NUPACK_REFLECT(Sample, structures);};
    // Suboptimal structure results
    struct Subopt {vec<StructureEnergy> structures; NUPACK_REFLECT(Subopt, structures);};

    std::optional<PF>     pfunc;
    std::optional<MFE>    mfe;
    std::optional<Pairs>  pairs;
    std::optional<Subopt> subopt;
    std::optional<Sample> sample;
    std::optional<Costs>  costs;

    NUPACK_REFLECT(Result, pfunc, mfe, pairs, subopt, sample, costs);
};

/// One of the newly calculated quantities for a given complex
using Update = std::variant<Result::PF, Result::MFE, Result::Pairs, Result::Subopt, Result::Sample, Result::Costs>;

/******************************************************************************************/

/// Job specification for a single complex and single type of job
struct Job {
    /// Job specification for a partition function
    struct PF {
        NUPACK_REFLECT_EMPTY(PF);
    };
    /// Job specification for an MFE
    struct MFE {
        NUPACK_REFLECT_EMPTY(MFE);
    };
    /// Job specification for pair probabilities
    struct Pairs {
        Sparsity sparsity;
        NUPACK_REFLECT(Pairs, sparsity);
    };
    /// Job specification for suboptimal structures
    struct Subopt {
        real gap=0;
        std::size_t max_number=100000;
        NUPACK_REFLECT(Subopt, gap, max_number);
    };
    /// Job specification for free energy costs
    struct Costs {
        NUPACK_REFLECT_EMPTY(Costs);
    };
    /// Job specification for Boltzmann samples
    struct Sample {
        std::size_t number;
        vec<std::uint64_t> seeds; // The real seeds used for the RNG initialization within each task
        NUPACK_REFLECT(Sample, number, seeds);
    };

    using Kind = std::variant<PF, Pairs, Sample, MFE, Costs, Subopt>;
    Complex strands; // also need indices on each strand.
    Kind kind;

    NUPACK_REFLECT(Job, strands, kind);
};

/// Callback function to call when any complex result is calculated
using Sink = std::function<void(Complex const &, Update)>;

/******************************************************************************************/

// Undefined class that should be specialized for each model type to yield a compatible thermo computer generator
template <class U>
struct DefaultFactory;

/******************************************************************************************/

using ComputerList = vec<Computer>;

/// Factory holding the different implementations of the dynamic programming algorithm (generally in different data types)
struct Factory {
    ComputerList computers;

    Factory() = default;

    template <class Model>
    Factory(Model mod) : computers(DefaultFactory<Model>::create(std::move(mod))) {}
};

/******************************************************************************************/

// Specialization to make a factory of thermo computers using a Model<>
template <>
struct DefaultFactory<Model<>> {
    static ComputerList create(Model<>);
};

/******************************************************************************************/

using ResultMap = std::map<Complex, Result>;

/******************************************************************************************/

/// Future for a set of calculated complex quantities that have been submitted to a SharedExecutor object
struct Future {
    std::future<void> future;
    SharedExecutor executor;
    std::shared_ptr<tf::Taskflow> flow;
    std::shared_ptr<ResultMap> result;
    SharedError error;

    NUPACK_REFLECT(Future, future, executor, flow, result, error);

    /// Block and get the actual results
    ResultMap get();
};

/******************************************************************************************/

// Compute the results for a list of jobs, calling the "Sink" function when any quantity is calculated
void compute_callback(Sink put, vec<Job> jobs, Factory const &, ComputeOptions const &ops={});

// Compute the results for a list of jobs asynchronously and return the result as a future
Future submit(vec<Job> jobs, Factory const &, ComputeOptions const &ops={});

// Schedule a list of jobs in a given taskflow
tf::Task schedule(tf::Taskflow &flow, Sink put, vec<Job> jobs, Factory const &, ComputeOptions const &ops);

// Compute the results for a list of jobs synchronously
ResultMap computes(vec<Job> jobs, Factory const &, ComputeOptions const &ops={});

// Compute the result for a single job synchronously
Result compute(Job, Factory const &, ComputeOptions const &ops={});

// Specializations for common jobs

/// Calculate sparse pair probability matrix
std::pair<PairMatrix<real>, real> sparse_pair_probability(SequenceList seqs, Factory const &, ComputeOptions const &ops={});

/// Calculate dense pair probability matrix
std::pair<Mat<real>, real> pair_probability(SequenceList seqs, Factory const &, ComputeOptions const &ops={});

/// Calculate free energy cost matrix
std::pair<Mat<real>, real> mfe_cost(SequenceList seqs, Factory const &, ComputeOptions const &ops={});

/// Calculate Boltzmann sampled structures
std::pair<vec<PairList>, real> sample(std::size_t n, SequenceList seqs, Factory const &, ComputeOptions const &ops={}, std::size_t copies=1);

/// Calculate suboptimal structures
vec<StructureEnergy> subopt(real gap, SequenceList seqs, Factory const &, ComputeOptions const &ops={});
inline StructureEnergy mfe_structure(SequenceList seqs, Factory const &f, ComputeOptions const &ops={}) {
    auto v = subopt(-1, std::move(seqs), f, ops);
    NUPACK_REQUIRE(len(v), ==, 1);
    return std::move(v.at(0));
}

/// Calculate log partition function
real log_partition_function(SequenceList seqs, Factory const &, ComputeOptions const &ops={});

/// Calculate indistinguishable log partition function
real indistinguishable_logq(SequenceList seqs, Factory const &f, ComputeOptions const &ops={});

/// Calculate minimum free energy
real min_free_energy(SequenceList seqs, Factory const &, ComputeOptions const &ops={});

// Compute unpaired melt fraction for a list of complexes and strand mole fractions; assumes strand IDs have been set to count up from 0!
real fraction_bases_unpaired(vec<Complex> complexes, Factory const &factory, Col<real> const &mole_fractions, ComputeOptions const &ops={});

// Compute unpaired melt fractions for a list of distinguishable strands, temperatures, model generating function, strand concentrations, and max complex size
Col<real> melt_curve(vec<Sequence> strands, Col<real> const &concentrations, uint max_size, Col<real> const &temperatures, std::function<Factory(real)> factory, ComputeOptions const &ops={});

/******************************************************************************************/

}
