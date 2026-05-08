/**
 * @brief Defines physical constants, default random number generator
 *
 * @file Constants.h
 * @author Mark Fornace
 * @date 2018-05-31
 */
#pragma once
#include "../common/Config.h"
#include "../common/Error.h"
#include "../algorithms/Numeric.h"
#include "../reflect/Print.h"

namespace nupack {

/// Occasionally useful variable to put in some value without recompiling the whole project
extern std::string HackHelper;

/******************************************************************************************/

constexpr real const ZeroCinK = 273.15;
constexpr real const DefaultTemperature = ZeroCinK + 37.0;
constexpr real const LogOf10 = 2.302585092994046;
constexpr real const Pi = M_PI;

//constexpr real Kb = 0.001987204118; // This is the correct value
constexpr real const Kb = 0.00198717; // This agrees with NUPACK 3
constexpr real const DefaultBeta= 1 / (Kb * DefaultTemperature);

/******************************************************************************************/

/// Boltzmann factor from energy and beta
template <class B, class E>
auto boltzmann_factor(B const &beta, E const &energy) {return exp(-beta * energy);}
/// Energy from Boltzmann factor and beta
template <class B, class F>
auto inverse_boltzmann(B const &beta, F const &factor) {return -log(factor) / beta;}

/******************************************************************************************/

/// A constant DNA sequence, used for testing when a random one isn't desired
extern std::string ReferenceSequence;
/// Get a constant DNA sequence, used for testing when a random one isn't desired
std::string reference_dna(std::size_t length);

/// moles per liter of water from temperature in Kelvin
// In Kelvin, Tanaka M., Girard, G. et al, Metrologia, 2001 38, 301-309.
template <class Temperature>
auto water_molarity(Temperature T) {
    NUPACK_REQUIRE(T, >=, 273.15);
    NUPACK_REQUIRE(T, <=, 373.15);
    constexpr real const a1 = -3.983035 - ZeroCinK, a2 = 301.797 - ZeroCinK,
                         a3 = 522528.9, a4 = 69.34881 - ZeroCinK, a5 = 999.974950;
    return a5 * (1 - (T + a1) * (T + a1) * (T + a2) / a3 / (T + a4)) / 18.0152;
};

// Stabilization energy from salt concentration for each loop
// No correction for RNA since we don't have parameters
template <class Temperature, class Na, class Mg>
auto dna_salt_correction(Temperature const &t, Na const &na, Mg const &mg, bool long_helix=false) {
  // Ignore magnesium for long helix mode (not cited why, for consistency with Mfold)
  NUPACK_REQUIRE(na, >=, 0.05); NUPACK_REQUIRE(na, <=, 1.1);
  NUPACK_REQUIRE(mg, >=, 0);    NUPACK_REQUIRE(mg, <=, 0.2);
  if (long_helix) return -(0.2 + 0.175 * log(na)) * t / DefaultTemperature;
  return -0.114 * log(na + 3.3 * sqrt(mg)) * t / DefaultTemperature;
}

/******************************************************************************************/

/// Encode a string into base64
std::string encode64(std::string_view val);

/// Decode a string from base64
std::string decode64(std::string_view val);

}
