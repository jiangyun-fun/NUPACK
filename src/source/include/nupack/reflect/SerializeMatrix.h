#pragma once
#include "Serialize.h"
#include "../types/Matrix.h"

namespace nlohmann {

/******************************************************************************************/

template <class T>
struct adl_serializer<arma::Col<T>> {
    static void to_json(json &j, arma::Col<T> const &t) {
        j = nlohmann::json::array_t(t.begin(), t.end());
    }
    static void from_json(json const &j, arma::Col<T> &t) {
        t.set_size(j.size());
        std::copy(j.begin(), j.end(), t.begin());
    }
};

/******************************************************************************************/

template <class T>
struct adl_serializer<arma::Mat<T>> {
    static void to_json(json &j, arma::Mat<T> const &t) {
        j["shape"] = {t.n_rows, t.n_cols};
        j["data"] = nlohmann::json::array_t(t.begin(), t.end());
    }
    static void from_json(json const &j, arma::Mat<T> &t) {
        if (j.is_object()) {
            auto shape = j.at("shape").get<std::array<std::size_t, 2>>();
            t.set_size(shape[0], shape[1]);
            auto const &data = j.at("data");
            std::copy(data.begin(), data.end(), t.begin());
        } else if (j.is_array()) {
            if (!j.empty()) {
                t = arma::Mat<T>(j.size(), j[0].size(), arma::fill::none);
                arma::uword r = 0;
                for (auto const &x : j) {
                    NUPACK_REQUIRE(x.size(), ==, t.n_cols);
                    std::copy(x.begin(), x.end(), t.begin_row(r++));
                }
            } else t = arma::Mat<T>();
        } else NUPACK_ERROR("Matrix could not be loaded", j);
    }
};

/******************************************************************************************/

template <class T>
struct adl_serializer<arma::Cube<T>> {
    static void to_json(json &j, arma::Cube<T> const &t) {
        j["shape"] = {t.n_rows, t.n_cols, t.n_slices};
        j["data"] = nlohmann::json::array_t(t.begin(), t.end());
    }
    static void from_json(json const &j, arma::Cube<T> &t) {
        auto shape = j.at("shape").get<std::array<std::size_t, 3>>();
        t.set_size(shape[0], shape[1], shape[2]);
        auto const &data = j.at("data");
        std::copy(data.begin(), data.end(), t.begin());
    }
};

/******************************************************************************************/

template <class T>
struct adl_serializer<arma::SpMat<T>> {
    static void to_json(json &j, arma::SpMat<T> const &t) {
        t.sync();
        j["shape"] = nupack::la::shape(t);
        j["values"] = std::vector<T>(t.values, t.values + t.n_nonzero + 1);
        j["row_indices"] = std::vector<arma::uword>(t.row_indices, t.row_indices + t.n_nonzero + 1);
        j["col_ptrs"] = std::vector<arma::uword>(t.col_ptrs, t.col_ptrs + t.n_cols + 1);
    }
    static void from_json(json const &j, arma::SpMat<T> &t) {
        arma::Col<T> values;
        arma::uvec rows, cols;
        std::array<arma::uword, 2> shape;
        j.at("shape").get_to(shape);
        j.at("values").get_to(values);
        if (auto r = j.find("row_indices"); r != j.end()) {
            r->get_to(rows);
            j.at("col_ptrs").get_to(cols);
            t = arma::SpMat<T>(std::move(rows), std::move(cols), std::move(values), shape[0], shape[1]);
        } else {
            j.at("rows").get_to(rows);
            j.at("cols").get_to(cols);
            t = arma::SpMat<T>(arma::join_cols(rows.t(), cols.t()), std::move(values), shape[0], shape[1]);
        }
    }
};

/******************************************************************************************/

}

namespace nupack {

/******************************************************************************************/

template <class T>
json to_json_array(arma::Col<T> const &m) {return m;}

template <class T>
json to_json_array(arma::Mat<T> const &m) {
    return vmap(range(m.n_rows), [&](auto i) {
        return vmap(range(m.n_cols), [&](auto j) {
            return m.at(i, j);
        });
    });
}

template <class T>
json to_json_array(arma::Cube<T> const &m) {
    return vmap(range(m.n_slices), [&](auto s) {
        return vmap(range(m.n_rows), [&](auto i) {
            return vmap(range(m.n_cols), [&](auto j) {
                return m.at(i, j, s);
            });
        });
    });
}

/******************************************************************************************/

}