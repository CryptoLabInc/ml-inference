// Copyright (c) 2026 CryptoLab, Inc.
//
// This software is licensed under the terms of the Apache v2 License.
// See the LICENSE.md file for details.
//============================================================================

#include "mlp_pipeline.hpp"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>

using namespace heaan;

namespace mlp {

namespace {

// Used to parse CSV files
std::vector<double> parseNumbers(const std::string &text) {
    std::vector<double> v;
    std::string cur;
    auto flush = [&] {
        if (!cur.empty()) {
            v.push_back(std::stod(cur));
            cur.clear();
        }
    };
    for (char ch : text) {
        if (ch == ',' || ch == '\n' || ch == '\r' || ch == ' ' || ch == '\t')
            flush();
        else
            cur.push_back(ch);
    }
    flush();
    return v;
}

std::string slurp(const std::string &path) {
    std::ifstream f(path);
    if (!f.good())
        throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace

std::vector<double> readFlatCsv(const std::string &path) {
    return parseNumbers(slurp(path));
}

std::vector<std::vector<double>> readMatrixCsv(const std::string &path) {
    std::ifstream f(path);
    if (!f.good())
        throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        auto row = parseNumbers(line);
        if (!row.empty())
            rows.push_back(std::move(row));
    }
    if (rows.empty())
        throw std::runtime_error("empty matrix csv: " + path);
    for (const auto &r : rows)
        if (r.size() != rows.front().size())
            throw std::runtime_error("ragged matrix csv: " + path);
    return rows;
}

namespace {
std::string markerPath() {
    return std::string(MODEL_CACHE_DIR) + "/instance.txt";
}
} // namespace

void writeInstanceMarker(InstanceSize size) {
    fs::create_directories(MODEL_CACHE_DIR);
    std::ofstream f(markerPath());
    if (!f.good())
        throw std::runtime_error("cannot write " + markerPath());
    f << static_cast<int>(size) << "\n";
    if (!f)
        throw std::runtime_error("short write to " + markerPath());
}

std::optional<InstanceSize> readInstanceMarker() {
    std::ifstream f(markerPath());
    int v = -1;
    if (!f.good() || !(f >> v) || v < 0 || v > static_cast<int>(InstanceSize::LARGE))
        return std::nullopt;
    return static_cast<InstanceSize>(v);
}

std::vector<std::vector<double>> readSamples(const std::string &path, u32 dim) {
    std::ifstream f(path);
    if (!f.good())
        throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos)
            continue;
        std::istringstream iss(line);
        std::vector<double> row;
        row.reserve(dim);
        double v = 0.0;
        while (iss >> v)
            row.push_back(v);
        if (row.size() != dim)
            throw std::runtime_error(
                "expected " + std::to_string(dim) + " values per line in " +
                path + ", got " + std::to_string(row.size()));
        rows.push_back(std::move(row));
    }
    if (rows.empty())
        throw std::runtime_error("no data found in " + path);
    return rows;
}

void writeSamples(const std::vector<std::vector<double>> &rows,
                  const std::string &path) {
    std::ofstream f(path);
    if (!f.good())
        throw std::runtime_error("cannot write " + path);
    char buf[32];
    for (const auto &row : rows) {
        for (size_t i = 0; i < row.size(); ++i) {
            std::snprintf(buf, sizeof(buf), "%.9g", row[i]);
            f << buf;
            if (i + 1 < row.size())
                f << ' ';
        }
        f << '\n';
    }
    if (!f)
        throw std::runtime_error("short write to " + path);
}

InstanceSize parseInstanceSize(int argc, char *argv[]) {
    if (argc < 2 || !std::isdigit(static_cast<unsigned char>(argv[1][0])))
        throw std::runtime_error(
            std::string("usage: ") + argv[0] +
            " <instance-size>   (0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE)");
    const int v = std::stoi(argv[1]);
    if (v < 0 || v > int(InstanceSize::LARGE))
        throw std::runtime_error("instance size out of range: " +
                                 std::to_string(v));
    return static_cast<InstanceSize>(v);
}

Device targetDevice() {
#ifdef MLP_WITH_CUDA
    return Device::GPU_CUDA;
#else
    return Device::CPU;
#endif
}

} // namespace mlp
