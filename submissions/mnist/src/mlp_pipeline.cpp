// Copyright (c) 2026 Crypto Lab Inc.
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

Levels buildLevels() {
    paramsUtils::LevelsBuilder lb;
    lb.setRing(LOG_DEGREE, POLY_TYPE);
    lb.initMod(BASE_BITS);
    return lb.buildAbove(NUM_MULTS, RESCALE_BITS);
}

EnDecoder makeEncoder(const Levels &levels) {
    const EncodeParams ecd_params(POLY_TYPE, LOG_DEGREE, levels, NTT_ALG,
                                  /*coeff_encoding=*/false);
    return EnDecoder(ecd_params);
}

MatrixVectorEvalParams makeMvParams(const LayerGeom &geom, const Levels &levels,
                                    u32 in_level, Real128 in_scale) {
    const PolyRing in_ring{levels.mods[in_level], LOG_DEGREE, NTT_ALG};
    const Encoding in_ecd{LOG_SLOTS, in_scale, /*dft=*/true};

    MatrixVectorEvalParams p;
    p.setSteps(bsIndices(geom), gsIndices(geom))
        .setInput(in_ring, in_ecd)
        .setLevels(levels)
        .setPolyType(POLY_TYPE);
    p.checkValidity();
    return p;
}

SwKeyGenParams makeSwkParams(const Levels &levels, u32 level) {
    paramsUtils::SwKeyGenParamsBuilder swk;
    swk.setNoiseDistribution(DiscreteGaussian(NOISE_STDDEV));
    swk.setRing(LOG_DEGREE, POLY_TYPE);
    swk.setModUpPrimes(swkMaxBits(), SWK_MARGIN);
    return swk.build(levels.mods[level], NTT_ALG == NTTAlgorithm::CYC_FOR_CI);
}

namespace {

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

LayerWeights padWeights(const LayerGeom &geom,
                        const std::vector<std::vector<double>> &dense,
                        const std::vector<double> &bias) {
    if (dense.size() != geom.n_out)
        throw std::runtime_error("weight rows != n_out");
    if (bias.size() != geom.n_out)
        throw std::runtime_error("bias length != n_out");
    if (geom.n_out > geom.p)
        throw std::runtime_error("n_out exceeds output period p");

    LayerWeights lw;
    lw.W.assign(static_cast<size_t>(geom.p) * geom.q, 0.0);
    lw.b = bias;
    for (u32 r = 0; r < dense.size(); ++r) {
        if (dense[r].size() > geom.q)
            throw std::runtime_error("weight cols exceed input period q");
        for (u32 c = 0; c < dense[r].size(); ++c)
            lw.W[static_cast<size_t>(r) * geom.q + c] = dense[r][c];
    }
    return lw;
}

Message packImages(const std::vector<std::vector<double>> &images,
                   size_t first, size_t count) {
    if (count > IMAGES_PER_CTXT)
        throw std::runtime_error("more images than fit in a ciphertext");
    Message in(LOG_SLOTS);
    for (u32 s = 0; s < SLOTS; ++s)
        in[s] = Complex(0.0);
    for (size_t i = 0; i < count; ++i) {
        const auto &img = images[first + i];
        if (img.size() != INPUT_DIM)
            throw std::runtime_error("packImages expects cropped INPUT_DIM rows");
        for (u32 c = 0; c < INPUT_DIM; ++c)
            in[slotOf(c, static_cast<u32>(i))] =
                Complex(static_cast<Real>(img[c]));
    }
    return in;
}

std::map<i32, Message> buildDiags(const LayerGeom &geom,
                                  const std::vector<double> &W) {
    if (W.size() != static_cast<size_t>(geom.p) * geom.q)
        throw std::runtime_error("weight block is not p x q");

    std::map<i32, Message> diags;
    for (u32 k = 0; k < geom.p; ++k) {
        Message diag(LOG_SLOTS);
        for (u32 s = 0; s < SLOTS; ++s) {
            const u32 c = s / IMAGES_PER_CTXT;
            diag[s] = Complex(static_cast<Real>(
                W[(c % geom.p) * geom.q + (c + k) % geom.q]));
        }
        diags[static_cast<i32>(IMAGES_PER_CTXT * k)] = std::move(diag);
    }
    return diags;
}

Message buildBiasMessage(const LayerGeom &geom, const std::vector<double> &b) {
    Message msg(LOG_SLOTS);
    for (u32 s = 0; s < SLOTS; ++s) {
        const u32 r = (s / IMAGES_PER_CTXT) % geom.p;
        msg[s] = (r < geom.n_out) ? Complex(static_cast<Real>(b[r]))
                                  : Complex(0.0);
    }
    return msg;
}

MatrixVectorEvalEncoded encodeDiags(const LayerGeom &geom,
                                    const std::vector<double> &W,
                                    const Levels &levels, u32 in_level,
                                    Device dev) {
    const auto mv_params =
        makeMvParams(geom, levels, in_level, levels.scales[in_level]);
    const auto gadget = makeSwkParams(levels, in_level).getGadgetDecomp();
    const auto diags = buildDiags(geom, W);
    return MatrixVectorEvalEncoded(mv_params, gadget, diags, dev);
}

Layer makeLayer(const LayerGeom &geom, const std::vector<double> &bias,
                const MatrixVectorEvalEncoded &encoded, const Levels &levels,
                u32 in_level, u32 out_level, const EnDecoder &encoder,
                std::unique_ptr<RotKeyPtrs> rot_keys, KeyPtr relin_key,
                Device dev, RotKeyPtrs fold_keys) {
    if (out_level >= in_level)
        throw std::runtime_error("layer must consume at least one level");

    Layer lyr;
    lyr.activate = geom.activate;
    lyr.mod_to = levels.mods[out_level];
    lyr.scale_to = levels.scales[out_level];

    const auto mv_params =
        makeMvParams(geom, levels, in_level, levels.scales[in_level]);

    lyr.rot_keys = std::move(rot_keys);
    lyr.matvec = std::make_unique<MatrixVectorEval>(mv_params, *lyr.rot_keys,
                                                    encoded);
    lyr.fold_stride = static_cast<i32>(IMAGES_PER_CTXT * geom.p);
    lyr.fold_factor = geom.q / geom.p;
    if (lyr.fold_factor > 1) {
        for (u32 j = 1; j < lyr.fold_factor; ++j)
            lyr.fold_steps.push_back(lyr.fold_stride * static_cast<i32>(j));
        lyr.fold_keys = std::move(fold_keys);
        if (lyr.fold_keys.empty())
            throw std::runtime_error(
                "layer folds " + std::to_string(lyr.fold_factor) +
                ":1 but no fold keys were supplied");
    }

    lyr.relin_key = std::move(relin_key);
    if (geom.activate && !lyr.relin_key)
        throw std::runtime_error("activating layer needs a relinearization key");

    auto bias_msg = buildBiasMessage(geom, bias);
    bias_msg.to(dev);
    lyr.bias = IPlaintext::make(PtxtType::NORMAL);
    encoder.encode(bias_msg, *lyr.bias, lyr.mod_to, lyr.scale_to);

    return lyr;
}

void homLayer(Ptr<ICiphertext> &ct, const Layer &lyr, const HomEval &eval,
              const HomEvalFlexible &flex) {
    auto res = ICiphertext::make(EncType::RLWE);
    lyr.matvec->eval(*ct, *res);

    flex.adjust(*res, *res, lyr.mod_to, lyr.scale_to);

    if (!lyr.fold_steps.empty()) {
        std::vector<Ptr<ICiphertext>> rots;
        std::vector<ICiphertext *> rot_ptrs;
        rots.reserve(lyr.fold_steps.size());
        rot_ptrs.reserve(lyr.fold_steps.size());
        for (size_t i = 0; i < lyr.fold_steps.size(); ++i) {
            rots.push_back(ICiphertext::make(EncType::RLWE));
            rot_ptrs.push_back(&*rots.back());
        }
        eval.rot(*res, lyr.fold_steps, rot_ptrs, lyr.fold_keys);
        for (auto *r : rot_ptrs)
            eval.add(*res, *r, *res);
    }

    eval.add(*res, *lyr.bias, *res);

    if (lyr.activate) {
        auto sq = ICiphertext::make(EncType::RLWE);
        eval.tensor(*res, *res, *sq);
        eval.relin(*sq, *lyr.relin_key);
        auto rescaled = ICiphertext::make(EncType::RLWE);
        eval.rescale(*sq, *rescaled);
        ct = std::move(rescaled);
    } else {
        ct = std::move(res);
    }
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
