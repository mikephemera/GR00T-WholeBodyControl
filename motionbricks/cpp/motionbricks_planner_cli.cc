#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "ctrl_env/motionbricks/planner.h"

namespace {

using namespace bfm::ctrl_env::motionbricks;

constexpr std::uint32_t kProtocolVersion = 1;
constexpr std::uint32_t kMaxRequestPayload = 1U << 20;
constexpr char kRequestMagic[8] = {'M', 'B', 'R', 'E', 'Q', '1', 0, 0};
constexpr char kResponseMagic[8] = {'M', 'B', 'R', 'E', 'S', '1', 0, 0};

struct PlannerRequest final {
    std::array<float, kContextFrames * kQposDim> qpos{};
    MotionBricksControl control;
};

struct Options final {
    std::string asset_pack;
    std::string model_root;
    int device_id{-1};
    int mode{0};
    int warmup{0};
    std::uint32_t seed{0};
    bool require_musa{false};
    bool serve{false};
};

void usage(std::ostream& output) {
    output << "motionbricks_planner_cli --asset-pack PACK [options]\n"
              "  --model-root DIR        directory containing the three ONNX models\n"
              "  --device NAME           cpu (default), musa, or a numeric MUSA device\n"
              "  --require-musa          reject CPU fallback\n"
              "  --warmup N              unmeasured warmup planner calls\n"
              "  --mode N --seed N       one-shot final-qpos smoke request\n"
              "  --serve                 persistent framed final-planner protocol on stdin/stdout\n";
}

template <typename T>
T read_scalar(std::istream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error("truncated MotionBricks planner request");
    return value;
}

template <typename T>
void write_scalar(std::ostream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!output) throw std::runtime_error("cannot write MotionBricks planner response");
}

void require_little_endian() {
    const std::uint16_t probe = 1;
    if (*reinterpret_cast<const std::uint8_t*>(&probe) != 1) {
        throw std::runtime_error("MotionBricks planner protocol requires little-endian host order");
    }
}

int parse_device(const std::string& value) {
    if (value == "cpu") return -1;
    if (value == "musa") return 0;
    const int device = std::stoi(value);
    if (device < 0) throw std::invalid_argument("numeric --device must be non-negative");
    return device;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto value = [&]() -> std::string {
            if (++index >= argc) throw std::invalid_argument("missing value for " + argument);
            return argv[index];
        };
        if (argument == "--help" || argument == "-h") {
            usage(std::cout);
            std::exit(0);
        } else if (argument == "--asset-pack") {
            options.asset_pack = value();
        } else if (argument == "--model-root") {
            options.model_root = value();
        } else if (argument == "--device") {
            options.device_id = parse_device(value());
        } else if (argument == "--mode") {
            options.mode = std::stoi(value());
        } else if (argument == "--seed") {
            const auto parsed = std::stoull(value());
            if (parsed > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--seed must fit uint32");
            }
            options.seed = static_cast<std::uint32_t>(parsed);
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(value());
        } else if (argument == "--require-musa") {
            options.require_musa = true;
        } else if (argument == "--serve") {
            options.serve = true;
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.asset_pack.empty()) throw std::invalid_argument("--asset-pack is required");
    if (options.warmup < 0 || options.warmup > 10000) {
        throw std::invalid_argument("--warmup must be in [0,10000]");
    }
    if (options.require_musa && options.device_id < 0) {
        throw std::invalid_argument("--require-musa requires a MUSA --device");
    }
    return options;
}

MotionBricksPlannerConfig make_config(const Options& options) {
    MotionBricksPlannerConfig config;
    config.asset_pack = options.asset_pack;
    config.device_id = options.device_id;
    config.require_musa = options.require_musa;
    if (!options.model_root.empty()) {
        const std::filesystem::path root(options.model_root);
        config.root_model = (root / "root_backbone.onnx").string();
        config.pose_model = (root / "pose_backbone.onnx").string();
        config.decoder_model = (root / "vqvae_decoder.onnx").string();
    }
    return config;
}

PlannerRequest parse_request_payload(std::istream& input) {
    PlannerRequest request;
    input.read(reinterpret_cast<char*>(request.qpos.data()), sizeof(request.qpos));
    request.control.mode = read_scalar<std::int32_t>(input);
    request.control.movement_direction = read_scalar<float>(input);
    request.control.facing_direction = read_scalar<float>(input);
    request.control.random_seed = read_scalar<std::uint32_t>(input);
    request.control.target_clip_seed = read_scalar<std::int32_t>(input);
    std::array<std::int64_t, 11> allowed_pred_num_tokens{};
    for (auto& token : allowed_pred_num_tokens) {
        token = read_scalar<std::int64_t>(input);
    }
    if (std::any_of(
            allowed_pred_num_tokens.begin(), allowed_pred_num_tokens.end(),
            [](std::int64_t value) { return value != 0; })) {
        request.control.allowed_pred_num_tokens = allowed_pred_num_tokens;
    }
    const std::uint32_t name_size = read_scalar<std::uint32_t>(input);
    if (name_size > 4096) throw std::runtime_error("MotionBricks target clip name is too long");
    request.control.target_clip_name.resize(name_size);
    input.read(request.control.target_clip_name.data(), name_size);
    if (!input) throw std::runtime_error("truncated MotionBricks target clip name");
    if (input.peek() != std::istream::traits_type::eof()) {
        throw std::runtime_error("unexpected trailing bytes in MotionBricks planner request");
    }
    return request;
}

void fill_default_context(MotionBricksPlanner& planner, PlannerRequest* request) {
    const bool empty = std::all_of(
        request->qpos.begin(), request->qpos.end(), [](float value) { return value == 0.0F; });
    if (!empty) return;
    const auto idle = planner.assets().read_float32("clips/idle/initial_qpos");
    if (idle.size() != kQposDim) throw std::runtime_error("invalid packaged idle qpos");
    for (int frame = 0; frame < kContextFrames; ++frame) {
        std::copy(idle.begin(), idle.end(), request->qpos.begin() + frame * kQposDim);
    }
}

void serialize_response(std::ostream& output, const MotionBricksPlanResult& result) {
    write_scalar<std::int32_t>(output, result.num_pred_frames);
    write_scalar<std::int32_t>(output, result.predicted_tokens);
    write_scalar<std::int32_t>(output, result.target_clip_index);
    write_scalar<double>(output, result.timings.total_ms);
    write_scalar<double>(output, result.timings.feature_ms);
    write_scalar<double>(output, result.timings.root_ms);
    write_scalar<double>(output, result.timings.pose_ms);
    write_scalar<double>(output, result.timings.decoder_ms);
    write_scalar<double>(output, result.timings.postprocess_ms);
    write_scalar<std::uint32_t>(output, static_cast<std::uint32_t>(result.qpos.size()));
    output.write(reinterpret_cast<const char*>(result.qpos.data()),
                 static_cast<std::streamsize>(result.qpos.size() * sizeof(float)));
    if (!output) throw std::runtime_error("cannot write MotionBricks qpos response");
}

int serve(MotionBricksPlanner& planner) {
    while (true) {
        char magic[8]{};
        std::cin.read(magic, sizeof(magic));
        if (std::cin.gcount() == 0 && std::cin.eof()) return 0;
        if (std::cin.gcount() != sizeof(magic)) throw std::runtime_error("truncated request magic");
        if (std::memcmp(magic, kRequestMagic, sizeof(magic)) != 0) {
            throw std::runtime_error("unsupported MotionBricks planner request magic");
        }
        if (read_scalar<std::uint32_t>(std::cin) != kProtocolVersion) {
            throw std::runtime_error("unsupported MotionBricks planner protocol version");
        }
        const std::uint32_t payload_size = read_scalar<std::uint32_t>(std::cin);
        if (payload_size > kMaxRequestPayload) throw std::runtime_error("planner request too large");
        std::string payload(payload_size, '\0');
        std::cin.read(payload.data(), payload.size());
        if (!std::cin) throw std::runtime_error("truncated MotionBricks planner payload");

        std::istringstream body(payload, std::ios::binary);
        PlannerRequest request = parse_request_payload(body);
        fill_default_context(planner, &request);
        const MotionBricksPlanResult result = planner.plan(request.qpos, request.control);

        std::ostringstream response(std::ios::binary);
        serialize_response(response, result);
        const std::string bytes = response.str();
        std::cout.write(kResponseMagic, sizeof(kResponseMagic));
        write_scalar<std::uint32_t>(std::cout, kProtocolVersion);
        write_scalar<std::uint32_t>(std::cout, static_cast<std::uint32_t>(bytes.size()));
        std::cout.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        std::cout.flush();
        if (!std::cout) throw std::runtime_error("cannot flush MotionBricks planner response");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require_little_endian();
        const Options options = parse_options(argc, argv);
        MotionBricksPlanner planner(make_config(options));
        planner.warmup(options.serve ? std::max(1, options.warmup) : options.warmup);
        if (options.serve) return serve(planner);

        PlannerRequest request;
        request.control.mode = options.mode;
        request.control.random_seed = options.seed;
        fill_default_context(planner, &request);
        const MotionBricksPlanResult result = planner.plan(request.qpos, request.control);
        std::cout << "num_pred_frames=" << result.num_pred_frames
                  << " predicted_tokens=" << result.predicted_tokens
                  << " target_clip_index=" << result.target_clip_index << '\n'
                  << std::fixed << std::setprecision(3)
                  << "latency_ms total=" << result.timings.total_ms
                  << " feature=" << result.timings.feature_ms
                  << " root=" << result.timings.root_ms
                  << " pose=" << result.timings.pose_ms
                  << " decoder=" << result.timings.decoder_ms
                  << " postprocess=" << result.timings.postprocess_ms << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "motionbricks_planner_cli failed: " << error.what() << '\n';
        return 1;
    }
}
