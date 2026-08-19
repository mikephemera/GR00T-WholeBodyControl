#include "motionbricks_protocol.h"

#include <onnxruntime_cxx_api.h>
#include <musa_provider_options.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using motionbricks::protocol::FrameHeader;
using motionbricks::protocol::Operation;

constexpr std::size_t kRootInputBytes = 1 * 8 * 5 * 4 + 1 * 8 + 1 * 8 * 4 * 4 + 1 * 8 +
                                         1 * 8 * 304 * 4 + 1 * 8 + 1 * 8 + 11 * 8;
constexpr std::size_t kPoseInputBytes = 16 * 8 * 8 + 64 * 4 * 4 + 64 * 304 * 4 + 64 + 8;
constexpr std::size_t kDecoderInputBytes = 16 * 8 * 8 + 64 * 2 * 4 + 64 * 304 * 4 + 64 + 16;

struct Options { std::filesystem::path dir; int device_id{-1}; bool use_musa_graph{false}; };

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--onnx-dir") {
            if (++i >= argc) throw std::invalid_argument("--onnx-dir needs a value");
            options.dir = argv[i];
        } else if (arg == "--device-id") {
            if (++i >= argc) throw std::invalid_argument("--device-id needs a value");
            options.device_id = std::stoi(argv[i]);
        } else if (arg == "--use-musa-graph") {
            options.use_musa_graph = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: motionbricks_ort_cli --onnx-dir DIR [--device-id -1]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.dir.empty()) throw std::invalid_argument("--onnx-dir is required");
    return options;
}

std::size_t element_count(const std::vector<int64_t>& shape) {
    std::size_t count = 1;
    for (const auto dim : shape) {
        if (dim <= 0 || count > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dim)) {
            throw std::runtime_error("ONNX graph has a non-static or invalid tensor shape");
        }
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

std::string shape_string(const std::vector<int64_t>& shape) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) out << (i ? "," : "") << shape[i];
    return out.str() + "]";
}

class Session {
public:
    Session(const std::filesystem::path& path, int device_id, bool use_musa_graph)
        : session_(env_, path.c_str(), make_options(device_id, use_musa_graph)) {
        Ort::AllocatorWithDefaultOptions allocator;
        for (std::size_t i = 0; i < session_.GetInputCount(); ++i) {
            auto name = session_.GetInputNameAllocated(i, allocator);
            input_names_.emplace_back(name.get());
            auto type_info = session_.GetInputTypeInfo(i);
            auto info = type_info.GetTensorTypeAndShapeInfo();
            const auto rank = info.GetDimensionsCount();
            if (rank > 16) throw std::runtime_error("invalid input rank");
            input_shapes_.push_back(info.GetShape());
            input_types_.push_back(info.GetElementType());
        }
        for (std::size_t i = 0; i < session_.GetOutputCount(); ++i) {
            auto name = session_.GetOutputNameAllocated(i, allocator);
            output_names_.emplace_back(name.get());
            auto type_info = session_.GetOutputTypeInfo(i);
            auto info = type_info.GetTensorTypeAndShapeInfo();
            const auto rank = info.GetDimensionsCount();
            if (rank > 16) throw std::runtime_error("invalid output rank");
            output_shapes_.push_back(info.GetShape());
            output_types_.push_back(info.GetElementType());
        }
    }

    const std::vector<std::string>& inputs() const { return input_names_; }
    const std::vector<std::string>& outputs() const { return output_names_; }
    const std::vector<int64_t>& shape(std::size_t i) const { return input_shapes_.at(i); }
    ONNXTensorElementDataType type(std::size_t i) const { return input_types_.at(i); }
    const std::vector<int64_t>& output_shape(std::size_t i) const { return output_shapes_.at(i); }
    ONNXTensorElementDataType output_type(std::size_t i) const { return output_types_.at(i); }

    std::vector<Ort::Value> run(const std::vector<const void*>& data) {
        if (data.size() != input_names_.size()) throw std::runtime_error("input count mismatch");
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> values;
        values.reserve(data.size());
        for (std::size_t i = 0; i < data.size(); ++i) {
            const auto count = element_count(input_shapes_[i]);
            const auto dtype = input_types_[i];
            if (dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
                dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
                throw std::runtime_error("unsupported ONNX input dtype");
            }
            values.emplace_back(Ort::Value::CreateTensor(
                memory, const_cast<void*>(data[i]), count * (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? 8 : (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? 4 : 1)),
                input_shapes_[i].data(), input_shapes_[i].size(), dtype));
        }
        std::vector<const char*> input_names, output_names;
        for (const auto& n : input_names_) input_names.push_back(n.c_str());
        for (const auto& n : output_names_) output_names.push_back(n.c_str());
        return session_.Run(Ort::RunOptions{nullptr}, input_names.data(), values.data(), values.size(), output_names.data(), output_names.size());
    }

private:
    static Ort::SessionOptions make_options(int device_id, bool use_musa_graph) {
        Ort::SessionOptions options;
        if (device_id >= 0) {
            OrtMUSAProviderOptions musa{};
            musa.device_id = device_id;
            musa.enable_musa_graph = static_cast<int>(use_musa_graph);
            options.AppendExecutionProvider_MUSA(musa);
        }
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        return options;
    }

    Ort::Env env_{ORT_LOGGING_LEVEL_ERROR, "motionbricks_cpp"};
    Ort::Session session_;
    std::vector<std::string> input_names_, output_names_;
    std::vector<std::vector<int64_t>> input_shapes_, output_shapes_;
    std::vector<ONNXTensorElementDataType> input_types_, output_types_;
};

template <typename T>
void take(const std::vector<std::uint8_t>& bytes, std::size_t& offset, T* output, std::size_t count) {
    const std::size_t size = sizeof(T) * count;
    if (offset > bytes.size() || size > bytes.size() - offset) throw std::runtime_error("truncated request payload");
    std::memcpy(output, bytes.data() + offset, size);
    offset += size;
}

template <typename T>
void append(std::vector<std::uint8_t>& bytes, const T* data, std::size_t count) {
    const auto* begin = reinterpret_cast<const std::uint8_t*>(data);
    bytes.insert(bytes.end(), begin, begin + sizeof(T) * count);
}

bool read_exact(void* data, std::size_t size) {
    std::cin.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    return std::cin.gcount() == static_cast<std::streamsize>(size);
}

void write_response(std::uint16_t operation, std::uint32_t status, const std::vector<std::uint8_t>& payload) {
    FrameHeader response{motionbricks::protocol::kResponseMagic, motionbricks::protocol::kVersion,
                         operation, static_cast<std::uint32_t>(payload.size()), status};
    std::cout.write(reinterpret_cast<const char*>(&response), sizeof(response));
    if (!payload.empty()) std::cout.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    std::cout.flush();
    if (!std::cout) throw std::runtime_error("failed to write response");
}

void validate(const Session& session, const std::vector<std::string>& names,
              const std::vector<std::vector<int64_t>>& shapes,
              const std::vector<ONNXTensorElementDataType>& types) {
    if (session.inputs() != names) throw std::runtime_error("ONNX input names do not match contract");
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (session.shape(i) != shapes[i] || session.type(i) != types[i]) {
            throw std::runtime_error("ONNX input contract mismatch for " + names[i] + " shape=" + shape_string(session.shape(i)));
        }
    }
}

void validate_outputs(const Session& session, const std::vector<std::string>& names,
                      const std::vector<std::vector<int64_t>>& shapes,
                      const std::vector<ONNXTensorElementDataType>& types) {
    if (session.outputs() != names) throw std::runtime_error("ONNX output names do not match contract");
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (session.output_shape(i) != shapes[i] || session.output_type(i) != types[i]) {
            throw std::runtime_error("ONNX output contract mismatch for " + names[i]);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        std::cerr << "Loading root model...\n";
        Session root(options.dir / "root_backbone.onnx", options.device_id, options.use_musa_graph);
        std::cerr << "Loading pose model...\n";
        Session pose(options.dir / "pose_backbone.onnx", options.device_id, options.use_musa_graph);
        std::cerr << "Loading decoder model...\n";
        Session decoder(options.dir / "vqvae_decoder.onnx", options.device_id, options.use_musa_graph);
        validate(root, {"global_root_values", "has_global_root_values", "local_root_values", "has_local_root_values", "local_poses", "has_local_poses", "num_tokens", "allowed_pred_num_tokens"}, {{1,8,5},{1,8},{1,8,4},{1,8},{1,8,304},{1,8},{1,1},{1,11}}, {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64});
        validate_outputs(root, {"num_token_logits", "pred_num_tokens", "pred_global_root_values"}, {{1,12},{1},{1,64,5}}, {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT});
        validate(pose, {"pose_tokens", "local_root_values", "pose_cond", "has_pose_cond", "num_tokens"}, {{1,16,8},{1,64,4},{1,64,304},{1,64},{1,1}}, {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64});
        validate_outputs(pose, {"pose_logits"}, {{1,16,8,10}}, {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT});
        validate(decoder, {"pose_tokens", "external_root_cond", "target_pose_cond", "has_target_cond", "token_mask"}, {{1,16,8},{1,64,2},{1,64,304},{1,64},{1,16}}, {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL});
        validate_outputs(decoder, {"recon_local_state"}, {{1,64,413}}, {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT});
        std::cerr << "MotionBricks C++ ORT backend ready\n";
        FrameHeader request{};
        while (read_exact(&request, sizeof(request))) {
            if (request.magic != motionbricks::protocol::kRequestMagic || request.version != motionbricks::protocol::kVersion) throw std::runtime_error("invalid request header");
            if (request.payload_bytes > 4u * 1024u * 1024u) throw std::runtime_error("request payload too large");
            std::vector<std::uint8_t> bytes(request.payload_bytes);
            if (!bytes.empty() && !read_exact(bytes.data(), bytes.size())) throw std::runtime_error("truncated request");
            try {
                const auto op = static_cast<Operation>(request.operation);
                if (op == Operation::kShutdown) { write_response(request.operation, motionbricks::protocol::kStatusOk, {}); return 0; }
                std::size_t offset = 0;
                std::vector<std::uint8_t> out;
                if (op == Operation::kRoot) {
                    std::array<float,40> global{}; std::array<std::uint8_t,8> global_mask{}; std::array<float,32> local{}; std::array<std::uint8_t,8> local_mask{}; std::array<float,2432> poses{}; std::array<std::uint8_t,8> pose_mask{}; std::array<int64_t,1> nt{}; std::array<int64_t,11> allowed{};
                    take(bytes,offset,global.data(),global.size()); take(bytes,offset,global_mask.data(),global_mask.size()); take(bytes,offset,local.data(),local.size()); take(bytes,offset,local_mask.data(),local_mask.size()); take(bytes,offset,poses.data(),poses.size()); take(bytes,offset,pose_mask.data(),pose_mask.size()); take(bytes,offset,nt.data(),nt.size()); take(bytes,offset,allowed.data(),allowed.size());
                    auto result = root.run({global.data(),global_mask.data(),local.data(),local_mask.data(),poses.data(),pose_mask.data(),nt.data(),allowed.data()});
                    append(out,result[0].GetTensorData<float>(),12); append(out,result[1].GetTensorData<int64_t>(),1); append(out,result[2].GetTensorData<float>(),320);
                } else if (op == Operation::kPose) {
                    std::array<int64_t,128> tokens{}; std::array<float,256> local{}; std::array<float,19456> cond{}; std::array<std::uint8_t,64> mask{}; std::array<int64_t,1> nt{};
                    take(bytes,offset,tokens.data(),tokens.size()); take(bytes,offset,local.data(),local.size()); take(bytes,offset,cond.data(),cond.size()); take(bytes,offset,mask.data(),mask.size()); take(bytes,offset,nt.data(),nt.size());
                    auto result = pose.run({tokens.data(),local.data(),cond.data(),mask.data(),nt.data()}); append(out,result[0].GetTensorData<float>(),1280);
                } else if (op == Operation::kDecoder) {
                    std::array<int64_t,128> tokens{}; std::array<float,128> external{}; std::array<float,19456> cond{}; std::array<std::uint8_t,64> mask{}; std::array<std::uint8_t,16> token_mask{};
                    take(bytes,offset,tokens.data(),tokens.size()); take(bytes,offset,external.data(),external.size()); take(bytes,offset,cond.data(),cond.size()); take(bytes,offset,mask.data(),mask.size()); take(bytes,offset,token_mask.data(),token_mask.size());
                    auto result = decoder.run({tokens.data(),external.data(),cond.data(),mask.data(),token_mask.data()}); append(out,result[0].GetTensorData<float>(),26432);
                } else throw std::runtime_error("unknown operation");
                write_response(request.operation, motionbricks::protocol::kStatusOk, out);
            } catch (const std::exception& error) { const std::string message = error.what(); write_response(request.operation, motionbricks::protocol::kStatusError, std::vector<std::uint8_t>(message.begin(), message.end())); }
        }
        return 0;
    } catch (const std::exception& error) { std::cerr << "MotionBricks C++ backend failed: " << error.what() << '\n'; return 1; }
}
