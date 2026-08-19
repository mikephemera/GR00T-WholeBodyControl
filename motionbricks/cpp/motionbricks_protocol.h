#pragma once

#include <cstdint>

namespace motionbricks::protocol {

constexpr std::uint32_t kRequestMagic = 0x4d424351;   // MBCQ
constexpr std::uint32_t kResponseMagic = 0x4d424352;  // MBCR
constexpr std::uint16_t kVersion = 1;

enum class Operation : std::uint16_t { kRoot = 1, kPose = 2, kDecoder = 3, kShutdown = 255 };

struct FrameHeader {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t operation;
    std::uint32_t payload_bytes;
    std::uint32_t status;
};

constexpr std::uint32_t kStatusOk = 0;
constexpr std::uint32_t kStatusError = 1;

}  // namespace motionbricks::protocol
