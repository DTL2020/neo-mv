#pragma once

#include <dualsynth/format.hpp>
#include <cstddef>

namespace neo_mv::ds2 {
struct BridgeDefaults {
  // The AviSynth adapter supplies its own enabled descriptor and registration names.
  static constexpr const char* avs_name = "";
  static constexpr const char* avs_signature = "";
  static constexpr const char* missing_input_error = "neo-mv: missing required video node";
  static constexpr const char* vs_format_error = "neo-mv: fixed DS2-compatible video format required";
  static constexpr const char* avs_format_error = vs_format_error;
  static constexpr std::size_t parity_source_index = 0;
  static constexpr bool forward_audio = false;
  static bool accepts_video_format(const ds::VideoFormat& format) {
    auto supported = ds::is_supported_video_format(format);
    return supported.has_value() && supported.value();
  }
};
} // namespace neo_mv::ds2
