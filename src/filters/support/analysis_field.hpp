#pragma once

#include "filters/support/host.hpp"
#include "core/motion/analysis_field.hpp"
#include <map>

namespace neo_mv::ds2 {
// These host runtimes own fields produced by the fully validating decoder.
// Request state is not modified between eligibility and pixel processing.
// Standalone core/kernel APIs continue to use validating counters.
template <class Kernels>
struct DecodedFieldKernels : Kernels {
  static constexpr auto scene_count = Kernels::scene_count_validated;
};

inline AnalysisField read_field(const ds::RequestedVideoFrame& frame, const std::string& prefix, bool vectors = true) {
  const auto& props = properties(frame.frame);
  std::map<std::string, std::vector<std::int64_t>> storage;
  auto read = [&](const std::string& key) -> IntegerPropertyView {
    const auto info = props.inspect(key);
    if (!info)
      return {};
    if (info->type != ds::PropertyType::Integer)
      return {false, info->count, nullptr};
    if (info->count == 0)
      return {true, 0, nullptr};
    auto found = storage.find(key);
    if (found == storage.end())
      found = storage.emplace(key, property_array<std::int64_t>(props, key)).first;
    require(found->second.size() == info->count, "analysis property count changed during read");
    return {true, info->count, found->second.data()};
  };
  auto metadata = read_analysis_field(read, false, prefix);
  if (!vectors || metadata.state == FieldState::invalid_metadata)
    return metadata;
  // Inspect both lengths before materializing either array, including arrays
  // of host object types which DS2 intentionally cannot read as values.
  const auto v = props.inspect(prefix + "AnalysisVectors"), s = props.inspect(prefix + "AnalysisSAD");
  const auto count = field_detail::count(metadata.metadata);
  if (!v || !s || v->count != count || s->count != count)
    return metadata;
  return read_analysis_field(read, true, prefix);
}
inline void write_field(ds::FrameProperties& props, const AnalysisField& field, const std::string& prefix) {
  auto encoded = encode_analysis_field(field, prefix);
  props.erase(prefix + "AnalysisVectors");
  props.erase(prefix + "AnalysisSAD");
  for (const auto& entry : encoded)
    props.set(entry.first, entry.second);
}
inline bool same_metadata(AnalysisMetadata a, AnalysisMetadata b, bool compare_delta = true) {
  for (auto field : field_detail::scalars) {
    if (!compare_delta && field.member == &AnalysisMetadata::delta)
      continue;
    if (a.*field.member != b.*field.member)
      return false;
  }
  return a.chroma == b.chroma;
}
} // namespace neo_mv::ds2
