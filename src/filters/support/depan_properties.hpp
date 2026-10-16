#pragma once

#include "filters/support/host.hpp"
#include "core/depan/temporal.hpp"

namespace neo_mv::ds2 {
template <class T>
T depan_property(const ds::FrameProperties& props, const char* key) {
  auto values = property_array<T>(props, key);
  require(!values.empty(), "empty Depan property");
  return values[0];
}
inline depan::Motion read_motion(const ds::FrameProperties& props) {
  return depan::decode_motion(depan_property<double>(props, "Depan_dx"), depan_property<double>(props, "Depan_dy"),
                              depan_property<double>(props, "Depan_rot"), depan_property<double>(props, "Depan_zoom"),
                              depan_property<std::int64_t>(props, "Depan_goodmotion"));
}
inline void write_motion(ds::FrameProperties& props, depan::Motion m) {
  const char* keys[] = {"Depan_dx", "Depan_dy", "Depan_rot", "Depan_zoom"};
  const float values[] = {m.dx, m.dy, m.rotation, m.zoom};
  for (int i = 0; i < 4; ++i) {
    props.erase(keys[i]);
    props.set(keys[i], std::vector<double>{double(values[i])});
  }
  props.erase("Depan_goodmotion");
  set_scalar(props, "Depan_goodmotion", m.good ? 1 : 0);
}
inline void write_diagnostic(ds::FrameProperties& props, const char* key, std::string value) {
  props.erase(key);
  props.set(key, std::vector<ds::PropertyData>{{std::move(value), ds::DataHint::Utf8}});
}
inline void depan_description(const ds::VideoInputInfo& in) {
  require(in.width > 0 && in.height > 0 && in.num_frames > 0, "Depan requires constant video description");
}
} // namespace neo_mv::ds2
