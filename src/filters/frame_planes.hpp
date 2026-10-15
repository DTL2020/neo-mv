#pragma once

#include "core/base/plane.hpp"
#include <dualsynth/frame.hpp>
#include <array>
#include <limits>
#include <stdexcept>

namespace neo_mv::ds2 {
template <class T>
span2d::Plane<const T> plane(const ds::PlaneView& p) {
  return checked_plane(static_cast<const T*>(p.data), p.width, p.height, p.stride_bytes,
                       std::numeric_limits<std::size_t>::max());
}
template <class T>
span2d::Plane<T> plane(const ds::MutablePlaneView& p) {
  return checked_plane(static_cast<T*>(p.data), p.width, p.height, p.stride_bytes,
                       std::numeric_limits<std::size_t>::max());
}

// Rendering accepts only one gray plane or three YUV planes. Keep the storage
// bound local even when callers have already validated their frame format.
template <class T, class Frame>
auto render_frame_planes(const Frame& frame) {
  if (frame.plane_count != 1 && frame.plane_count != 3)
    throw std::invalid_argument("render frame requires one or three planes");
  std::array<decltype(plane<T>(frame.plane(0))), 3> result{};
  for (std::size_t k = 0; k < result.size(); ++k) {
    if (k >= static_cast<std::size_t>(frame.plane_count))
      break;
    result[k] = plane<T>(frame.plane(static_cast<int>(k)));
  }
  return result;
}
} // namespace neo_mv::ds2
