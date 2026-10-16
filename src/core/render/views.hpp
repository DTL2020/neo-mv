#pragma once

#include "core/super/subpixel.hpp"
#include <array>

namespace neo_mv {
template <class T>
class SuperPlan;

// Borrowed level-zero views. The payload owner and plan must outlive each call.
template <class T>
struct RenderImage {
  const SuperPlan<T>* plan = nullptr;
  std::array<SubpixelPhases<T>, 3> planes{};
};
template <class T>
using RenderPixels = std::array<span2d::Plane<const T>, 3>;
// A destination is internal host integration: all active planes must be fresh,
// mutually disjoint, and disjoint from every input. Its lifetime spans the call.
template <class T>
using RenderDestination = std::array<span2d::Plane<T>, 3>;

} // namespace neo_mv
