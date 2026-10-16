#include "filters/support/frame_planes.hpp"
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace {
void check(bool value) {
  if (!value) throw std::runtime_error("frame plane conversion failed");
}
template <class Frame>
void invalid_counts() {
  // Invalid plane descriptors must never be read for an invalid count.
  for (int count : {-1, 0, 2, 4, INT32_MAX}) {
    Frame frame{};
    frame.plane_count = count;
    bool caught = false;
    try { (void)neo_mv::ds2::render_frame_planes<std::uint8_t>(frame); }
    catch (const std::invalid_argument& error) {
      caught = std::string(error.what()) == "render frame requires one or three planes";
    }
    check(caught);
  }
}
template <class T>
void valid_planes() {
  std::array<std::array<T, 15>, 3> storage{};
  for (int count : {1, 3}) {
    ds::VideoFrameView source{};
    ds::MutableVideoFrameView destination{};
    source.plane_count = destination.plane_count = count;
    for (int k = 0; k < count; ++k) {
      // Distinct planes with padding and a nonzero starting offset.
      source.planes[k] = {storage[k].data() + 1, 5 * sizeof(T), 3, 2};
      destination.planes[k] = {storage[k].data() + 1, 5 * sizeof(T), 3, 2};
    }
    const auto input = neo_mv::ds2::render_frame_planes<T>(source);
    const auto output = neo_mv::ds2::render_frame_planes<T>(destination);
    static_assert(std::is_same_v<std::decay_t<decltype(input)>, std::array<span2d::Plane<const T>, 3>>);
    static_assert(std::is_same_v<std::decay_t<decltype(output)>, std::array<span2d::Plane<T>, 3>>);
    for (int k = 0; k < count; ++k) {
      check(input[k].width() == 3 && input[k].height() == 2);
      check(input[k].row(1).data() == storage[k].data() + 6);
      output[k].row(1)[2] = T(10 + k);
      check(input[k].row(1)[2] == T(10 + k));
      check(storage[k][0] == T(0) && storage[k][9] == T(0));
    }
    for (int k = count; k < 3; ++k)
      check(input[k].width() == 0 && output[k].width() == 0);
  }
}
}
int main() {
  invalid_counts<ds::VideoFrameView>();
  invalid_counts<ds::MutableVideoFrameView>();
  valid_planes<std::uint8_t>();
  valid_planes<std::uint16_t>();
  valid_planes<float>();
  std::cout << "frame planes passed\n";
}
