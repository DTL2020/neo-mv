#pragma once

#include <dualsynth/video_filter.hpp>

namespace neo_mv::ds2 {
struct OrdinaryRuntime {
  ds::VideoInputInfo source;
  explicit OrdinaryRuntime(ds::VideoInputInfo in) : source(in) {}
  virtual ~OrdinaryRuntime() = default;
  virtual void request(ds::VideoRequestContext&) const = 0;
  virtual void process(ds::VideoProcessContext&) const = 0;
  virtual ds::VideoRequestPattern pattern(int) const { return ds::VideoRequestPattern::General; }
};
} // namespace neo_mv::ds2
