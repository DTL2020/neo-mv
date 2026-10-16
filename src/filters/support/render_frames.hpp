#pragma once

#include "filters/support/super_payload.hpp"
#include "filters/support/frame_planes.hpp"
#include "core/render/views.hpp"
#include "core/render/reference.hpp"

namespace neo_mv::ds2 {
template <class T>
RenderDestination<T> fresh_render_destination(ds::VideoProcessContext& ctx) {
  return render_frame_planes<T>(ctx.dst);
}

inline RenderVideo render_video(const ds::VideoInputInfo& in) {
  return {in.width,
          in.height,
          ds::bits_per_sample(in.format.sample_format),
          in.format.color_family == ds::ColorFamily::Yuv,
          1 << in.format.subsampling_w,
          1 << in.format.subsampling_h,
          in.num_frames};
}
template <class T>
RenderImage<T> render_image(const FrameSuper<T>& source) {
  RenderImage<T> result{&source.plan, {}};
  for (int k = 0; k < source.plan.geometry().plane_count; ++k) {
    result.planes[k].pel = source.plan.params().pel;
    for (int a = 0; a < result.planes[k].pel * result.planes[k].pel; ++a)
      result.planes[k].planes[a] = source.phase(k, 0, a);
  }
  return result;
}

} // namespace neo_mv::ds2
