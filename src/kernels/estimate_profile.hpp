#pragma once

#include "kernels/selection.hpp"
#include "core/depan/estimate_fft.hpp"

namespace neo_mv {
inline depan::estimate::FftProfile estimate_fft_profile() {
  return selected_backend() == KernelBackend::highway ? depan::estimate::FftProfile::native
                                                      : depan::estimate::FftProfile::scalar;
}
} // namespace neo_mv
