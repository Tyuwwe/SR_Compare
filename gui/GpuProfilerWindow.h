#pragma once
// ============================================================================
// GpuProfilerWindow — ImGui front end for renderer/core/GpuProfiler, styled
// after the UE GPU Profiler: CPU time + frame-time plot on top, a per-frame
// stacked bar graph (one column per frame, pass-colored segments), a pass
// list (color chip + ms + depth + name), and Max/Min/Ave + Pause + Graph Max
// + Max Depth controls at the bottom.
//
// The window owns only presentation state.  It snapshots the profiler's
// history ring every drawn frame unless paused (Pause freezes the snapshot,
// not the renderer).  `open` doubles as the profiler's enable switch: GuiApp
// feeds it into GpuProfiler::setEnabled so a closed panel records nothing.
// ============================================================================
#include "renderer/core/GpuProfiler.h"

#include <cstdint>
#include <vector>

namespace sr {

class GpuProfilerWindow {
public:
    bool open = false; // default closed; toggled from the main panel

    // Draw the window (no-op when closed).  cpuMs is the CPU-side cost of the
    // last recorded frame (recordFrame duration).
    void draw(const GpuProfiler& profiler, float cpuMs);

private:
    static constexpr uint32_t kMaxGraphFrames = 240; // columns in the bar graph

    bool paused_ = false;
    int graphMaxIndex_ = 2; // index into kGraphMaxMs (default 16ms)
    int maxDepth_ = 3;      // zones deeper than this are hidden
    bool sortByTime_ = true; // pass list sorted by ms desc (false = record order)

    // Frozen display snapshot (refreshed every drawn frame unless paused).
    std::vector<GpuProfiler::Frame> frames_; // oldest first
    std::vector<float> cpuHistory_;          // oldest first, capped at kMaxGraphFrames
};

} // namespace sr
