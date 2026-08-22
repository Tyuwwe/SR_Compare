#pragma once
// ============================================================================
// PassRegistry — declarative mirror of the GUI's pass chain for the Render
// Graph editor window.
//
// This is NOT the renderer's render graph (renderer/core/RenderGraph.h covers
// the viewer host only; Phase 7d deliberately left the GUI on its handwritten
// recordFrame chain).  It is a manually maintained description of
// GuiApp::recordFrame: one entry per pass in record order, with the resources
// it reads/writes (matched by name to derive producer->consumer links), the
// GpuProfiler zone that times it, and the runtime toggle it maps to.  When a
// pass is added/reordered in recordFrame, update the table in
// PassRegistry.cpp to match.
// ============================================================================
#include <vector>

namespace sr::rg {

// Runtime on/off switches reachable from a node.  `None` marks a structural
// pass that cannot be disabled safely (its node checkbox is locked).
enum class PassToggle {
    None,
    Shadows,
    ContactShadows, // not a pass: a screen-space ray march inside lighting
    Ssr,
    VolFog,
    Occlusion,
    Bloom,
    MotionBlur, // shares the "postfx" pass + zone with Dof
    Dof,
    AutoExposure,
    LensCa,       // not passes: compose-pass push constants
    LensVignette, // ditto
    LensGrain,    // ditto
};

struct PassNode {
    const char* name;                 // node title
    const char* zone = nullptr;       // GpuProfiler zone for the ms readout
    PassToggle toggle = PassToggle::None;
    // Editable only while this toggle is on (contact shadows need CSM shadows).
    PassToggle requires = PassToggle::None;
    std::vector<const char*> reads;   // input resources (link targets)
    std::vector<const char*> writes;  // output resources (link sources)
    const char* note = nullptr;       // tooltip (conditions, lock reason, ...)
};

// The pass table in recordFrame record order (the canonical LR chain first,
// then the cross-path/compare passes).  Static data, built once.
const std::vector<PassNode>& passTable();

} // namespace sr::rg
