#pragma once
// ============================================================================
// RenderGraphEditor — "Render Graph" window: an ImNodes visualization of the
// GUI's pass chain (gui/PassRegistry.h mirror table).
//
// Scope is deliberately narrow: nodes = passes from the mirror table, links =
// resource dependencies (producer -> consumer, derived from the table's
// reads/writes by name), and each node carries its GpuProfiler timing plus,
// for passes with a runtime switch, an "enabled" checkbox wired to the same
// bools as the main panel's checkboxes (bidirectional — both edit the same
// state through GuiApp's applyPassToggle).  Structural passes show a locked
// checkbox; SSR is CLI/engine.toml-only and its node checkbox is locked too.
// Auto layout is layered (topological depth = column, terminal sinks parked
// in a row below the main chain, barycenter ordering within a column); it
// applies once on first open and on the "Auto Layout" button — user-dragged
// positions are kept otherwise.  Not a generic graph-editing framework:
// nodes/links are rebuilt from the table every frame, user link edits are
// ignored.
// ============================================================================
#include "gui/PassRegistry.h"
#include "renderer/core/GpuProfiler.h"

#include <functional>
#include <utility>
#include <vector>

struct ImNodesContext;

namespace sr {

class RenderGraphEditor {
public:
    bool open = false; // default closed; toggled from the main panel

    // Lazily creates the ImNodes context.  Idempotent.
    void create();
    void destroy();

    // Draw the window (no-op when closed).  getToggle/setToggle resolve the
    // PassToggle enum against GuiApp's state; setToggle applies the same side
    // effects as the matching panel checkbox.
    void draw(const GpuProfiler& profiler,
              const std::function<bool(rg::PassToggle)>& getToggle,
              const std::function<void(rg::PassToggle, bool)>& setToggle);

private:
    // Topological-layer grid layout (columns = dependency depth, sinks in a
    // bottom row).  Sets imnodes grid positions directly; caller must have
    // entered BeginNodeEditor.
    void autoLayout(const std::vector<rg::PassNode>& nodes,
                    const std::vector<std::pair<int, int>>& edges);

    ImNodesContext* ctx_ = nullptr;
    bool layoutDone_ = false; // auto layout applied (first open / Auto Layout)
};

} // namespace sr
