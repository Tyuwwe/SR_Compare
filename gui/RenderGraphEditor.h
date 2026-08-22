#pragma once
// ============================================================================
// RenderGraphEditor — "Render Graph" window: an ImNodes visualization of the
// GUI's pass chain (gui/PassRegistry.h mirror table).
//
// Scope is deliberately narrow: nodes = passes in record order (one column
// each, auto-laid-out once — the user can rearrange afterwards), links =
// resource dependencies (producer -> consumer, derived from the table's
// reads/writes by name), and each node carries its GpuProfiler timing plus,
// for passes with a runtime switch, an "enabled" checkbox wired to the same
// bools as the main panel's checkboxes (bidirectional — both edit the same
// state through GuiApp's applyPassToggle).  Structural passes show a locked
// checkbox.  Not a generic graph-editing framework: nodes/links are rebuilt
// from the table every frame, user link edits are ignored.
// ============================================================================
#include "gui/PassRegistry.h"
#include "renderer/core/GpuProfiler.h"

#include <functional>

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
    ImNodesContext* ctx_ = nullptr;
    bool layoutDone_ = false; // one-shot auto layout (record order -> columns)
};

} // namespace sr
