// ============================================================================
// RenderGraphEditor — implementation.  See the header for the scope notes.
// ============================================================================
#include "gui/RenderGraphEditor.h"

#include <imgui.h>
#include <imnodes.h>

#include <cstring>
#include <unordered_map>

namespace sr {

namespace {
// Pin id packing: node index in the high bits, input/output split at 64.
// The pass table is tiny (< 64 reads+writes per node by a wide margin).
constexpr int kPinStride = 128;
int inputPinId(int node, int pin) { return node * kPinStride + pin; }
int outputPinId(int node, int pin) { return node * kPinStride + 64 + pin; }

// Sum the inclusive ms of every zone with this name in the frame (compare
// mode records the LR and GT paths under the same zone names).  Returns
// false when no zone matched (profiler off, or the pass did not run).
bool zoneMs(const GpuProfiler::Frame& frame, const char* zone, float& outMs) {
    bool found = false;
    outMs = 0.f;
    for (const GpuProfiler::Zone& z : frame.zones) {
        if (std::strcmp(z.name.c_str(), zone) == 0) {
            outMs += z.ms;
            found = true;
        }
    }
    return found;
}
} // namespace

void RenderGraphEditor::create() {
    if (ctx_) return;
    ctx_ = ImNodes::CreateContext();
    ImNodes::SetCurrentContext(ctx_);
    ImNodes::StyleColorsDark();
}

void RenderGraphEditor::destroy() {
    if (!ctx_) return;
    ImNodes::DestroyContext(ctx_);
    ctx_ = nullptr;
}

void RenderGraphEditor::draw(const GpuProfiler& profiler,
                             const std::function<bool(rg::PassToggle)>& getToggle,
                             const std::function<void(rg::PassToggle, bool)>& setToggle) {
    if (!open) return;
    create();
    ImNodes::SetCurrentContext(ctx_);

    const std::vector<rg::PassNode>& nodes = rg::passTable();
    const GpuProfiler::Frame* frame = profiler.count() > 0 ? &profiler.latest() : nullptr;

    ImGui::SetNextWindowSize(ImVec2(960.f, 620.f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Render Graph", &open)) {
        ImGui::End();
        return;
    }
    ImGui::TextDisabled("mirror of GuiApp::recordFrame (gui/PassRegistry) — node toggles are the "
                        "same switches as the main panel checkboxes");
    if (frame == nullptr)
        ImGui::TextDisabled("no GPU timings yet — this window enables the profiler while open");

    ImNodes::BeginNodeEditor();

    // One-shot auto layout: one column per pass, in record order.
    if (!layoutDone_) {
        for (size_t i = 0; i < nodes.size(); ++i)
            ImNodes::SetNodeGridSpacePos(static_cast<int>(i) + 1,
                                         ImVec2(40.f + 220.f * static_cast<float>(i), 60.f));
        layoutDone_ = true;
    }

    // Derive producer -> consumer links from the table while drawing: the
    // latest writer of a resource feeds every later reader (RMW chains like
    // the HDR color read as read edges on the in-place pass).
    std::unordered_map<std::string, int> producer; // resource -> output pin id
    std::vector<std::pair<int, int>> links;        // (output pin, input pin)

    for (size_t i = 0; i < nodes.size(); ++i) {
        const rg::PassNode& node = nodes[i];
        const int ni = static_cast<int>(i);
        ImGui::PushID(ni + 1);
        ImNodes::BeginNode(ni + 1);

        ImNodes::BeginNodeTitleBar();
        ImGui::TextUnformatted(node.name);
        ImNodes::EndNodeTitleBar();

        // Enable/disable — the same switch as the main panel checkbox.
        if (node.toggle != rg::PassToggle::None) {
            bool value = getToggle(node.toggle);
            const bool editable =
                node.requires == rg::PassToggle::None || getToggle(node.requires);
            ImGui::BeginDisabled(!editable);
            if (ImGui::Checkbox("enabled", &value)) setToggle(node.toggle, value);
            ImGui::EndDisabled();
        } else {
            ImGui::BeginDisabled();
            bool alwaysOn = true;
            ImGui::Checkbox("enabled", &alwaysOn);
            ImGui::EndDisabled();
        }
        if (node.note != nullptr && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", node.note);

        // GPU timing from the profiler's latest harvested frame.
        float ms = 0.f;
        if (node.zone != nullptr && frame != nullptr && zoneMs(*frame, node.zone, ms))
            ImGui::Text("%.2f ms", static_cast<double>(ms));
        else
            ImGui::TextDisabled("-- ms");

        for (size_t r = 0; r < node.reads.size(); ++r) {
            const int pin = inputPinId(ni, static_cast<int>(r));
            ImNodes::BeginInputAttribute(pin);
            ImGui::TextUnformatted(node.reads[r]);
            ImNodes::EndInputAttribute();
            const auto it = producer.find(node.reads[r]);
            if (it != producer.end()) links.emplace_back(it->second, pin);
        }
        for (size_t w = 0; w < node.writes.size(); ++w) {
            const int pin = outputPinId(ni, static_cast<int>(w));
            ImNodes::BeginOutputAttribute(pin);
            // Right-justify output pin labels (imnodes convention).
            const float wLabel = ImGui::CalcTextSize(node.writes[w]).x;
            ImGui::Indent(std::max(0.f, 120.f - wLabel));
            ImGui::TextUnformatted(node.writes[w]);
            ImNodes::EndOutputAttribute();
            producer[node.writes[w]] = pin;
        }

        ImNodes::EndNode();
        ImGui::PopID();
    }

    int linkId = 0;
    for (const auto& l : links) ImNodes::Link(linkId++, l.first, l.second);

    ImNodes::MiniMap(0.2f, ImNodesMiniMapLocation_BottomRight);
    ImNodes::EndNodeEditor();
    // User link edits are ignored on purpose (mirror view, not an editor of
    // the chain): links are rebuilt from the table every frame.

    ImGui::End();
}

} // namespace sr
