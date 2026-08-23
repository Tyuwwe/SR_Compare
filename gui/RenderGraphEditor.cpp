// ============================================================================
// RenderGraphEditor — implementation.  See the header for the scope notes.
// ============================================================================
#include "gui/RenderGraphEditor.h"

#include <imgui.h>
#include <imnodes.h>

#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace sr {

namespace {
// Pin id packing: node index in the high bits, input/output split at 64.
// The pass table is tiny (< 64 reads+writes per node by a wide margin).
constexpr int kPinStride = 128;
int inputPinId(int node, int pin) { return node * kPinStride + pin; }
int outputPinId(int node, int pin) { return node * kPinStride + 64 + pin; }
int nodeOfPin(int pin) { return pin / kPinStride; }

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

struct DerivedGraph {
    std::vector<std::pair<int, int>> links; // (output pin, input pin)
    std::vector<std::pair<int, int>> edges; // (producer node, consumer node), deduped
};

// Derive producer -> consumer links from the table: the latest writer of a
// resource feeds every later reader (RMW chains like the HDR color read as
// read edges on the in-place pass).  Links only point backward in record
// order, so the node-level graph is a DAG by construction.
DerivedGraph deriveGraph(const std::vector<rg::PassNode>& nodes) {
    DerivedGraph g;
    std::unordered_map<std::string, int> producer; // resource -> output pin id
    for (size_t i = 0; i < nodes.size(); ++i) {
        const int ni = static_cast<int>(i);
        for (size_t r = 0; r < nodes[i].reads.size(); ++r) {
            const auto it = producer.find(nodes[i].reads[r]);
            if (it == producer.end()) continue;
            g.links.emplace_back(it->second, inputPinId(ni, static_cast<int>(r)));
            const std::pair<int, int> edge{nodeOfPin(it->second), ni};
            if (std::find(g.edges.begin(), g.edges.end(), edge) == g.edges.end())
                g.edges.push_back(edge);
        }
        for (size_t w = 0; w < nodes[i].writes.size(); ++w)
            producer[nodes[i].writes[w]] = outputPinId(ni, static_cast<int>(w));
    }
    return g;
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

// Layered auto layout (UE RDG Insight / Frostbite frame-graph style):
//   - columns are topological layers (longest path from a source) instead of
//     record order, so dependent passes stagger instead of piling onto one
//     horizontal line;
//   - terminal sink nodes (no outgoing edges: present/metrics, plus the
//     annotation-only contact-shadow/lens nodes) are parked in a separate row
//     below the main chain instead of sharing a column with it;
//   - within a column, nodes are ordered by a single left-to-right
//     barycenter sweep over their producers' rows (cuts link crossings; not
//     full Sugiyama) and spread evenly over the band height.
void RenderGraphEditor::autoLayout(const std::vector<rg::PassNode>& nodes,
                                   const std::vector<std::pair<int, int>>& edges) {
    const int n = static_cast<int>(nodes.size());
    std::vector<std::vector<int>> preds(static_cast<size_t>(n));
    std::vector<int> outDegree(static_cast<size_t>(n), 0);
    for (const auto& e : edges) {
        preds[static_cast<size_t>(e.second)].push_back(e.first);
        ++outDegree[static_cast<size_t>(e.first)];
    }

    // Longest-path layering over the DAG.  Sinks take no layer; they get the
    // bottom band below.
    std::vector<int> depth(static_cast<size_t>(n), -1);
    const std::function<int(int)> layerOf = [&](int i) -> int {
        int& d = depth[static_cast<size_t>(i)];
        if (d >= 0) return d;
        d = 0;
        for (const int p : preds[static_cast<size_t>(i)])
            d = std::max(d, layerOf(p) + 1);
        return d;
    };
    int maxDepth = -1;
    for (int i = 0; i < n; ++i) {
        if (outDegree[static_cast<size_t>(i)] == 0) continue;
        maxDepth = std::max(maxDepth, layerOf(i));
    }

    constexpr float kLeft = 40.f;
    constexpr float kTop = 40.f;
    constexpr float kColW = 260.f;
    constexpr float kRowH = 110.f;

    float bandH = kRowH;
    if (maxDepth >= 0) {
        std::vector<std::vector<int>> cols(static_cast<size_t>(maxDepth + 1));
        for (int i = 0; i < n; ++i)
            if (outDegree[static_cast<size_t>(i)] > 0)
                cols[static_cast<size_t>(depth[static_cast<size_t>(i)])].push_back(i);

        size_t maxRows = 0;
        for (const auto& c : cols) maxRows = std::max(maxRows, c.size());
        bandH = static_cast<float>(maxRows) * kRowH;

        std::vector<float> row(static_cast<size_t>(n), 0.f);
        for (size_t c = 0; c < cols.size(); ++c) {
            std::vector<int>& col = cols[c];
            if (c > 0) {
                // Barycenter: mean row of the node's producers (already
                // assigned — every producer sits in an earlier column).
                std::stable_sort(col.begin(), col.end(), [&](int a, int b) {
                    auto bary = [&](int i) {
                        const auto& ps = preds[static_cast<size_t>(i)];
                        float s = 0.f;
                        for (const int p : ps) s += row[static_cast<size_t>(p)];
                        return ps.empty() ? 0.f : s / static_cast<float>(ps.size());
                    };
                    return bary(a) < bary(b);
                });
            }
            const float cn = static_cast<float>(col.size());
            for (size_t i = 0; i < col.size(); ++i) {
                row[static_cast<size_t>(col[i])] = static_cast<float>(i);
                const float y = kTop + (static_cast<float>(i) + 0.5f) * bandH / cn;
                ImNodes::SetNodeGridSpacePos(col[i] + 1,
                                             ImVec2(kLeft + kColW * static_cast<float>(c), y));
            }
        }
    }

    // Terminal sinks: one dedicated row under the main band, ordered by mean
    // producer layer so the end of the chain (present) lands at the right.
    std::vector<int> sinks;
    for (int i = 0; i < n; ++i)
        if (outDegree[static_cast<size_t>(i)] == 0) sinks.push_back(i);
    std::stable_sort(sinks.begin(), sinks.end(), [&](int a, int b) {
        auto key = [&](int i) {
            const auto& ps = preds[static_cast<size_t>(i)];
            if (ps.empty()) return -1.f;
            float s = 0.f;
            for (const int p : ps) s += static_cast<float>(std::max(0, depth[static_cast<size_t>(p)]));
            return s / static_cast<float>(ps.size());
        };
        return key(a) < key(b);
    });
    const float sinkY = kTop + bandH + (maxDepth >= 0 ? 90.f : 0.f);
    for (size_t i = 0; i < sinks.size(); ++i)
        ImNodes::SetNodeGridSpacePos(sinks[i] + 1,
                                     ImVec2(kLeft + kColW * static_cast<float>(i), sinkY));
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
    // Re-apply the layered auto layout (overrides user-dragged positions).
    if (ImGui::Button("Auto Layout")) layoutDone_ = false;

    ImNodes::BeginNodeEditor();

    const DerivedGraph graph = deriveGraph(nodes);
    if (!layoutDone_) {
        autoLayout(nodes, graph.edges);
        layoutDone_ = true;
    }

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
            // SSR is CLI/engine.toml-only (--ssr); the GUI shows the state
            // but cannot change it (panel checkbox is locked the same way).
            const bool cliOnly = node.toggle == rg::PassToggle::Ssr;
            const bool editable =
                !cliOnly && (node.requires == rg::PassToggle::None || getToggle(node.requires));
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
        }
        for (size_t w = 0; w < node.writes.size(); ++w) {
            const int pin = outputPinId(ni, static_cast<int>(w));
            ImNodes::BeginOutputAttribute(pin);
            // Right-justify output pin labels (imnodes convention).
            const float wLabel = ImGui::CalcTextSize(node.writes[w]).x;
            ImGui::Indent(std::max(0.f, 120.f - wLabel));
            ImGui::TextUnformatted(node.writes[w]);
            ImNodes::EndOutputAttribute();
        }

        ImNodes::EndNode();
        ImGui::PopID();
    }

    int linkId = 0;
    for (const auto& l : graph.links) ImNodes::Link(linkId++, l.first, l.second);

    ImNodes::MiniMap(0.2f, ImNodesMiniMapLocation_BottomRight);
    ImNodes::EndNodeEditor();
    // User link edits are ignored on purpose (mirror view, not an editor of
    // the chain): links are rebuilt from the table every frame.

    ImGui::End();
}

} // namespace sr
