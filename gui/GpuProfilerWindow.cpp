#include "gui/GpuProfilerWindow.h"

#include <imgui.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <string>

namespace sr {
namespace {

const float kGraphMaxMs[] = {4.f, 8.f, 16.f, 33.f, 50.f};
const char* kGraphMaxLabels[] = {"4ms", "8ms", "16ms", "33ms", "50ms"};
constexpr int kGraphMaxCount = 5;

// Deterministic per-zone color (name hash -> hue), so a pass keeps its color
// across frames and matches between the graph and the list.
ImU32 zoneColor(const std::string& name) {
    uint32_t h = 2166136261u;
    for (char c : name) {
        h ^= static_cast<uint8_t>(c);
        h *= 16777619u;
    }
    const float hue = static_cast<float>(h % 360u) / 360.f;
    float r, g, b;
    ImGui::ColorConvertHSVtoRGB(hue, 0.65f, 0.85f, r, g, b);
    return IM_COL32(static_cast<int>(r * 255.f), static_cast<int>(g * 255.f),
                    static_cast<int>(b * 255.f), 255);
}

} // namespace

void GpuProfilerWindow::draw(const GpuProfiler& profiler, float cpuMs) {
    if (!open) return;

    const ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 580.f, 24.f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(560.f, 620.f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("GPU Profiler", &open)) {
        ImGui::End();
        return;
    }

    // Refresh the snapshot unless frozen.
    if (!paused_) {
        const uint32_t n = std::min(profiler.count(), kMaxGraphFrames);
        frames_.resize(n);
        for (uint32_t i = 0; i < n; ++i)
            frames_[i] = profiler.at(profiler.count() - n + i);
        cpuHistory_.push_back(cpuMs);
        if (cpuHistory_.size() > kMaxGraphFrames)
            cpuHistory_.erase(cpuHistory_.begin(),
                              cpuHistory_.begin() +
                                  static_cast<std::ptrdiff_t>(cpuHistory_.size() - kMaxGraphFrames));
    }

    const float graphMaxMs = kGraphMaxMs[graphMaxIndex_];

    // --- top: CPU time + frame-time plot -------------------------------------
    ImGui::Text("Cpu Time %.3fms", cpuHistory_.empty() ? 0.f : cpuHistory_.back());
    if (!cpuHistory_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_PlotLines, IM_COL32(230, 60, 60, 255));
        ImGui::PlotLines("##cpuplot", cpuHistory_.data(),
                         static_cast<int>(cpuHistory_.size()), 0, nullptr, 0.f, graphMaxMs,
                         ImVec2(-1.f, 40.f));
        ImGui::PopStyleColor();
    }

    if (frames_.empty()) {
        ImGui::TextUnformatted("no frames captured yet");
        ImGui::End();
        return;
    }

    // Deepest zone in the snapshot; the Max Depth slider cannot exceed it.
    uint32_t maxDepthInData = 0;
    for (const GpuProfiler::Frame& f : frames_)
        for (const GpuProfiler::Zone& z : f.zones)
            maxDepthInData = std::max(maxDepthInData, z.depth);
    maxDepth_ = std::min(maxDepth_, static_cast<int>(maxDepthInData));

    const GpuProfiler::Frame& latest = frames_.back();

    // --- middle: stacked bar graph (left) + pass list (right) -----------------
    const float listWidth = 240.f;
    ImGui::BeginChild("##graph", ImVec2(-listWidth, 200.f), false);
    {
        ImGui::Text("%.3fms  %llu Frame", latest.totalMs,
                    static_cast<unsigned long long>(latest.frameIndex));
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x;
        const float h = ImGui::GetContentRegionAvail().y - 4.f;
        ImGui::InvisibleButton("##grapharea", ImVec2(w, h));
        dl->AddRectFilled(origin, ImVec2(origin.x + w, origin.y + h), IM_COL32(18, 18, 18, 255));

        const uint32_t n = static_cast<uint32_t>(frames_.size());
        const float colW = w / static_cast<float>(n);
        for (uint32_t i = 0; i < n; ++i) {
            const GpuProfiler::Frame& f = frames_[i];
            float yBase = origin.y + h;
            const float x0 = origin.x + static_cast<float>(i) * colW;
            const float x1 = x0 + std::max(1.f, colW - 1.f);
            for (const GpuProfiler::Zone& z : f.zones) {
                if (static_cast<int>(z.depth) > maxDepth_ || z.selfMs <= 0.f) continue;
                const float segH = (z.selfMs / graphMaxMs) * h;
                const float y1 = yBase;
                const float y0 = std::max(origin.y, y1 - segH);
                dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), zoneColor(z.name));
                yBase = y0;
                if (yBase <= origin.y) break;
            }
        }
        // Frame-budget reference lines (8.33ms / 16.66ms) when in range.
        const auto refLine = [&](float ms, ImU32 col, const char* label) {
            if (ms >= graphMaxMs) return;
            const float y = origin.y + h - (ms / graphMaxMs) * h;
            dl->AddLine(ImVec2(origin.x, y), ImVec2(origin.x + w, y), col);
            dl->AddText(ImVec2(origin.x + 2.f, y - 12.f), col, label);
        };
        refLine(16.666f, IM_COL32(230, 60, 60, 255), "16.6660ms");
        refLine(8.333f, IM_COL32(230, 220, 60, 255), "8.3330ms");
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("##passes", ImVec2(0.f, 200.f), true);
    {
        // Sorted copy of the latest frame's zone indices.
        std::vector<uint32_t> order(latest.zones.size());
        for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
        if (sortByTime_) {
            std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                return latest.zones[a].ms > latest.zones[b].ms;
            });
        }
        for (uint32_t idx : order) {
            const GpuProfiler::Zone& z = latest.zones[idx];
            if (static_cast<int>(z.depth) > maxDepth_) continue;
            ImGui::PushID(static_cast<int>(idx));
            ImGui::ColorButton("##chip", ImColor(zoneColor(z.name)),
                               ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop,
                               ImVec2(12.f, 12.f));
            ImGui::PopID();
            ImGui::SameLine();
            ImGui::Text("%7.3fms %u", static_cast<double>(z.ms), z.depth);
            ImGui::SameLine(110.f + static_cast<float>(z.depth) * 12.f);
            ImGui::TextUnformatted(z.name.c_str());
        }
    }
    ImGui::EndChild();

    // --- bottom: stats + controls ---------------------------------------------
    float vMax = 0.f, vMin = FLT_MAX, vSum = 0.f;
    for (const GpuProfiler::Frame& f : frames_) {
        vMax = std::max(vMax, f.totalMs);
        vMin = std::min(vMin, f.totalMs);
        vSum += f.totalMs;
    }
    const float vAve = vSum / static_cast<float>(frames_.size());
    ImGui::Text("Max %.4fms    Min %.4fms    Ave %.4fms", static_cast<double>(vMax),
                static_cast<double>(vMin), static_cast<double>(vAve));

    ImGui::Separator();
    if (ImGui::Button(paused_ ? "Resume" : "Pause", ImVec2(80.f, 0.f))) paused_ = !paused_;

    ImGui::SetNextItemWidth(120.f);
    ImGui::Combo("Graph Max", &graphMaxIndex_, kGraphMaxLabels, kGraphMaxCount);
    ImGui::SetNextItemWidth(-1.f);
    ImGui::SliderInt("Max Depth", &maxDepth_, 0, static_cast<int>(maxDepthInData));
    ImGui::Checkbox("sort by time", &sortByTime_);

    ImGui::End();
}

} // namespace sr
