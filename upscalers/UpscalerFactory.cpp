#include "upscalers/UpscalerFactory.h"

#include <cstring>
#include <unordered_map>

namespace sr {
namespace {

std::unordered_map<std::string, UpscalerCreateFn>& registry() {
    static auto* map = new std::unordered_map<std::string, UpscalerCreateFn>();
    return *map;
}

std::vector<std::string>& registrationOrder() {
    static auto* order = new std::vector<std::string>();
    return *order;
}

} // namespace

bool registerUpscaler(const char* name, UpscalerCreateFn fn) {
    if (!name || !fn) return false;
    auto& map = registry();
    if (map.find(name) != map.end()) return false;
    map.emplace(name, fn);
    registrationOrder().emplace_back(name);
    return true;
}

std::unique_ptr<IUpscaler> createUpscaler(const char* name) {
    if (!name) return nullptr;
    auto& map = registry();
    auto it = map.find(name);
    if (it == map.end()) return nullptr;
    return it->second();
}

std::vector<std::string> listUpscalers() { return registrationOrder(); }

namespace {
bool g_allPluginsEnabled = false;
} // namespace

void setAllPluginsEnabled(bool enabled) { g_allPluginsEnabled = enabled; }
bool allPluginsEnabled() { return g_allPluginsEnabled; }

namespace {

std::vector<VulkanDeviceNeeds>& deviceNeeds() {
    static auto* needs = new std::vector<VulkanDeviceNeeds>();
    return *needs;
}

} // namespace

bool registerVulkanDeviceNeeds(const VulkanDeviceNeeds& needs) {
    if (!needs.appendExtensions && !needs.featureChain && !needs.appendInstanceExtensions &&
        !needs.appendInstanceLayers) {
        return false;
    }
    deviceNeeds().push_back(needs);
    return true;
}

std::vector<VulkanDeviceNeeds> collectVulkanDeviceNeeds() { return deviceNeeds(); }

namespace {

std::unordered_map<std::string, FrameGenCreateFn>& frameGenRegistry() {
    static auto* map = new std::unordered_map<std::string, FrameGenCreateFn>();
    return *map;
}

std::vector<std::string>& frameGenOrder() {
    static auto* order = new std::vector<std::string>();
    return *order;
}

} // namespace

bool registerFrameGen(const char* name, FrameGenCreateFn fn) {
    if (!name || !fn) return false;
    auto& map = frameGenRegistry();
    if (map.find(name) != map.end()) return false;
    map.emplace(name, fn);
    frameGenOrder().emplace_back(name);
    return true;
}

std::unique_ptr<IFrameGen> createFrameGen(const char* name) {
    if (!name || !*name) return nullptr;
    auto& map = frameGenRegistry();
    auto it = map.find(name);
    if (it == map.end()) return nullptr;
    return it->second();
}

std::vector<std::string> listFrameGens() { return frameGenOrder(); }

} // namespace sr
