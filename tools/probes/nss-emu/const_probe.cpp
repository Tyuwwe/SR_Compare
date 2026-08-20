// Offline probe: load the dumped graph SPIR-V with upstream SPIRV-Tools and
// look up constant id 289 in the constant manager, mimicking the layer pass.
#include <cstdio>
#include <cstdint>
#include <vector>
#include "spirv-tools/libspirv.hpp"
#include "spirv-tools/optimizer.hpp"
#include "source/opt/build_module.h"
#include "source/opt/constants.h"

int main(int argc, char** argv) {
    const char* spvPath = argc > 1 ? argv[1] : "graph_dump.spv";
    FILE* f = fopen(spvPath, "rb");
    if (!f) { printf("open %s failed\n", spvPath); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> code(static_cast<size_t>(sz) / 4);
    if (fread(code.data(), 1, static_cast<size_t>(sz), f) != static_cast<size_t>(sz)) return 1;
    fclose(f);

    // Build IRContext the same way optimizer.Run does.
    auto ctx = spvtools::BuildModule(SPV_ENV_UNIVERSAL_1_6, nullptr, code.data(), code.size());
    if (!ctx) { printf("BuildModule failed\n"); return 1; }
    auto* mgr = ctx->get_constant_mgr();
    const spvtools::opt::analysis::Constant* c = mgr->FindDeclaredConstant(289);
    printf("FindDeclaredConstant(289) = %p\n", static_cast<const void*>(c));
    if (c) printf("  AsCompositeConstant = %p\n", static_cast<const void*>(c->AsCompositeConstant()));
    // Also check the def-use manager sees %289 at all.
    auto* def = ctx->get_def_use_mgr()->GetDef(289);
    printf("GetDef(289) = %p opcode=%u\n", static_cast<const void*>(def), def ? static_cast<unsigned>(def->opcode()) : 0u);
    return 0;
}
