#include "renderer/math/Math.h"
#include "upscalers/UpscalerFactory.h"
#include "upscalers/sgsr/Sgsr2Upscaler.h"
#include "renderer/core/PathUtil.h"
#include "upscalers/sgsr/SgsrCommon.h"

#include <cmath>
#include <cstring>

namespace sr {

namespace {

constexpr VkFormat kHdrFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
// Encoded velocity uses UNORM16: the SGSR2 encoding (bias 32767/65535) is
// designed for it.  RGBA16F would quantize the ~0.5-biased value to ~5e-4,
// i.e. ~0.5 render pixel of reprojection error after decoding.
constexpr VkFormat kVelocityFormat = VK_FORMAT_R16G16B16A16_UNORM;
constexpr VkFormat kUIntFormat = VK_FORMAT_R32_UINT;
constexpr int kFrameSlots = 2; // matches the renderer's frames-in-flight

// std140 layout shared by all three SGSR2 passes (see the SGSR2 README).
struct Sgsr2Params {
    uint32_t renderSize[2];
    uint32_t displaySize[2];
    float renderSizeRcp[2];
    float displaySizeRcp[2];
    float jitterOffset[2];
    float padding1[2];
    float clipToPrevClip[16]; // column-major mat4
    float preExposure;
    float cameraFovAngleHor;
    float cameraNear;
    float minLerpContribution;
    uint32_t bSameCamera;
    uint32_t reset;
    float padding2[2];
};
static_assert(sizeof(Sgsr2Params) == 144, "Sgsr2Params std140 size mismatch");

struct UboBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize allocationSize = 0;
};

bool createUbo(const VulkanEnv& env, UboBuffer& out) {
    VkBufferCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = sizeof(Sgsr2Params);
    ci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(env.device, &ci, nullptr, &out.buffer) != VK_SUCCESS) return false;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(env.device, out.buffer, &req);
    const uint32_t type = sgsr::findMemoryType(
        env.physicalDevice, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == 0xFFFFFFFFu) {
        vkDestroyBuffer(env.device, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (vkAllocateMemory(env.device, &ai, nullptr, &out.memory) != VK_SUCCESS) {
        vkDestroyBuffer(env.device, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return false;
    }
    vkBindBufferMemory(env.device, out.buffer, out.memory, 0);
    vkMapMemory(env.device, out.memory, 0, sizeof(Sgsr2Params), 0, &out.mapped);
    out.allocationSize = req.size;
    return true;
}

void destroyUbo(VkDevice device, UboBuffer& ubo) {
    if (ubo.buffer) { vkDestroyBuffer(device, ubo.buffer, nullptr); ubo.buffer = VK_NULL_HANDLE; }
    if (ubo.memory) { vkFreeMemory(device, ubo.memory, nullptr); ubo.memory = VK_NULL_HANDLE; }
    ubo.mapped = nullptr;
}

Mat4 mat4From(const float* m) {
    Mat4 r;
    std::memcpy(r.m, m, sizeof(r.m));
    return r;
}

bool isCameraStill(const Mat4& curVP, const Mat4& prevVP, float threshold = 1e-5f) {
    float diff = 0.f;
    for (int i = 0; i < 16; ++i) diff += std::fabs(curVP.m[i] - prevVP.m[i]);
    return diff < threshold;
}

// One descriptor set layout per pass, matching the shader bindings exactly:
// binding 0 is always the Params UBO; `samplers`/`storages` are binding counts
// following the UBO (samplers first, matching the GLSL declarations).
VkDescriptorSetLayout createPassSetLayout(VkDevice device, uint32_t samplers, uint32_t storages) {
    VkDescriptorSetLayoutBinding bindings[8] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    uint32_t count = 1;
    for (uint32_t i = 0; i < samplers; ++i, ++count) {
        bindings[count].binding = count;
        bindings[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[count].descriptorCount = 1;
        bindings[count].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    for (uint32_t i = 0; i < storages; ++i, ++count) {
        bindings[count].binding = count;
        bindings[count].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[count].descriptorCount = 1;
        bindings[count].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = count;
    ci.pBindings = bindings;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(device, &ci, nullptr, &layout);
    return layout;
}

struct PassResources {
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorSet set[kFrameSlots] = {};
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

} // namespace

struct Sgsr2Upscaler::Impl {
    VulkanEnv env;
    UpscalerDesc desc;

    VkSampler samplerLinear = VK_NULL_HANDLE;
    VkSampler samplerNearest = VK_NULL_HANDLE;

    // Internal images (render resolution unless noted).  All stay in GENERAL
    // layout after the first-frame transition; passes are ordered by memory
    // barriers only.
    sgsr::Image2D encodedVelocity; // rgba16f render, clip-space encoded MV
    sgsr::Image2D ycocg;           // r32ui render, packed YCoCg
    sgsr::Image2D motionDepthAlpha;// rgba16f render
    sgsr::Image2D motionDepthClip; // rgba16f render
    sgsr::Image2D luma[2];         // r32ui render, ping-pong
    sgsr::Image2D history[2];      // rgba16f display, ping-pong
    int lumaRead = 0;
    int historyRead = 0;
    bool firstFrame = true;

    UboBuffer ubo[kFrameSlots];

    VkDescriptorPool pool = VK_NULL_HANDLE;

    PassResources mv;       // motion encode (input adapter)
    PassResources convert;  // SGSR2 pass 1
    PassResources activate; // SGSR2 pass 2
    PassResources upscale;  // SGSR2 pass 3

    uint64_t memoryBytes = 0;
};

Sgsr2Upscaler::~Sgsr2Upscaler() {
    shutdown();
}

const char* Sgsr2Upscaler::name() const {
    return "SGSR2";
}

uint32_t Sgsr2Upscaler::capabilities() const {
    return Cap_Temporal;
}

bool Sgsr2Upscaler::isAvailable(const VulkanEnv& env) {
    return env.device != VK_NULL_HANDLE;
}

bool Sgsr2Upscaler::init(const VulkanEnv& env, const UpscalerDesc& desc) {
    shutdown();
    impl_ = new Impl();
    impl_->env = env;
    impl_->desc = desc;

    impl_->samplerLinear = sgsr::createSampler(env.device, VK_FILTER_LINEAR);
    impl_->samplerNearest = sgsr::createSampler(env.device, VK_FILTER_NEAREST);
    if (!impl_->samplerLinear || !impl_->samplerNearest) { shutdown(); return false; }

    const VkImageUsageFlags kUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    bool ok = sgsr::createImage2D(env, desc.renderWidth, desc.renderHeight, kVelocityFormat, kUsage,
                                  impl_->encodedVelocity) &&
              sgsr::createImage2D(env, desc.renderWidth, desc.renderHeight, kUIntFormat, kUsage,
                                  impl_->ycocg) &&
              sgsr::createImage2D(env, desc.renderWidth, desc.renderHeight, kHdrFormat, kUsage,
                                  impl_->motionDepthAlpha) &&
              sgsr::createImage2D(env, desc.renderWidth, desc.renderHeight, kHdrFormat, kUsage,
                                  impl_->motionDepthClip);
    for (int i = 0; ok && i < 2; ++i) {
        ok = sgsr::createImage2D(env, desc.renderWidth, desc.renderHeight, kUIntFormat, kUsage,
                                 impl_->luma[i]) &&
             sgsr::createImage2D(env, desc.displayWidth, desc.displayHeight, kHdrFormat, kUsage,
                                 impl_->history[i]);
    }
    if (!ok) { shutdown(); return false; }
    impl_->memoryBytes += impl_->encodedVelocity.allocationSize + impl_->ycocg.allocationSize +
                          impl_->motionDepthAlpha.allocationSize +
                          impl_->motionDepthClip.allocationSize;
    for (int i = 0; i < 2; ++i)
        impl_->memoryBytes += impl_->luma[i].allocationSize + impl_->history[i].allocationSize;

    for (int i = 0; i < kFrameSlots; ++i) {
        if (!createUbo(env, impl_->ubo[i])) { shutdown(); return false; }
        impl_->memoryBytes += impl_->ubo[i].allocationSize;
    }

    // --- Descriptor set layouts (one per pass, matching the shaders) --------
    // mv:      b0 motion sampler, b1 encoded velocity storage (no UBO)
    // convert: b0 UBO, b1..b4 samplers, b5..b6 storage
    // activate/upscale: b0 UBO, b1..b3 samplers, b4..b5 storage
    {
        VkDescriptorSetLayoutBinding bindings[2] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 2;
        ci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(env.device, &ci, nullptr, &impl_->mv.setLayout) != VK_SUCCESS) {
            shutdown();
            return false;
        }
    }
    impl_->convert.setLayout = createPassSetLayout(env.device, 4, 2);
    impl_->activate.setLayout = createPassSetLayout(env.device, 3, 2);
    impl_->upscale.setLayout = createPassSetLayout(env.device, 3, 2);
    if (!impl_->convert.setLayout || !impl_->activate.setLayout || !impl_->upscale.setLayout) {
        shutdown();
        return false;
    }

    // --- Descriptor pool / sets --------------------------------------------
    VkDescriptorPoolSize poolSizes[3] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 3 * kFrameSlots;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = 11 * kFrameSlots;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[2].descriptorCount = 7 * kFrameSlots;
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.maxSets = 4 * kFrameSlots;
    poolCi.poolSizeCount = 3;
    poolCi.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(env.device, &poolCi, nullptr, &impl_->pool) != VK_SUCCESS) {
        shutdown();
        return false;
    }

    auto allocSets = [&](PassResources& pass) -> bool {
        VkDescriptorSetLayout layouts[kFrameSlots] = {pass.setLayout, pass.setLayout};
        VkDescriptorSetAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = impl_->pool;
        ai.descriptorSetCount = kFrameSlots;
        ai.pSetLayouts = layouts;
        return vkAllocateDescriptorSets(env.device, &ai, pass.set) == VK_SUCCESS;
    };
    if (!allocSets(impl_->mv) || !allocSets(impl_->convert) || !allocSets(impl_->activate) ||
        !allocSets(impl_->upscale)) {
        shutdown();
        return false;
    }
    // Static per-slot bindings: Params UBO at binding 0 of the SGSR2 passes.
    for (int i = 0; i < kFrameSlots; ++i) {
        VkDescriptorBufferInfo bi = {};
        bi.buffer = impl_->ubo[i].buffer;
        bi.offset = 0;
        bi.range = sizeof(Sgsr2Params);
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.pBufferInfo = &bi;
        for (VkDescriptorSet set :
             {impl_->convert.set[i], impl_->activate.set[i], impl_->upscale.set[i]}) {
            write.dstSet = set;
            vkUpdateDescriptorSets(env.device, 1, &write, 0, nullptr);
        }
    }

    // --- Pipeline layouts / pipelines ---------------------------------------
    {
        VkPushConstantRange push = {};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = 16; // vec4: xy = 2/renderSize, zw = renderSize
        VkPipelineLayoutCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        ci.setLayoutCount = 1;
        ci.pSetLayouts = &impl_->mv.setLayout;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(env.device, &ci, nullptr, &impl_->mv.pipelineLayout) != VK_SUCCESS) {
            shutdown();
            return false;
        }
    }
    for (PassResources* pass : {&impl_->convert, &impl_->activate, &impl_->upscale}) {
        VkPipelineLayoutCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        ci.setLayoutCount = 1;
        ci.pSetLayouts = &pass->setLayout;
        if (vkCreatePipelineLayout(env.device, &ci, nullptr, &pass->pipelineLayout) != VK_SUCCESS) {
            shutdown();
            return false;
        }
    }

    struct ShaderJob {
        std::string path;
        PassResources* pass;
    };
    const ShaderJob jobs[4] = {
        {sr::resolveShaderPath(SR_SHADER_DIR, "sgsr2_mv_encode.comp.spv"), &impl_->mv},
        {sr::resolveShaderPath(SR_SHADER_DIR, "sgsr2_convert.comp.spv"), &impl_->convert},
        {sr::resolveShaderPath(SR_SHADER_DIR, "sgsr2_activate.comp.spv"), &impl_->activate},
        {sr::resolveShaderPath(SR_SHADER_DIR, "sgsr2_upscale.comp.spv"), &impl_->upscale},
    };
    for (const ShaderJob& job : jobs) {
        VkShaderModule module = sgsr::loadShaderModule(env, job.path.c_str());
        if (!module) { shutdown(); return false; }
        job.pass->pipeline =
            sgsr::createComputePipeline(env, job.pass->pipelineLayout, module);
        vkDestroyShaderModule(env.device, module, nullptr);
        if (!job.pass->pipeline) { shutdown(); return false; }
    }

    impl_->lumaRead = 0;
    impl_->historyRead = 0;
    impl_->firstFrame = true;
    return true;
}

void Sgsr2Upscaler::dispatch(VkCommandBuffer cmd, const UpscalerResources& res,
                             const CameraParams& cam, const FrameParams& frame) {
    if (!impl_) return;
    Impl& im = *impl_;
    const int slot = frame.frameIndex % kFrameSlots;
    const uint32_t rw = im.desc.renderWidth;
    const uint32_t rh = im.desc.renderHeight;
    const uint32_t dw = im.desc.displayWidth;
    const uint32_t dh = im.desc.displayHeight;

    // First use: bring all internal images to GENERAL (they never leave it).
    if (im.firstFrame) {
        const VkImage images[] = {im.encodedVelocity.image, im.ycocg.image,
                                  im.motionDepthAlpha.image, im.motionDepthClip.image,
                                  im.luma[0].image,          im.luma[1].image,
                                  im.history[0].image,       im.history[1].image};
        for (VkImage image : images)
            sgsr::transitionImage(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        im.firstFrame = false;
    }

    // --- UBO ---------------------------------------------------------------
    const Mat4 view = mat4From(cam.view);
    const Mat4 proj = mat4From(cam.proj);
    const Mat4 prevVP = mat4From(cam.prevViewProj);
    const Mat4 curVP = Mat4::multiply(proj, view);
    const Mat4 clipToPrevClip = Mat4::multiply(prevVP, Mat4::inverse(curVP));

    Sgsr2Params params = {};
    params.renderSize[0] = rw;
    params.renderSize[1] = rh;
    params.displaySize[0] = dw;
    params.displaySize[1] = dh;
    params.renderSizeRcp[0] = 1.f / static_cast<float>(rw);
    params.renderSizeRcp[1] = 1.f / static_cast<float>(rh);
    params.displaySizeRcp[0] = 1.f / static_cast<float>(dw);
    params.displaySizeRcp[1] = 1.f / static_cast<float>(dh);
    params.jitterOffset[0] = cam.jitterX;
    params.jitterOffset[1] = cam.jitterY;
    std::memcpy(params.clipToPrevClip, clipToPrevClip.m, sizeof(params.clipToPrevClip));
    params.preExposure = frame.preExposure;
    // tan(vertical fov / 2) * renderWidth / renderHeight (SGSR2 README).
    params.cameraFovAngleHor =
        std::tan(cam.fovY * 0.5f) * static_cast<float>(rw) / static_cast<float>(rh);
    params.cameraNear = cam.cameraNear;
    params.minLerpContribution = 0.5f; // 2-pass variant only; unused here
    params.bSameCamera = isCameraStill(curVP, prevVP) ? 1u : 0u;
    params.reset = frame.resetHistory ? 1u : 0u;
    std::memcpy(im.ubo[slot].mapped, &params, sizeof(params));

    // --- Descriptor refresh (ping-pong parity + renderer-provided views) ----
    const int lumaPrev = im.lumaRead;
    const int lumaCur = 1 - im.lumaRead;
    const int histPrev = im.historyRead;
    const int histCur = 1 - im.historyRead;

    auto generalInfo = [](VkImageView view) {
        VkDescriptorImageInfo ii = {};
        ii.imageView = view;
        ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        return ii;
    };
    auto sampledInfo = [](VkSampler sampler, VkImageView view, VkImageLayout layout) {
        VkDescriptorImageInfo ii = {};
        ii.sampler = sampler;
        ii.imageView = view;
        ii.imageLayout = layout;
        return ii;
    };
    auto writeImage = [&](VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                          const VkDescriptorImageInfo* ii) {
        VkWriteDescriptorSet w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = type;
        w.pImageInfo = ii;
        vkUpdateDescriptorSets(im.env.device, 1, &w, 0, nullptr);
    };

    // MV encode pass.
    {
        VkDescriptorImageInfo in =
            sampledInfo(im.samplerNearest, res.motionView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        writeImage(im.mv.set[slot], 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &in);
        VkDescriptorImageInfo out = generalInfo(im.encodedVelocity.view);
        writeImage(im.mv.set[slot], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &out);
    }
    // Convert pass: b1 opaque color (= color, no translucency), b2 color,
    // b3 depth, b4 encoded velocity, b5 YCoCg (out), b6 MotionDepthAlpha (out).
    {
        VkDescriptorImageInfo color =
            sampledInfo(im.samplerLinear, res.colorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        VkDescriptorImageInfo depth =
            sampledInfo(im.samplerLinear, res.depthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        VkDescriptorImageInfo vel =
            sampledInfo(im.samplerNearest, im.encodedVelocity.view, VK_IMAGE_LAYOUT_GENERAL);
        VkDescriptorImageInfo yc = generalInfo(im.ycocg.view);
        VkDescriptorImageInfo mda = generalInfo(im.motionDepthAlpha.view);
        VkDescriptorSet set = im.convert.set[slot];
        writeImage(set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &color);
        writeImage(set, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &color);
        writeImage(set, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depth);
        writeImage(set, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &vel);
        writeImage(set, 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &yc);
        writeImage(set, 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mda);
    }
    // Activate pass: b1 PrevLumaHistory, b2 MotionDepthAlpha, b3 YCoCg,
    // b4 MotionDepthClipAlpha (out), b5 LumaHistory (out).
    {
        VkDescriptorImageInfo prevLuma =
            sampledInfo(im.samplerNearest, im.luma[lumaPrev].view, VK_IMAGE_LAYOUT_GENERAL);
        VkDescriptorImageInfo mda =
            sampledInfo(im.samplerLinear, im.motionDepthAlpha.view, VK_IMAGE_LAYOUT_GENERAL);
        VkDescriptorImageInfo yc =
            sampledInfo(im.samplerNearest, im.ycocg.view, VK_IMAGE_LAYOUT_GENERAL);
        VkDescriptorImageInfo mdca = generalInfo(im.motionDepthClip.view);
        VkDescriptorImageInfo lumaOut = generalInfo(im.luma[lumaCur].view);
        VkDescriptorSet set = im.activate.set[slot];
        writeImage(set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &prevLuma);
        writeImage(set, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &mda);
        writeImage(set, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &yc);
        writeImage(set, 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mdca);
        writeImage(set, 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &lumaOut);
    }
    // Upscale pass: b1 PrevHistory, b2 MotionDepthClipAlpha, b3 YCoCg,
    // b4 HistoryOutput (out), b5 SceneColorOutput (out).
    {
        VkDescriptorImageInfo prevHist =
            sampledInfo(im.samplerLinear, im.history[histPrev].view, VK_IMAGE_LAYOUT_GENERAL);
        VkDescriptorImageInfo mdca =
            sampledInfo(im.samplerLinear, im.motionDepthClip.view, VK_IMAGE_LAYOUT_GENERAL);
        VkDescriptorImageInfo yc =
            sampledInfo(im.samplerNearest, im.ycocg.view, VK_IMAGE_LAYOUT_GENERAL);
        VkDescriptorImageInfo histOut = generalInfo(im.history[histCur].view);
        VkDescriptorImageInfo sceneOut = generalInfo(res.outputView);
        VkDescriptorSet set = im.upscale.set[slot];
        writeImage(set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &prevHist);
        writeImage(set, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &mdca);
        writeImage(set, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &yc);
        writeImage(set, 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &histOut);
        writeImage(set, 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &sceneOut);
    }

    const uint32_t renderGroupsX = (rw + 7) / 8;
    const uint32_t renderGroupsY = (rh + 7) / 8;
    const uint32_t displayGroupsX = (dw + 7) / 8;
    const uint32_t displayGroupsY = (dh + 7) / 8;

    // Pass 0: motion encode (pixel units -> clip-space encoded).
    float mvPush[4] = {2.f / static_cast<float>(rw), 2.f / static_cast<float>(rh),
                       static_cast<float>(rw), static_cast<float>(rh)};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, im.mv.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, im.mv.pipelineLayout, 0, 1,
                            &im.mv.set[slot], 0, nullptr);
    vkCmdPushConstants(cmd, im.mv.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mvPush),
                       mvPush);
    vkCmdDispatch(cmd, renderGroupsX, renderGroupsY, 1);
    sgsr::computeBarrier(cmd);

    // Pass 1: Convert.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, im.convert.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, im.convert.pipelineLayout, 0, 1,
                            &im.convert.set[slot], 0, nullptr);
    vkCmdDispatch(cmd, renderGroupsX, renderGroupsY, 1);
    sgsr::computeBarrier(cmd);

    // Pass 2: Activate.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, im.activate.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, im.activate.pipelineLayout, 0, 1,
                            &im.activate.set[slot], 0, nullptr);
    vkCmdDispatch(cmd, renderGroupsX, renderGroupsY, 1);
    sgsr::computeBarrier(cmd);

    // Pass 3: Upscale (display resolution).
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, im.upscale.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, im.upscale.pipelineLayout, 0, 1,
                            &im.upscale.set[slot], 0, nullptr);
    vkCmdDispatch(cmd, displayGroupsX, displayGroupsY, 1);

    im.lumaRead = lumaCur;
    im.historyRead = histCur;
}

void Sgsr2Upscaler::shutdown() {
    if (!impl_) return;
    const VkDevice device = impl_->env.device;
    if (device) {
        for (PassResources* pass : {&impl_->mv, &impl_->convert, &impl_->activate, &impl_->upscale}) {
            if (pass->pipeline) { vkDestroyPipeline(device, pass->pipeline, nullptr); pass->pipeline = VK_NULL_HANDLE; }
            if (pass->pipelineLayout) { vkDestroyPipelineLayout(device, pass->pipelineLayout, nullptr); pass->pipelineLayout = VK_NULL_HANDLE; }
            if (pass->setLayout) { vkDestroyDescriptorSetLayout(device, pass->setLayout, nullptr); pass->setLayout = VK_NULL_HANDLE; }
        }
        if (impl_->pool) { vkDestroyDescriptorPool(device, impl_->pool, nullptr); impl_->pool = VK_NULL_HANDLE; }
        if (impl_->samplerLinear) { vkDestroySampler(device, impl_->samplerLinear, nullptr); impl_->samplerLinear = VK_NULL_HANDLE; }
        if (impl_->samplerNearest) { vkDestroySampler(device, impl_->samplerNearest, nullptr); impl_->samplerNearest = VK_NULL_HANDLE; }
        for (auto& u : impl_->ubo) destroyUbo(device, u);
        sgsr::destroyImage2D(device, impl_->encodedVelocity);
        sgsr::destroyImage2D(device, impl_->ycocg);
        sgsr::destroyImage2D(device, impl_->motionDepthAlpha);
        sgsr::destroyImage2D(device, impl_->motionDepthClip);
        for (int i = 0; i < 2; ++i) {
            sgsr::destroyImage2D(device, impl_->luma[i]);
            sgsr::destroyImage2D(device, impl_->history[i]);
        }
    }
    delete impl_;
    impl_ = nullptr;
}

uint64_t Sgsr2Upscaler::gpuMemoryBytes() const {
    return impl_ ? impl_->memoryBytes : 0;
}

std::unique_ptr<IUpscaler> createSgsr2Upscaler() { return std::make_unique<Sgsr2Upscaler>(); }

} // namespace sr

SR_REGISTER_UPSCALER("sgsr2", &sr::createSgsr2Upscaler);
