#include "renderer/scene/CameraPath.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace sr {

namespace {

// --- Minimal JSON parser (objects/arrays/strings/numbers/bools/null) ---
struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object } type = Type::Null;
    bool bval = false;
    double num = 0.0;
    std::string str;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    const JsonValue* find(const std::string& key) const {
        for (const auto& kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : p_(text.c_str()), end_(p_ + text.size()) {}

    bool parse(JsonValue& out, std::string& err) {
        if (!parseValue(out)) {
            err = err_;
            return false;
        }
        return true;
    }

private:
    const char* p_;
    const char* end_;
    std::string err_;

    void skipWs() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
    }

    bool fail(const char* msg) {
        err_ = msg;
        return false;
    }

    bool parseValue(JsonValue& out) {
        skipWs();
        if (p_ >= end_) return fail("unexpected end of input");
        switch (*p_) {
        case '{': return parseObject(out);
        case '[': return parseArray(out);
        case '"': return parseString(out.str);
        case 't':
            if (end_ - p_ >= 4 && std::string(p_, 4) == "true") {
                out.type = JsonValue::Type::Bool;
                out.bval = true;
                p_ += 4;
                return true;
            }
            return fail("bad literal");
        case 'f':
            if (end_ - p_ >= 5 && std::string(p_, 5) == "false") {
                out.type = JsonValue::Type::Bool;
                p_ += 5;
                return true;
            }
            return fail("bad literal");
        case 'n':
            if (end_ - p_ >= 4 && std::string(p_, 4) == "null") {
                out.type = JsonValue::Type::Null;
                p_ += 4;
                return true;
            }
            return fail("bad literal");
        default:
            if (*p_ == '-' || (*p_ >= '0' && *p_ <= '9')) return parseNumber(out);
            return fail("unexpected character");
        }
    }

    bool parseObject(JsonValue& out) {
        out.type = JsonValue::Type::Object;
        ++p_; // '{'
        skipWs();
        if (p_ < end_ && *p_ == '}') { ++p_; return true; }
        while (true) {
            skipWs();
            if (p_ >= end_ || *p_ != '"') return fail("expected object key");
            std::string key;
            if (!parseString(key)) return false;
            skipWs();
            if (p_ >= end_ || *p_ != ':') return fail("expected ':'");
            ++p_;
            JsonValue value;
            if (!parseValue(value)) return false;
            out.obj.emplace_back(std::move(key), std::move(value));
            skipWs();
            if (p_ < end_ && *p_ == ',') { ++p_; continue; }
            if (p_ < end_ && *p_ == '}') { ++p_; return true; }
            return fail("expected ',' or '}'");
        }
    }

    bool parseArray(JsonValue& out) {
        out.type = JsonValue::Type::Array;
        ++p_; // '['
        skipWs();
        if (p_ < end_ && *p_ == ']') { ++p_; return true; }
        while (true) {
            JsonValue value;
            if (!parseValue(value)) return false;
            out.arr.push_back(std::move(value));
            skipWs();
            if (p_ < end_ && *p_ == ',') { ++p_; continue; }
            if (p_ < end_ && *p_ == ']') { ++p_; return true; }
            return fail("expected ',' or ']'");
        }
    }

    bool parseString(std::string& out) {
        ++p_; // '"'
        out.clear();
        while (p_ < end_) {
            const char c = *p_++;
            if (c == '"') return true;
            if (c == '\\') {
                if (p_ >= end_) return fail("bad escape");
                const char e = *p_++;
                switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                default: out += e; break;
                }
            } else {
                out += c;
            }
        }
        return fail("unterminated string");
    }

    bool parseNumber(JsonValue& out) {
        char* endp = nullptr;
        const double v = std::strtod(p_, &endp);
        if (endp == p_) return fail("bad number");
        out.type = JsonValue::Type::Number;
        out.num = v;
        p_ = endp;
        return true;
    }
};

bool readVec3(const JsonValue* arr, Vec3& out) {
    if (!arr || arr->type != JsonValue::Type::Array || arr->arr.size() < 3) return false;
    for (int i = 0; i < 3; ++i) {
        if (arr->arr[static_cast<size_t>(i)].type != JsonValue::Type::Number) return false;
    }
    out.x = static_cast<float>(arr->arr[0].num);
    out.y = static_cast<float>(arr->arr[1].num);
    out.z = static_cast<float>(arr->arr[2].num);
    return true;
}

std::string formatNumber(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.9g", v);
    return std::string(buf);
}

} // namespace

bool loadCameraPath(const char* path, CameraPath& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    JsonValue root;
    std::string err;
    JsonParser parser(text);
    if (!parser.parse(root, err)) return false;
    if (root.type != JsonValue::Type::Array) return false;

    out.clear();
    out.reserve(root.arr.size());
    for (const auto& item : root.arr) {
        if (item.type != JsonValue::Type::Object) return false;
        CameraKeyframe kf;
        if (!readVec3(item.find("position"), kf.position)) return false;
        if (!readVec3(item.find("forward"), kf.forward)) return false;
        if (!readVec3(item.find("up"), kf.up)) return false;
        kf.forward = normalize(kf.forward);
        kf.up = normalize(kf.up);
        out.push_back(kf);
    }
    return true;
}

bool saveCameraPath(const char* path, const CameraPath& in) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "[";
    for (size_t i = 0; i < in.size(); ++i) {
        if (i) out << ",";
        const CameraKeyframe& kf = in[i];
        out << "\n  {\"position\":[" << formatNumber(kf.position.x) << ","
            << formatNumber(kf.position.y) << "," << formatNumber(kf.position.z)
            << "],\"forward\":[" << formatNumber(kf.forward.x) << ","
            << formatNumber(kf.forward.y) << "," << formatNumber(kf.forward.z)
            << "],\"up\":[" << formatNumber(kf.up.x) << "," << formatNumber(kf.up.y) << ","
            << formatNumber(kf.up.z) << "]}";
    }
    out << "\n]\n";
    return true;
}

CameraPath generateOrbitPath(int frames, const Vec3& center, float radius, float heightMin,
                             float heightMax, float rotations) {
    CameraPath path;
    if (frames <= 0) return path;
    path.resize(static_cast<size_t>(frames));
    const float twoPi = 6.283185307179586f;
    for (int i = 0; i < frames; ++i) {
        const float t = frames > 1 ? static_cast<float>(i) / static_cast<float>(frames - 1) : 0.f;
        const float angle = t * rotations * twoPi;
        const float height = heightMin + (heightMax - heightMin) * t;
        CameraKeyframe& kf = path[static_cast<size_t>(i)];
        kf.position = center + Vec3{radius * std::cos(angle), height, radius * std::sin(angle)};
        kf.forward = normalize(center - kf.position);
        kf.up = {0.f, 1.f, 0.f};
    }
    return path;
}

} // namespace sr
