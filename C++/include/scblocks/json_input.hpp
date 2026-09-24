#pragma once
// Small strict JSON reader for frozen scientific inputs. Number tokens remain
// decimal strings until the caller selects its arithmetic precision.
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace scblocks {
struct Json {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    std::string text;
    std::vector<Json> array;
    std::map<std::string, Json> object;
    const Json &at(const std::string &key) const {
        if (kind != Object || !object.count(key))
            throw std::runtime_error("missing JSON field: " + key);
        return object.at(key);
    }
    const Json &at(size_t i) const {
        if (kind != Array || i >= array.size())
            throw std::runtime_error("invalid JSON array index");
        return array[i];
    }
    bool has(const std::string &key) const { return kind == Object && object.count(key); }
    std::string scalar() const {
        if (kind != Number && kind != String)
            throw std::runtime_error("expected a JSON number or string");
        return text;
    }
    double real() const {
        size_t n = 0;
        double v = std::stod(scalar(), &n);
        if (n != text.size()) throw std::runtime_error("invalid real JSON value");
        return v;
    }
    int integer() const {
        size_t n = 0;
        int v = std::stoi(scalar(), &n);
        if (n != text.size()) throw std::runtime_error("invalid integer JSON value");
        return v;
    }
    class Parser;
    static Json parse(const std::string &s);
    static Json read(const std::string &path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot read JSON: " + path);
        std::ostringstream s; s << in.rdbuf();
        return parse(s.str());
    }
};
class Json::Parser {
    const std::string &s;
    size_t p = 0;
    [[noreturn]] void fail() const {
        throw std::runtime_error("invalid JSON at byte " + std::to_string(p));
    }
    void space() { while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) p++; }
    bool take(char c) { space(); if (p < s.size() && s[p] == c) { p++; return true; } return false; }
    std::string string() {
        if (!take('"')) fail();
        std::string out;
        while (p < s.size()) {
            char c = s[p++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 32) fail();
            if (c != '\\') { out += c; continue; }
            if (p == s.size()) fail();
            switch (s[p++]) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                if (p + 4 > s.size()) fail();
                unsigned code = 0;
                for (int i = 0; i < 4; i++) {
                    char h = s[p++];
                    int v = h >= '0' && h <= '9' ? h-'0' : h >= 'a' && h <= 'f' ? h-'a'+10 : h >= 'A' && h <= 'F' ? h-'A'+10 : -1;
                    if (v < 0) fail();
                    code = 16*code+unsigned(v);
                }
                if (code >= 0xd800 && code <= 0xdfff) fail();
                if (code < 128) out += char(code);
                else if (code < 2048) { out += char(0xc0 | (code>>6)); out += char(0x80 | (code&63)); }
                else { out += char(0xe0 | (code>>12)); out += char(0x80 | ((code>>6)&63)); out += char(0x80 | (code&63)); }
                break;
            }
            default: fail();
            }
        }
        fail();
    }
    Json value() {
        space(); if (p == s.size()) fail();
        Json v;
        if (s[p] == '"') { v.kind = String; v.text = string(); return v; }
        if (take('{')) {
            v.kind = Object;
            if (take('}')) return v;
            do {
                auto key = string(); if (!take(':')) fail();
                if (!v.object.emplace(key, value()).second) fail();
                if (take('}')) return v;
            } while (take(','));
            fail();
        }
        if (take('[')) {
            v.kind = Array;
            if (take(']')) return v;
            do { v.array.push_back(value()); if (take(']')) return v; } while (take(','));
            fail();
        }
        for (auto literal : {"null", "true", "false"}) {
            std::string t(literal);
            if (s.compare(p,t.size(),t) == 0) { p += t.size(); v.kind = t == "null" ? Null : Bool; v.text = t; return v; }
        }
        size_t start = p;
        if (s[p] == '-') p++;
        if (p == s.size()) fail();
        if (s[p] == '0') p++;
        else {
            if (s[p] < '1' || s[p] > '9') fail();
            while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) p++;
        }
        if (p < s.size() && s[p] == '.') {
            size_t begin = ++p;
            while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) p++;
            if (p == begin) fail();
        }
        if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) {
            p++; if (p < s.size() && (s[p] == '+' || s[p] == '-')) p++;
            size_t begin = p;
            while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) p++;
            if (p == begin) fail();
        }
        v.kind = Number; v.text = s.substr(start,p-start); return v;
    }
  public:
    explicit Parser(const std::string &input) : s(input) {}
    Json run() { Json v = value(); space(); if (p != s.size()) fail(); return v; }
};
inline Json Json::parse(const std::string &s) { return Parser(s).run(); }
inline std::string json_quote(const std::string &s) {
    std::ostringstream out; out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c < 32) throw std::runtime_error("control character in JSON output");
        else out << c;
    }
    return out.str() + '"';
}
} // namespace scblocks
