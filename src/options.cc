#include "src/options.h"

#include <utility>

namespace mie {
namespace internal {

void SetOption(Options* options, const char* key, std::string value) {
  for (auto& kv : *options) {
    if (kv.first == key) return;  // an explicit caller value wins
  }
  options->emplace_back(key, std::move(value));
}

const std::string* FindOption(const Options& options, const char* key) {
  for (const auto& kv : options) {
    if (kv.first == key) return &kv.second;
  }
  return nullptr;
}

void RemoveOption(Options* options, const char* key) {
  for (auto it = options->begin(); it != options->end();) {
    if (it->first == key) it = options->erase(it);
    else ++it;
  }
}

bool IsTruthy(const std::string& value) {
  return value == "true" || value == "1";
}

}  // namespace internal
}  // namespace mie
