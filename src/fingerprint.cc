#include "src/fingerprint.h"

#include <cstdint>
#include <cstdio>

namespace mie {
namespace internal {
namespace {

constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

std::uint64_t HashBytes(std::uint64_t hash, const unsigned char* data,
                        std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= static_cast<std::uint64_t>(data[i]);
    hash *= kFnvPrime;
  }
  return hash;
}

std::string ToHex(std::uint64_t hash) {
  char hex[17];
  // ::snprintf, not std::snprintf: old NDK libstdc++/libc++ revisions (r14b)
  // do not declare it in namespace std.
  ::snprintf(hex, sizeof(hex), "%016llx",
                static_cast<unsigned long long>(hash));
  return std::string(hex);
}

}  // namespace

std::string ModelToken(const std::string& path) {
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) return std::string();

  constexpr std::size_t kChunk = 1u << 16;
  char buffer[kChunk];
  std::uint64_t hash = kFnvOffsetBasis;
  std::size_t n = 0;
  while ((n = std::fread(buffer, 1, kChunk, file)) > 0) {
    hash = HashBytes(hash, reinterpret_cast<const unsigned char*>(buffer), n);
  }
  // Distinguish EOF from a mid-read I/O error: a truncated hash would
  // otherwise look like a valid cache token and could pick a stale GPU/QNN
  // cache entry. Same check ReadWholeFile() performs in run_model.cc.
  const bool ok = std::ferror(file) == 0;
  std::fclose(file);
  if (!ok) return std::string();
  return ToHex(hash);
}

std::string ModelTokenFromMemory(const void* data, std::size_t size) {
  if (data == nullptr || size == 0) return std::string();
  const std::uint64_t hash =
      HashBytes(kFnvOffsetBasis, static_cast<const unsigned char*>(data), size);
  return ToHex(hash);
}

std::string CacheToken(const Config& cfg, const ModelSource& source) {
  if (!cfg.cache_token.empty()) return cfg.cache_token;
  if (source.is_buffer()) return ModelTokenFromMemory(source.data(), source.size());
  return ModelToken(source.path());
}

}  // namespace internal
}  // namespace mie
