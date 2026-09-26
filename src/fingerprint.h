#ifndef MIE_INTERNAL_FINGERPRINT_H_
#define MIE_INTERNAL_FINGERPRINT_H_

#include <cstddef>
#include <string>

#include "mie/engine.h"
#include "src/model_source.h"

namespace mie {
namespace internal {

// Stable 64-bit FNV-1a hash, hex-encoded (16 chars). Identical model bytes
// always map to an identical token, which is exactly the namespace guarantee a
// compiled-model cache needs: same bytes => same graph => the cached artifact
// is valid. Returns an empty string if the input cannot be read.
std::string ModelToken(const std::string& path);
std::string ModelTokenFromMemory(const void* data, std::size_t size);

// The token to use for cfg's cache namespace: the caller's explicit
// Config::cache_token, or one derived from whichever model source is in play.
std::string CacheToken(const Config& cfg, const ModelSource& source);

}  // namespace internal
}  // namespace mie

#endif  // MIE_INTERNAL_FINGERPRINT_H_
