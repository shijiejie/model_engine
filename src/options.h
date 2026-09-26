#ifndef MIE_INTERNAL_OPTIONS_H_
#define MIE_INTERNAL_OPTIONS_H_

#include <string>
#include <utility>
#include <vector>

namespace mie {
namespace internal {

// Key/value options handed to a vendor delegate.
using Options = std::vector<std::pair<std::string, std::string>>;

// Appends key/value unless an entry with that key already exists, so an
// explicit caller-supplied value always wins over a backend default.
void SetOption(Options* options, const char* key, std::string value);

// Returns the value for key, or nullptr when absent.
const std::string* FindOption(const Options& options, const char* key);

// "true" or "1".
bool IsTruthy(const std::string& value);

}  // namespace internal
}  // namespace mie

#endif  // MIE_INTERNAL_OPTIONS_H_
