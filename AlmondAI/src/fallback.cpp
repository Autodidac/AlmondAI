#include "almondai/fallback.hpp"

namespace almondai {

JsonObject fallback_response(const std::string& prompt) {
    JsonObject provenance;
    provenance["source"] = Json("gpt");
    provenance["status"] = Json("placeholder");

    JsonObject payload;
    payload["output"] = Json("Teacher model unavailable. Please provide teacher_output manually.");
    payload["provenance"] = Json(provenance);
    payload["prompt"] = Json(prompt);
    return payload;
}

} // namespace almondai

