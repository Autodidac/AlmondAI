#include "../AlmondAI/include/almondai/serve.hpp"

#include <algorithm>
#include <stdexcept>
#include <functional>
#include <sstream>
#include <variant>

namespace almondai {

namespace {

std::string compute_prompt_hash(const std::string& prompt) {
    std::hash<std::string> hasher;
    std::ostringstream oss;
    oss << std::hex << hasher(prompt);
    return oss.str();
}

bool is_null(const Json& value) {
    return std::holds_alternative<std::nullptr_t>(value.value());
}

std::string extract_string(const JsonObject& obj, const std::string& key) {
    auto it = obj.find(key);
    if (it != obj.end() && it->second.is_string()) {
        return it->second.as_string();
    }
    return std::string{};
}

std::string fetch_teacher_output(MCPBridge& bridge,
                                 const std::string& prompt,
                                 const Json& constraints) {
    if (prompt.empty()) {
        return std::string{};
    }
    JsonObject params;
    params["prompt"] = Json(prompt);
    if (!is_null(constraints)) {
        params["constraints"] = constraints;
    }
    Json response = bridge.call("gpt.generate", Json(params));
    if (response.is_object()) {
        const auto& obj = response.as_object();
        auto result_it = obj.find("result");
        if (result_it != obj.end() && result_it->second.is_object()) {
            const auto& result = result_it->second.as_object();
            if (auto out_it = result.find("output"); out_it != result.end() && out_it->second.is_string()) {
                return out_it->second.as_string();
            }
        }
    }
    return std::string{};
}

Json ensure_constraints(const JsonObject& params) {
    auto it = params.find("constraints");
    if (it != params.end()) {
        return it->second;
    }
    return Json();
}

std::string ensure_prompt_hash(const JsonObject& params, const std::string& prompt) {
    auto it = params.find("prompt_hash");
    if (it != params.end() && it->second.is_string() && !it->second.as_string().empty()) {
        return it->second.as_string();
    }
    return compute_prompt_hash(prompt);
}

} // namespace

Service::Service(ContinuousLearner& learner, MCPBridge bridge)
    : m_learner(&learner), m_bridge(std::move(bridge)) {}

void Service::run(std::istream& in, std::ostream& out) {
    while (auto request = m_bridge.read_request(in)) {
        try {
            Json result = handle_request(*request);
            m_bridge.send_response(out, request->id, result);
        } catch (const std::exception& ex) {
            m_bridge.send_error(out, request->id, ex.what());
        }
    }
}

Json Service::handle_request(const MCPBridge::Request& request) {
    if (request.method == "model.generate") {
        const auto& params = request.params.as_object();
        const std::string prompt = extract_string(params, "prompt");
        auto retrieval = m_learner->retrieval().query(prompt);
        std::string augmented = prompt;
        for (const auto& item : retrieval) {
            augmented += "\n" + m_learner->tokenizer().decode(item.tokens);
        }
        auto tokens = m_learner->tokenizer().encode(augmented);
        auto logits = m_learner->student().forward(tokens);
        auto it = std::max_element(logits.begin(), logits.end());
        int prediction = static_cast<int>(std::distance(logits.begin(), it));
        std::vector<int> decoded{prediction};
        std::string output = m_learner->tokenizer().decode(decoded);
        GovernorReport report = m_learner->governor().validate_output(output, Json());
        JsonObject payload;
        payload["output"] = Json(output);
        JsonArray hits;
        for (const auto& item : retrieval) {
            JsonObject h;
            h["document_id"] = Json(item.document_id);
            h["score"] = Json(item.score);
            hits.emplace_back(h);
        }
        payload["retrieval"] = Json(hits);
        JsonArray violations;
        for (const auto& violation : report.violations) {
            violations.emplace_back(Json(violation));
        }
        payload["violations"] = Json(violations);
        payload["allowed"] = Json(report.allowed);
        return Json(payload);
    }
    if (request.method == "retrieval.query") {
        const auto& params = request.params.as_object();
        const std::string query = params.at("query").as_string();
        auto results = m_learner->retrieval().query(query);
        JsonArray array;
        for (const auto& item : results) {
            JsonObject obj;
            obj["document_id"] = Json(item.document_id);
            obj["score"] = Json(item.score);
            array.emplace_back(obj);
        }
        return Json(array);
    }
    if (request.method == "compiler.build") {
        const auto& params = request.params.as_object();
        JsonObject response;
        if (auto it = params.find("clang"); it != params.end() && it->second.is_string()) {
            response["clang"] = parse_clang_diagnostics(it->second.as_string());
        }
        if (auto it = params.find("msbuild"); it != params.end() && it->second.is_string()) {
            response["msbuild"] = parse_msbuild_log(it->second.as_string());
        }
        return Json(response);
    }
    if (request.method == "admin.hot_swap") {
        const auto& params = request.params.as_object();
        if (auto it = params.find("name"); it != params.end() && it->second.is_string()) {
            m_learner->promote_adapter(it->second.as_string());
        } else {
            m_learner->rollback_adapter();
        }
        return Json(JsonObject{{"status", Json("ok")}});
    }
    if (request.method == "ingest.step") {
        const auto& params = request.params.as_object();
        const std::string prompt = extract_string(params, "prompt");
        std::string teacher_output = extract_string(params, "teacher_output");
        Json constraints = ensure_constraints(params);
        if (teacher_output.empty()) {
            teacher_output = fetch_teacher_output(m_bridge, prompt, constraints);
        }
        const std::string hash = ensure_prompt_hash(params, prompt);
        if (teacher_output.empty()) {
            JsonObject response;
            response["accepted"] = Json(false);
            response["teacher_output"] = Json(teacher_output);
            return Json(response);
        }
        auto curated = m_learner->ingest(prompt, teacher_output, constraints, hash);
        JsonObject response;
        response["accepted"] = Json(curated.has_value());
        response["teacher_output"] = Json(teacher_output);
        return Json(response);
    }
    if (request.method == "train.step") {
        const auto& params = request.params.as_object();
        const std::string prompt = extract_string(params, "prompt");
        std::string teacher_output = extract_string(params, "teacher_output");
        Json constraints = ensure_constraints(params);
        if (teacher_output.empty()) {
            teacher_output = fetch_teacher_output(m_bridge, prompt, constraints);
        }
        const std::string hash = ensure_prompt_hash(params, prompt);
        if (teacher_output.empty()) {
            return Json(JsonObject{{"status", Json("teacher_unavailable")}});
        }
        auto curated = m_learner->ingest(prompt, teacher_output, constraints, hash);
        if (!curated) {
            return Json(JsonObject{{"status", Json("skipped")}});
        }
        TrainingStats stats = m_learner->train_step(*curated);
        JsonObject response;
        response["status"] = Json("trained");
        response["loss"] = Json(stats.loss);
        response["accuracy"] = Json(stats.accuracy);
        response["adapter_norm"] = Json(stats.adapter_norm);
        response["retrieval_hit_rate"] = Json(stats.retrieval_hit_rate);
        response["teacher_output"] = Json(teacher_output);
        return Json(response);
    }
    if (request.method == "eval.canary") {
        TrainingStats stats = m_learner->evaluate_canary();
        JsonObject response;
        response["loss"] = Json(stats.loss);
        response["accuracy"] = Json(stats.accuracy);
        response["adapter_norm"] = Json(stats.adapter_norm);
        response["retrieval_hit_rate"] = Json(stats.retrieval_hit_rate);
        return Json(response);
    }
    throw std::runtime_error("unknown method: " + request.method);
}

} // namespace almondai

