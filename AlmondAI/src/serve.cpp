#include "../AlmondAI/include/almondai/serve.hpp"

#include <algorithm>
#include <stdexcept>
#include <functional>
#include <sstream>
#include <variant>
#include <type_traits>

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
            if (request->method == "trainer.fit") {
                handle_trainer_fit(*request, out);
            } else {
                Json result = handle_request(*request);
                m_bridge.send_response(out, request->id, result);
            }
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

void Service::handle_trainer_fit(const MCPBridge::Request& request, std::ostream& out) {
    if (!m_learner) {
        throw std::runtime_error("learner unavailable");
    }

    std::string file;
    int epochs = 1;
    int batch = 32;

    auto parse_int = [](const Json& value, int fallback) {
        return std::visit([
                               fallback
                           ](const auto& raw) -> int {
            using T = std::decay_t<decltype(raw)>;
            if constexpr (std::is_same_v<T, double>) {
                return static_cast<int>(raw);
            } else if constexpr (std::is_same_v<T, bool>) {
                return raw ? 1 : 0;
            } else if constexpr (std::is_same_v<T, std::string>) {
                try {
                    std::size_t idx = 0;
                    int parsed = std::stoi(raw, &idx);
                    if (idx == raw.size()) {
                        return parsed;
                    }
                } catch (...) {
                }
                return fallback;
            } else {
                return fallback;
            }
        }, value.value());
    };

    if (request.params.is_object()) {
        const auto& params = request.params.as_object();
        if (auto it = params.find("file"); it != params.end() && it->second.is_string()) {
            file = it->second.as_string();
        }
        if (auto it = params.find("epochs"); it != params.end()) {
            epochs = std::max(1, parse_int(it->second, epochs));
        }
        if (auto it = params.find("batch"); it != params.end()) {
            batch = std::max(1, parse_int(it->second, batch));
        }
    }

    double final_loss = 0.0;
    int final_step = 0;

    m_learner->fit(file, epochs, batch, [&](int step, double loss, double lr, double tokens_per_s) {
        JsonObject event;
        event["event"] = Json("batch");
        event["step"] = Json(step);
        event["loss"] = Json(loss);
        event["lr"] = Json(lr);
        event["tokens_per_s"] = Json(tokens_per_s);
        out << Json(event).dump() << '\n';
        out.flush();
        final_loss = loss;
        final_step = step;
    });

    JsonObject result;
    result["final_loss"] = Json(final_loss);
    result["steps"] = Json(final_step);

    JsonObject response;
    response["id"] = Json(request.id);
    response["result"] = Json(result);

    out << Json(response).dump() << '\n';
    out.flush();
}

} // namespace almondai

