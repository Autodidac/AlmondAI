#include "../include/almondai/serve.hpp"
#include "../include/almondai/fallback.hpp"

#include <algorithm>
#include <stdexcept>
#include <functional>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <limits>
#include <variant>
#include <type_traits>

namespace almondai {

namespace {

struct DecodeSettings {
    int min_tokens = 8;
    int max_tokens = 128;
    double temperature = 0.9;
    double top_p = 0.95;
};

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

std::mt19937 make_rng() {
    std::random_device rd;
    return std::mt19937(rd());
}

int sample_token(const std::vector<double>& logits,
                 const DecodeSettings& settings,
                 std::size_t generated_tokens,
                 int eos_token,
                 std::mt19937& rng) {
    if (logits.empty()) {
        return 0;
    }

    const double temperature = std::max(settings.temperature, 1e-3);
    std::vector<double> adjusted(logits.size());
    std::transform(logits.begin(), logits.end(), adjusted.begin(), [temperature](double value) {
        return value / temperature;
    });
    const double max_logit = *std::max_element(adjusted.begin(), adjusted.end());

    std::vector<double> raw(logits.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < adjusted.size(); ++i) {
        const double value = std::exp(adjusted[i] - max_logit);
        raw[i] = value;
        sum += value;
    }

    if (generated_tokens < static_cast<std::size_t>(settings.min_tokens) && eos_token >= 0) {
        const std::size_t eos_index = static_cast<std::size_t>(eos_token);
        if (eos_index < raw.size()) {
            sum -= raw[eos_index];
            raw[eos_index] = 0.0;
        }
    }

    if (sum <= 0.0) {
        const auto it = std::max_element(logits.begin(), logits.end());
        return static_cast<int>(std::distance(logits.begin(), it));
    }

    std::vector<double> probabilities(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        probabilities[i] = raw[i] / sum;
    }

    const double top_p = std::clamp(settings.top_p, 1e-3, 1.0);
    std::vector<std::size_t> order(probabilities.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        return probabilities[lhs] > probabilities[rhs];
    });

    double cumulative = 0.0;
    std::vector<std::size_t> allowed;
    allowed.reserve(order.size());
    for (std::size_t index : order) {
        cumulative += probabilities[index];
        allowed.push_back(index);
        if (cumulative >= top_p) {
            break;
        }
    }
    if (allowed.empty() && !order.empty()) {
        allowed.push_back(order.front());
    }

    std::vector<double> weights;
    weights.reserve(allowed.size());
    double weight_sum = 0.0;
    for (std::size_t index : allowed) {
        const double weight = probabilities[index];
        weights.push_back(weight);
        weight_sum += weight;
    }

    if (weight_sum <= 0.0) {
        const auto it = std::max_element(logits.begin(), logits.end());
        return static_cast<int>(std::distance(logits.begin(), it));
    }

    std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
    const std::size_t draw = distribution(rng);
    return static_cast<int>(allowed[draw]);
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

JsonArray build_retrieval_hits(const std::vector<RetrievalResult>& results) {
    JsonArray hits;
    for (const auto& item : results) {
        JsonObject entry;
        entry["document_id"] = Json(item.document_id);
        entry["score"] = Json(item.score);
        hits.emplace_back(Json(entry));
    }
    return hits;
}

std::string build_retrieval_context(const std::vector<RetrievalResult>& results, WordTokenizer& tokenizer) {
    std::ostringstream oss;
    bool first = true;
    for (const auto& item : results) {
        const std::string decoded = tokenizer.decode(item.tokens);
        if (decoded.empty()) {
            continue;
        }
        if (!first) {
            oss << "\n\n";
        }
        first = false;
        oss << "[Retrieved] " << decoded;
    }
    return oss.str();
}

std::string summarise_hits(const JsonArray& hits) {
    if (hits.empty()) {
        return "No retrieval hits.";
    }
    std::ostringstream oss;
    for (const auto& entry : hits) {
        if (!entry.is_object()) {
            continue;
        }
        const auto& obj = entry.as_object();
        std::string id;
        double score = 0.0;
        if (auto it = obj.find("document_id"); it != obj.end() && it->second.is_string()) {
            id = it->second.as_string();
        }
        if (auto it = obj.find("score"); it != obj.end()) {
            const auto& raw = it->second.value();
            std::visit([&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_arithmetic_v<T>) {
                    score = static_cast<double>(value);
                } else if constexpr (std::is_same_v<T, std::string>) {
                    try { score = std::stod(value); } catch (...) {}
                }
            }, raw);
        }
        oss << "- " << (id.empty() ? std::string{"<unknown>"} : id)
            << " (score: " << std::fixed << std::setprecision(3) << score << ")\n";
    }
    return oss.str();
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
                JsonObject payload = handle_request(*request);
                m_bridge.send_response(out, request->id, Json(payload));
                out.flush();
            }
        } catch (const std::exception& ex) {
            m_bridge.send_error(out, request->id, ex.what());
            out.flush();
        }
    }
    out.flush();
}

JsonObject Service::handle_request(const MCPBridge::Request& request) {
    if (!m_learner) {
        throw std::runtime_error("learner unavailable");
    }

    if (request.method == "model.generate") {
        const auto& params = request.params.as_object();
        const std::string prompt = extract_string(params, "prompt");

        DecodeSettings settings;
        auto retrieval_results = m_learner->retrieval().query(prompt);
        JsonArray hits = build_retrieval_hits(retrieval_results);
        std::string augmented = prompt;
        const std::string context = build_retrieval_context(retrieval_results, m_learner->tokenizer());
        if (!context.empty()) {
            if (!augmented.empty()) {
                augmented += "\n\n";
            }
            augmented += context;
        }

        std::vector<int> tokens = m_learner->tokenizer().encode(augmented);
        std::vector<int> generated;
        generated.reserve(settings.max_tokens);
        std::mt19937 rng = make_rng();
        const int eos_token = m_learner->tokenizer().token_id("<eos>");

        for (int step = 0; step < settings.max_tokens; ++step) {
            std::vector<double> logits = m_learner->student().forward(tokens);
            int next = sample_token(logits, settings, generated.size(), eos_token, rng);
            if (next == eos_token) {
                if (generated.size() >= static_cast<std::size_t>(settings.min_tokens)) {
                    break;
                }
                double best = std::numeric_limits<double>::lowest();
                int fallback = -1;
                for (std::size_t idx = 0; idx < logits.size(); ++idx) {
                    if (static_cast<int>(idx) == eos_token) {
                        continue;
                    }
                    if (logits[idx] > best) {
                        best = logits[idx];
                        fallback = static_cast<int>(idx);
                    }
                }
                if (fallback >= 0) {
                    next = fallback;
                } else {
                    break;
                }
            }
            if (next == eos_token) {
                break;
            }
            if (next < 0) {
                break;
            }
            generated.push_back(next);
            tokens.push_back(next);
        }

        std::string output = m_learner->tokenizer().decode(generated);
        bool used_fallback = false;
        if (output.empty()) {
            JsonObject fb = fallback_response(prompt);
            if (auto it = fb.find("output"); it != fb.end() && it->second.is_string()) {
                output = it->second.as_string();
            }
            used_fallback = true;
        }

        GovernorReport report = m_learner->governor().validate_output(output, Json());
        JsonArray violations;
        for (const auto& violation : report.violations) {
            violations.emplace_back(Json(violation));
        }

        JsonObject payload;
        payload["output"] = Json(output);
        payload["route"] = Json(used_fallback ? "fallback" : "local");
        payload["prompt_hash"] = Json(compute_prompt_hash(prompt));
        payload["tokens_generated"] = Json(static_cast<int>(generated.size()));
        payload["retrieval"] = Json(hits);
        payload["retrieval_summary"] = Json(summarise_hits(hits));
        payload["violations"] = Json(violations);
        payload["allowed"] = Json(report.allowed);
        return payload;
    }

    if (request.method == "retrieval.query") {
        const auto& params = request.params.as_object();
        const std::string query = params.at("query").as_string();
        auto results = m_learner->retrieval().query(query);
        JsonArray hits = build_retrieval_hits(results);
        JsonObject payload;
        payload["output"] = Json(summarise_hits(hits));
        payload["hits"] = Json(hits);
        return payload;
    }

    if (request.method == "compiler.build") {
        const auto& params = request.params.as_object();
        JsonObject diagnostics;
        if (auto it = params.find("clang"); it != params.end() && it->second.is_string()) {
            diagnostics["clang"] = parse_clang_diagnostics(it->second.as_string());
        }
        if (auto it = params.find("msbuild"); it != params.end() && it->second.is_string()) {
            diagnostics["msbuild"] = parse_msbuild_log(it->second.as_string());
        }
        JsonObject payload;
        payload["output"] = Json("Build diagnostics parsed.");
        payload["diagnostics"] = Json(diagnostics);
        return payload;
    }

    if (request.method == "admin.hot_swap") {
        const auto& params = request.params.as_object();
        std::string message;
        if (auto it = params.find("name"); it != params.end() && it->second.is_string()) {
            const std::string name = it->second.as_string();
            m_learner->promote_adapter(name);
            message = "Promoted adapter '" + name + "'.";
        } else {
            m_learner->rollback_adapter();
            message = "Rolled back to previous adapter.";
        }
        JsonObject payload;
        payload["output"] = Json(message);
        payload["status"] = Json("ok");
        return payload;
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

        JsonObject payload;
        if (teacher_output.empty()) {
            payload["output"] = Json("Teacher response unavailable.");
            payload["accepted"] = Json(false);
            return payload;
        }

        auto curated = m_learner->ingest(prompt, teacher_output, constraints, hash);
        payload["accepted"] = Json(curated.has_value());
        payload["teacher_output"] = Json(teacher_output);
        payload["output"] = Json(curated ? "Sample ingested." : "Sample rejected by curator.");
        return payload;
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

        JsonObject payload;
        if (teacher_output.empty()) {
            payload["output"] = Json("Teacher model unavailable.");
            payload["status"] = Json("teacher_unavailable");
            return payload;
        }

        auto curated = m_learner->ingest(prompt, teacher_output, constraints, hash);
        if (!curated) {
            payload["output"] = Json("Sample skipped by curator.");
            payload["status"] = Json("skipped");
            return payload;
        }

        TrainingStats stats = m_learner->train_step(*curated);
        payload["output"] = Json("Training step completed.");
        payload["status"] = Json("trained");
        payload["loss"] = Json(stats.loss);
        payload["accuracy"] = Json(stats.accuracy);
        payload["adapter_norm"] = Json(stats.adapter_norm);
        payload["retrieval_hit_rate"] = Json(stats.retrieval_hit_rate);
        payload["teacher_output"] = Json(teacher_output);
        return payload;
    }

    if (request.method == "eval.canary") {
        TrainingStats stats = m_learner->evaluate_canary();
        JsonObject payload;
        payload["output"] = Json("Evaluation completed.");
        payload["loss"] = Json(stats.loss);
        payload["accuracy"] = Json(stats.accuracy);
        payload["adapter_norm"] = Json(stats.adapter_norm);
        payload["retrieval_hit_rate"] = Json(stats.retrieval_hit_rate);
        return payload;
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

    std::ostringstream summary;
    summary << "Training complete (loss=" << std::fixed << std::setprecision(4) << final_loss
            << ", steps=" << final_step << ")";

    JsonObject payload;
    payload["output"] = Json(summary.str());
    payload["final_loss"] = Json(final_loss);
    payload["steps"] = Json(final_step);

    m_bridge.send_response(out, request.id, Json(payload));
    out.flush();
}

} // namespace almondai

