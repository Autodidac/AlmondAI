#include "../include/almondai/serve.hpp"
#include "../include/almondai/fallback.hpp"

#include <algorithm>
#include <cctype>
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
#include <utility>
#include <vector>
#include <chrono>
#include <thread>
#include <filesystem>
#include <fstream>
#include <unordered_set>

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

std::string trim_whitespace(const std::string& text) {
    auto begin = std::find_if_not(text.begin(), text.end(), [](unsigned char ch) { return std::isspace(ch) != 0; });
    auto end = std::find_if_not(text.rbegin(), text.rend(), [](unsigned char ch) { return std::isspace(ch) != 0; }).base();
    if (begin >= end) {
        return std::string();
    }
    return std::string(begin, end);
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

std::string summarise_hits(const JsonArray& hits);

std::vector<std::string> load_self_learning_prompts() {
    namespace fs = std::filesystem;

    const fs::path training_data{"data/training_data.jsonl"};
    const fs::path seed_data{"data/training_seed.jsonl"};

    std::unordered_set<std::string> seen;
    std::vector<std::string> prompts;
    prompts.reserve(64);

    auto add_prompt = [&](std::string prompt) {
        const std::string trimmed = trim_whitespace(prompt);
        if (trimmed.empty()) {
            return;
        }
        if (seen.insert(trimmed).second) {
            prompts.push_back(trimmed);
        }
    };

    auto load_file = [&](const fs::path& path) {
        std::ifstream in(path);
        if (!in) {
            return;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            try {
                Json record = Json::parse(line);
                if (!record.is_object()) {
                    continue;
                }
                const auto& obj = record.as_object();
                auto prompt_it = obj.find("prompt");
                if (prompt_it != obj.end() && prompt_it->second.is_string()) {
                    add_prompt(prompt_it->second.as_string());
                }
            } catch (...) {
                continue;
            }
            if (prompts.size() >= 128) {
                break;
            }
        }
    };

    if (fs::exists(seed_data)) {
        load_file(seed_data);
    }
    if (prompts.size() < 128 && fs::exists(training_data)) {
        load_file(training_data);
    }

    if (prompts.empty()) {
        const std::vector<std::string> fallback = {
            "Introduce AlmondAI to a new user.",
            "Explain how AlmondAI keeps learning between sessions.",
            "Offer a friendly greeting to someone joining the conversation.",
            "Describe how to connect AlmondAI to an external teacher model.",
            "Summarise the latest improvements to the AlmondAI runtime."
        };
        for (const auto& prompt : fallback) {
            add_prompt(prompt);
        }
    }

    return prompts;
}

struct GenerationContext {
    std::string original_prompt;
    std::string augmented_prompt;
    std::vector<RetrievalResult> retrieval;
    JsonArray hits;
    std::string retrieval_summary;
};

GenerationContext build_generation_context(ContinuousLearner& learner,
                                          const std::string& prompt,
                                          bool enable_retrieval) {
    GenerationContext ctx;
    ctx.original_prompt = prompt;
    ctx.augmented_prompt = prompt;
    if (enable_retrieval) {
        ctx.retrieval = learner.retrieval().query(prompt);
        ctx.hits = build_retrieval_hits(ctx.retrieval);
        ctx.retrieval_summary = summarise_hits(ctx.hits);
        const std::string context = build_retrieval_context(ctx.retrieval, learner.tokenizer());
        if (!context.empty()) {
            if (!ctx.augmented_prompt.empty()) {
                ctx.augmented_prompt += "\n\n";
            }
            ctx.augmented_prompt += context;
        }
    }
    return ctx;
}

struct LocalGenerationOutcome {
    std::string output;
    bool used_fallback = false;
    int tokens_generated = 0;
    JsonObject fallback_payload;
};

LocalGenerationOutcome generate_with_student(ContinuousLearner& learner,
                                             const GenerationContext& ctx,
                                             const DecodeSettings& settings) {
    LocalGenerationOutcome outcome;
    double best_score = -std::numeric_limits<double>::infinity();
    std::string retrieval_fallback;
    int retrieval_tokens = 0;
    for (const auto& result : ctx.retrieval) {
        if (result.score <= 0.0) {
            continue;
        }
        if (result.score <= best_score) {
            continue;
        }

        std::string candidate;
        if (!result.document_id.empty()) {
            if (const CuratedSample* sample = learner.recall_sample(result.document_id)) {
                if (!sample->teacher_output.empty()) {
                    candidate = sample->teacher_output;
                    retrieval_tokens = static_cast<int>(learner.tokenizer().encode(sample->teacher_output).size());
                }
            }
        }
        if (candidate.empty()) {
            const std::string decoded = learner.tokenizer().decode(result.tokens);
            if (!decoded.empty()) {
                candidate = decoded;
                retrieval_tokens = static_cast<int>(result.tokens.size());
            }
        }

        if (!candidate.empty()) {
            best_score = result.score;
            retrieval_fallback = std::move(candidate);
        }
    }
    std::vector<int> tokens = learner.tokenizer().encode(ctx.augmented_prompt);
    std::vector<int> generated;
    generated.reserve(settings.max_tokens);
    std::mt19937 rng = make_rng();
    const int eos_token = learner.tokenizer().token_id("<eos>");

    for (int step = 0; step < settings.max_tokens; ++step) {
        auto forward = learner.student().forward(tokens);
        std::vector<double> logits = std::move(forward.logits);
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
        if (next == eos_token || next < 0) {
            break;
        }
        generated.push_back(next);
        tokens.push_back(next);
    }

    outcome.tokens_generated = static_cast<int>(generated.size());
    outcome.output = learner.tokenizer().decode(generated);
    if (outcome.output.empty()) {
        if (!retrieval_fallback.empty()) {
            outcome.output = retrieval_fallback;
            outcome.tokens_generated = retrieval_tokens;
        } else {
            outcome.fallback_payload = fallback_response(ctx.original_prompt);
            outcome.used_fallback = true;
            if (auto it = outcome.fallback_payload.find("output");
                it != outcome.fallback_payload.end() && it->second.is_string()) {
                outcome.output = it->second.as_string();
            }
        }
    }
    return outcome;
}

bool has_placeholder_status(const JsonObject& payload) {
    if (auto prov_it = payload.find("provenance"); prov_it != payload.end() && prov_it->second.is_object()) {
        const auto& prov = prov_it->second.as_object();
        if (auto status_it = prov.find("status"); status_it != prov.end() && status_it->second.is_string()) {
            return status_it->second.as_string() == "placeholder";
        }
    }
    return false;
}

struct TeacherFetchOutcome {
    std::string output;
    bool placeholder = false;
    bool used_local = false;
    JsonObject fallback;
    std::string remote_error;
    std::string route;
    std::string source_label;
};

TeacherFetchOutcome fetch_teacher_output(ContinuousLearner& learner,
                                         MCPBridge& bridge,
                                         const std::string& prompt,
                                         const Json& constraints,
                                         const std::string& remote_label) {
    TeacherFetchOutcome outcome;
    if (prompt.empty()) {
        outcome.placeholder = true;
        outcome.fallback = fallback_response(prompt);
        if (auto it = outcome.fallback.find("output"); it != outcome.fallback.end() && it->second.is_string()) {
            outcome.output = it->second.as_string();
        }
        outcome.route = "fallback";
        outcome.source_label = "fallback_teacher";
        return outcome;
    }

    std::string teacher_prompt = prompt;
    if (!is_null(constraints)) {
        teacher_prompt += "\n\nConstraints:\n" + constraints.dump();
    }

    JsonObject params;
    params["prompt"] = Json(prompt);
    if (!is_null(constraints)) {
        params["constraints"] = constraints;
    }

    Json response = bridge.call("gpt.generate", Json(params));
    JsonObject payload;
    if (response.is_object()) {
        const auto& obj = response.as_object();
        if (auto result_it = obj.find("result"); result_it != obj.end() && result_it->second.is_object()) {
            payload = result_it->second.as_object();
            if (auto out_it = payload.find("output"); out_it != payload.end() && out_it->second.is_string()) {
                outcome.output = out_it->second.as_string();
            }
            if (auto err_it = payload.find("error"); err_it != payload.end() && err_it->second.is_string()) {
                outcome.remote_error = err_it->second.as_string();
            }
            if (auto prov_it = payload.find("provenance"); prov_it != payload.end() && prov_it->second.is_object()) {
                const auto& prov = prov_it->second.as_object();
                if (auto src_it = prov.find("source"); src_it != prov.end() && src_it->second.is_string()) {
                    outcome.source_label = src_it->second.as_string();
                }
            }
        }
    }

    const bool placeholder = payload.empty() || has_placeholder_status(payload);
    if (!outcome.output.empty() && !placeholder) {
        outcome.route = "remote";
        if (outcome.source_label.empty()) {
            outcome.source_label = remote_label.empty() ? std::string{"remote_teacher"} : remote_label;
        }
        return outcome;
    }

    outcome.placeholder = true;
    if (!payload.empty()) {
        outcome.fallback = payload;
    } else {
        outcome.fallback = fallback_response(prompt);
    }

    DecodeSettings settings;
    GenerationContext ctx = build_generation_context(learner, teacher_prompt, true);
    LocalGenerationOutcome local = generate_with_student(learner, ctx, settings);
    outcome.output = local.output;
    outcome.used_local = true;
    outcome.route = local.used_fallback ? "fallback" : "local";
    if (outcome.source_label.empty()) {
        outcome.source_label = local.used_fallback ? "fallback_teacher" : "local_student";
    }
    if (local.used_fallback) {
        outcome.fallback = local.fallback_payload;
    }
    return outcome;
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
    : m_learner(&learner), m_bridge(std::move(bridge)) {
    m_bridge.set_chat_backend(nullptr);
}

void Service::set_chat_backend(chat::Backend* backend, std::string route_label) {
    m_chat_backend = backend;
    m_chat_route = std::move(route_label);
    m_bridge.set_chat_backend(backend);
}

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
        GenerationContext ctx = build_generation_context(*m_learner, prompt, true);

        std::string output;
        std::string route = "local";
        bool used_fallback = false;
        int tokens_generated = 0;
        bool remote_used = false;
        std::string remote_error;
        bool include_fallback = false;
        JsonObject fallback_info;

        if (m_chat_backend) {
            try {
                std::vector<almondai::chat::Message> conversation;
                conversation.push_back({"user", ctx.augmented_prompt});
                std::string reply = trim_whitespace(m_chat_backend->complete(conversation));
                if (!reply.empty()) {
                    output = std::move(reply);
                    remote_used = true;
                    route = "remote";
                } else {
                    remote_error = "chat backend returned empty response";
                }
            } catch (const std::exception& ex) {
                remote_error = ex.what();
            }
        }

        if (!remote_used) {
            LocalGenerationOutcome local = generate_with_student(*m_learner, ctx, settings);
            output = local.output;
            used_fallback = local.used_fallback;
            tokens_generated = local.tokens_generated;
            route = used_fallback ? "fallback" : "local";
            if (local.used_fallback) {
                fallback_info = local.fallback_payload;
                include_fallback = true;
            }
            if (!include_fallback && !remote_error.empty()) {
                fallback_info = fallback_response(prompt);
                include_fallback = true;
            }
        }

        GovernorReport report = m_learner->governor().validate_output(output, Json());
        JsonArray violations;
        for (const auto& violation : report.violations) {
            violations.emplace_back(Json(violation));
        }

        JsonObject payload;
        payload["output"] = Json(output);
        payload["route"] = Json(route);
        payload["prompt_hash"] = Json(compute_prompt_hash(prompt));
        payload["tokens_generated"] = Json(remote_used ? 0 : tokens_generated);
        payload["retrieval"] = Json(ctx.hits);
        payload["retrieval_summary"] = Json(ctx.retrieval_summary.empty() ? summarise_hits(ctx.hits) : ctx.retrieval_summary);
        payload["violations"] = Json(violations);
        payload["allowed"] = Json(report.allowed);
        if (remote_used && !m_chat_route.empty()) {
            payload["backend"] = Json(m_chat_route);
        }
        if (!remote_error.empty() && !remote_used) {
            payload["remote_error"] = Json(remote_error);
        }
        if (include_fallback) {
            payload["fallback"] = Json(fallback_info);
        }
        return payload;
    }

    if (request.method == "gpt.generate") {
        const auto& params = request.params.as_object();
        const std::string prompt = extract_string(params, "prompt");
        Json constraints = ensure_constraints(params);

        std::string teacher_prompt = prompt;
        if (!is_null(constraints)) {
            teacher_prompt += "\n\nConstraints:\n" + constraints.dump();
        }

        DecodeSettings settings;
        GenerationContext ctx = build_generation_context(*m_learner, teacher_prompt, true);

        std::string output;
        bool remote_used = false;
        bool used_fallback = false;
        bool include_fallback = false;
        JsonObject fallback_info;
        std::string remote_error;

        if (m_chat_backend) {
            try {
                std::vector<almondai::chat::Message> conversation;
                conversation.push_back({"system", "You are AlmondAI's teacher model. Provide thorough, safe answers suitable for fine-tuning."});
                conversation.push_back({"user", ctx.augmented_prompt});
                std::string reply = trim_whitespace(m_chat_backend->complete(conversation));
                if (!reply.empty()) {
                    output = std::move(reply);
                    remote_used = true;
                } else {
                    remote_error = "chat backend returned empty response";
                }
            } catch (const std::exception& ex) {
                remote_error = ex.what();
            }
        }

        if (!remote_used) {
            LocalGenerationOutcome local = generate_with_student(*m_learner, ctx, settings);
            output = local.output;
            used_fallback = local.used_fallback;
            if (local.used_fallback) {
                fallback_info = local.fallback_payload;
                include_fallback = true;
            }
            if (!include_fallback) {
                fallback_info = fallback_response(prompt);
                include_fallback = true;
            }
        }

        GovernorReport report = m_learner->governor().validate_output(output, Json());
        JsonArray violations;
        for (const auto& violation : report.violations) {
            violations.emplace_back(Json(violation));
        }

        JsonObject provenance;
        provenance["source"] = Json(remote_used ? "chat_backend" : "local");
        provenance["status"] = Json(remote_used ? "remote" : (used_fallback ? "fallback" : "local"));
        if (remote_used && !m_chat_route.empty()) {
            provenance["backend"] = Json(m_chat_route);
        }

        JsonObject payload;
        payload["output"] = Json(output);
        payload["route"] = Json(remote_used ? "remote" : (used_fallback ? "fallback" : "local"));
        payload["provenance"] = Json(provenance);
        payload["violations"] = Json(violations);
        payload["allowed"] = Json(report.allowed);
        payload["retrieval"] = Json(ctx.hits);
        payload["retrieval_summary"] = Json(ctx.retrieval_summary.empty() ? summarise_hits(ctx.hits) : ctx.retrieval_summary);
        if (!remote_error.empty() && !remote_used) {
            payload["remote_error"] = Json(remote_error);
        }
        if (include_fallback) {
            payload["fallback"] = Json(fallback_info);
        }
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
        std::string teacher_source = extract_string(params, "teacher_source");
        Json constraints = ensure_constraints(params);
        TeacherFetchOutcome teacher;
        bool fetched = false;
        if (teacher_output.empty()) {
            teacher = fetch_teacher_output(*m_learner, m_bridge, prompt, constraints, m_chat_route);
            teacher_output = teacher.output;
            fetched = true;
            if (!teacher.source_label.empty()) {
                teacher_source = teacher.source_label;
            }
        }
        const std::string hash = ensure_prompt_hash(params, prompt);

        JsonObject payload;
        if (teacher_output.empty()) {
            payload["output"] = Json("Teacher response unavailable.");
            payload["accepted"] = Json(false);
            if (!teacher_source.empty()) {
                payload["teacher_source"] = Json(teacher_source);
            }
            if (fetched && !teacher.fallback.empty()) {
                payload["fallback"] = Json(teacher.fallback);
            }
            if (fetched && !teacher.remote_error.empty()) {
                payload["remote_error"] = Json(teacher.remote_error);
            }
            if (fetched) {
                std::string teacher_route = teacher.placeholder ? (teacher.used_local ? "local" : "fallback") : "remote";
                payload["teacher_route"] = Json(teacher_route);
            }
            return payload;
        }

        if (teacher_source.empty()) {
            if (fetched) {
                if (teacher.route == "remote") {
                    teacher_source = m_chat_route.empty() ? "remote_teacher" : m_chat_route;
                } else if (teacher.route == "local") {
                    teacher_source = "local_student";
                } else {
                    teacher_source = "fallback_teacher";
                }
            } else {
                teacher_source = "external_teacher";
            }
        }

        auto curated = m_learner->ingest(prompt, teacher_output, constraints, hash, teacher_source);
        payload["accepted"] = Json(curated.has_value());
        payload["teacher_output"] = Json(teacher_output);
        payload["output"] = Json(curated ? "Sample ingested." : "Sample rejected by curator.");
        payload["teacher_source"] = Json(teacher_source);
        if (fetched && !teacher.fallback.empty()) {
            payload["fallback"] = Json(teacher.fallback);
        }
        if (fetched && !teacher.remote_error.empty()) {
            payload["remote_error"] = Json(teacher.remote_error);
        }
        if (fetched) {
            std::string teacher_route = teacher.placeholder ? (teacher.used_local ? "local" : "fallback") : "remote";
            payload["teacher_route"] = Json(teacher_route);
        }
        return payload;
    }

    if (request.method == "train.step") {
        const auto& params = request.params.as_object();
        const std::string prompt = extract_string(params, "prompt");
        std::string teacher_output = extract_string(params, "teacher_output");
        std::string teacher_source = extract_string(params, "teacher_source");
        Json constraints = ensure_constraints(params);
        TeacherFetchOutcome teacher;
        bool fetched = false;
        if (teacher_output.empty()) {
            teacher = fetch_teacher_output(*m_learner, m_bridge, prompt, constraints, m_chat_route);
            teacher_output = teacher.output;
            fetched = true;
            if (!teacher.source_label.empty()) {
                teacher_source = teacher.source_label;
            }
        }
        const std::string hash = ensure_prompt_hash(params, prompt);

        JsonObject payload;
        if (teacher_output.empty()) {
            payload["output"] = Json("Teacher model unavailable.");
            payload["status"] = Json("teacher_unavailable");
            if (!teacher_source.empty()) {
                payload["teacher_source"] = Json(teacher_source);
            }
            if (fetched && !teacher.fallback.empty()) {
                payload["fallback"] = Json(teacher.fallback);
            }
            if (fetched && !teacher.remote_error.empty()) {
                payload["remote_error"] = Json(teacher.remote_error);
            }
            if (fetched) {
                std::string teacher_route = teacher.placeholder ? (teacher.used_local ? "local" : "fallback") : "remote";
                payload["teacher_route"] = Json(teacher_route);
            }
            return payload;
        }

        if (teacher_source.empty()) {
            if (fetched) {
                if (teacher.route == "remote") {
                    teacher_source = m_chat_route.empty() ? "remote_teacher" : m_chat_route;
                } else if (teacher.route == "local") {
                    teacher_source = "local_student";
                } else {
                    teacher_source = "fallback_teacher";
                }
            } else {
                teacher_source = "external_teacher";
            }
        }

        auto curated = m_learner->ingest(prompt, teacher_output, constraints, hash, teacher_source);
        if (!curated) {
            payload["output"] = Json("Sample skipped by curator.");
            payload["status"] = Json("skipped");
            payload["teacher_source"] = Json(teacher_source);
            if (fetched && !teacher.fallback.empty()) {
                payload["fallback"] = Json(teacher.fallback);
            }
            if (fetched && !teacher.remote_error.empty()) {
                payload["remote_error"] = Json(teacher.remote_error);
            }
            if (fetched) {
                std::string teacher_route = teacher.placeholder ? (teacher.used_local ? "local" : "fallback") : "remote";
                payload["teacher_route"] = Json(teacher_route);
            }
            return payload;
        }

        TrainingStats stats = m_learner->train_step(*curated);
        payload["output"] = Json("Training step completed.");
        payload["status"] = Json("trained");
        payload["loss"] = Json(stats.loss);
        payload["accuracy"] = Json(stats.accuracy);
        payload["adapter_norm"] = Json(stats.adapter_norm);
        payload["retrieval_hit_rate"] = Json(stats.retrieval_hit_rate);
        if (!stats.learning_tags.empty()) {
            JsonArray tags_json;
            tags_json.reserve(stats.learning_tags.size());
            for (const auto& tag : stats.learning_tags) {
                tags_json.emplace_back(Json(tag));
            }
            payload["learning_tags"] = Json(tags_json);
        }
        if (!stats.learning_trace.empty()) {
            payload["learning_trace"] = Json(stats.learning_trace);
        }
        payload["teacher_output"] = Json(teacher_output);
        payload["teacher_source"] = Json(teacher_source);
        if (fetched && !teacher.fallback.empty()) {
            payload["fallback"] = Json(teacher.fallback);
        }
        if (fetched && !teacher.remote_error.empty()) {
            payload["remote_error"] = Json(teacher.remote_error);
        }
        if (fetched) {
            std::string teacher_route = teacher.placeholder ? (teacher.used_local ? "local" : "fallback") : "remote";
            payload["teacher_route"] = Json(teacher_route);
        }
        return payload;
    }

    if (request.method == "train.self_loop") {
        const JsonObject& params = request.params.is_object() ? request.params.as_object() : JsonObject{};

        auto parse_int = [](const Json& value, int fallback) {
            return std::visit(
                [fallback](const auto& raw) -> int {
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
                },
                value.value());
        };

        auto parse_bool = [](const Json& value, bool fallback) {
            return std::visit(
                [fallback](const auto& raw) -> bool {
                    using T = std::decay_t<decltype(raw)>;
                    if constexpr (std::is_same_v<T, bool>) {
                        return raw;
                    } else if constexpr (std::is_same_v<T, double>) {
                        return raw != 0.0;
                    } else if constexpr (std::is_same_v<T, std::string>) {
                        std::string lowered = raw;
                        std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
                            return static_cast<char>(std::tolower(ch));
                        });
                        if (lowered == "true" || lowered == "1" || lowered == "yes") {
                            return true;
                        }
                        if (lowered == "false" || lowered == "0" || lowered == "no") {
                            return false;
                        }
                        return fallback;
                    } else {
                        return fallback;
                    }
                },
                value.value());
        };

        int loops = 1;
        int delay_ms = 0;
        bool shuffle = false;
        bool force_new = true;
        int limit = 0;

        if (auto it = params.find("loops"); it != params.end()) {
            loops = parse_int(it->second, loops);
        }
        if (auto it = params.find("delay_ms"); it != params.end()) {
            delay_ms = parse_int(it->second, delay_ms);
        }
        if (auto it = params.find("shuffle"); it != params.end()) {
            shuffle = parse_bool(it->second, shuffle);
        }
        if (auto it = params.find("force_new"); it != params.end()) {
            force_new = parse_bool(it->second, force_new);
        }
        if (auto it = params.find("limit"); it != params.end()) {
            limit = parse_int(it->second, limit);
        }

        loops = std::clamp(loops, 1, 1000);
        delay_ms = std::clamp(delay_ms, 0, 60000);
        if (limit < 0) {
            limit = 0;
        }

        std::vector<std::string> prompts = load_self_learning_prompts();
        if (prompts.empty()) {
            throw std::runtime_error("no prompts available for self-learning");
        }

        const std::size_t max_total = 10000;
        std::size_t desired_total = static_cast<std::size_t>(loops) * prompts.size();
        if (limit > 0) {
            desired_total = std::min<std::size_t>(desired_total, static_cast<std::size_t>(limit));
        }
        desired_total = std::min(desired_total, max_total);
        if (desired_total == 0) {
            desired_total = std::min<std::size_t>(prompts.size(), max_total);
        }

        std::vector<std::size_t> order(prompts.size());
        std::iota(order.begin(), order.end(), std::size_t{0});

        std::size_t processed = 0;
        int loops_completed = 0;
        int trained = 0;
        int skipped = 0;
        int unavailable = 0;
        double loss_accumulator = 0.0;
        double accuracy_accumulator = 0.0;
        bool events_truncated = false;

        JsonArray events;
        events.reserve(std::min<std::size_t>(desired_total, std::size_t{200}));

        std::mt19937 rng = make_rng();

        while (processed < desired_total && loops_completed < loops) {
            if (shuffle) {
                std::shuffle(order.begin(), order.end(), rng);
            }

            for (std::size_t index : order) {
                if (processed >= desired_total) {
                    break;
                }

                const std::string& prompt = prompts[index];
                TeacherFetchOutcome teacher = fetch_teacher_output(*m_learner, m_bridge, prompt, Json(), m_chat_route);

                JsonObject event;
                event["prompt"] = Json(prompt);
                event["loop"] = Json(loops_completed + 1);
                event["iteration"] = Json(static_cast<int>(processed + 1));
                if (!teacher.route.empty()) {
                    event["teacher_route"] = Json(teacher.route);
                }
                if (!teacher.source_label.empty()) {
                    event["teacher_source"] = Json(teacher.source_label);
                }
                if (!teacher.remote_error.empty()) {
                    event["remote_error"] = Json(teacher.remote_error);
                }
                if (teacher.placeholder) {
                    event["placeholder"] = Json(true);
                }

                std::string teacher_output = teacher.output;
                if (teacher_output.empty()) {
                    event["status"] = Json("teacher_unavailable");
                    ++unavailable;
                } else {
                    std::string teacher_source = teacher.source_label;
                    if (teacher_source.empty()) {
                        if (teacher.route == "remote") {
                            teacher_source = m_chat_route.empty() ? "remote_teacher" : m_chat_route;
                        } else if (teacher.route == "local") {
                            teacher_source = "local_student";
                        } else {
                            teacher_source = "fallback_teacher";
                        }
                    }

                    std::string prompt_hash;
                    if (force_new) {
                        std::ostringstream hash_seed;
                        hash_seed << prompt << "::selfloop::" << loops_completed << ':' << processed;
                        prompt_hash = compute_prompt_hash(hash_seed.str());
                    } else {
                        prompt_hash = compute_prompt_hash(prompt);
                    }

                    std::string curated_source = teacher_source;
                    if (force_new) {
                        std::ostringstream tagged_source;
                        tagged_source << teacher_source << "::selfloop::" << (loops_completed + 1) << ':' << (processed + 1);
                        curated_source = tagged_source.str();
                    }

                    auto curated = m_learner->ingest(prompt, teacher_output, Json(), prompt_hash, curated_source);
                    if (!curated) {
                        event["status"] = Json("skipped");
                        ++skipped;
                    } else {
                        TrainingStats stats = m_learner->train_step(*curated);
                        event["status"] = Json("trained");
                        event["loss"] = Json(stats.loss);
                        event["accuracy"] = Json(stats.accuracy);
                        event["adapter_norm"] = Json(stats.adapter_norm);
                        event["retrieval_hit_rate"] = Json(stats.retrieval_hit_rate);
                        if (!stats.learning_tags.empty()) {
                            JsonArray tags;
                            tags.reserve(stats.learning_tags.size());
                            for (const auto& tag : stats.learning_tags) {
                                tags.emplace_back(Json(tag));
                            }
                            event["learning_tags"] = Json(tags);
                        }
                        if (!stats.learning_trace.empty()) {
                            event["learning_trace"] = Json(stats.learning_trace);
                        }
                        loss_accumulator += stats.loss;
                        accuracy_accumulator += stats.accuracy;
                        ++trained;
                    }
                }

                if (events.size() < 200) {
                    events.emplace_back(Json(event));
                } else {
                    events_truncated = true;
                }

                ++processed;

                if (delay_ms > 0 && processed < desired_total) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                }
            }

            ++loops_completed;
        }

        std::ostringstream summary;
        summary << "Self-learning processed " << processed << " prompt" << (processed == 1 ? "" : "s")
                << " (trained=" << trained
                << ", skipped=" << skipped
                << ", teacher_unavailable=" << unavailable << ")";

        JsonObject payload;
        payload["output"] = Json(summary.str());
        payload["loops_requested"] = Json(loops);
        payload["loops_completed"] = Json(loops_completed);
        payload["processed"] = Json(static_cast<int>(processed));
        payload["trained"] = Json(trained);
        payload["skipped"] = Json(skipped);
        payload["teacher_unavailable"] = Json(unavailable);
        if (trained > 0) {
            payload["average_loss"] = Json(loss_accumulator / static_cast<double>(trained));
            payload["average_accuracy"] = Json(accuracy_accumulator / static_cast<double>(trained));
        }
        if (!events.empty()) {
            payload["events"] = Json(events);
            if (events_truncated) {
                payload["events_truncated"] = Json(true);
            }
        }

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
        if (!stats.learning_tags.empty()) {
            JsonArray tags_json;
            tags_json.reserve(stats.learning_tags.size());
            for (const auto& tag : stats.learning_tags) {
                tags_json.emplace_back(Json(tag));
            }
            payload["learning_tags"] = Json(tags_json);
        }
        if (!stats.learning_trace.empty()) {
            payload["learning_trace"] = Json(stats.learning_trace);
        }
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

