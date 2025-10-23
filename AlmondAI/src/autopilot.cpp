#include "../include/almondai/autopilot.hpp"

#include "../include/almondai/json.hpp"
#include "../include/almondai/retrieval_refresh.hpp"
#include "../include/almondai/train.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace almondai {

namespace {

const std::regex& forbidden_regex() {
    static const std::regex pattern(
        R"(\b(say nothing|i love you|i['’]m sorry|i apologize|unable to (comply|complete|help)|i cannot|i can['’]?t|as an ai( language model)?|as a language model|i do not have the ability|policy violation|safety policy|content policy)\b)",
        std::regex::ECMAScript | std::regex::icase);
    return pattern;
}

const std::regex& email_regex() {
    static const std::regex pattern(R"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
                                     std::regex::ECMAScript);
    return pattern;
}

const std::regex& phone_regex() {
    static const std::regex pattern(R"(\b\+?[0-9][0-9\s\-()]{7,}\b)", std::regex::ECMAScript);
    return pattern;
}

const std::regex& private_key_regex() {
    static const std::regex pattern(R"(-----BEGIN|[A-Fa-f0-9]{32,})", std::regex::ECMAScript);
    return pattern;
}

std::string timestamp_now() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const auto time = clock::to_time_t(now);
    std::tm tm {};
#if defined(_WIN32)
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << '.' << std::setw(3) << std::setfill('0') << milliseconds.count();
    return oss.str();
}

std::unordered_set<int> token_set(const BpeTokenizer& tokenizer, const std::string& text) {
    std::unordered_set<int> set;
    auto tokens = tokenizer.encode(text);
    for (int id : tokens) {
        if (id <= BpeTokenizer::PAD_ID || id == BpeTokenizer::EOS_ID) {
            continue;
        }
        set.insert(id);
    }
    return set;
}

std::vector<std::string> extract_tags_from_json(const Json& value) {
    std::vector<std::string> tags;
    if (!value.is_array()) {
        return tags;
    }
    for (const auto& entry : value.as_array()) {
        if (entry.is_string()) {
            tags.push_back(entry.as_string());
        }
    }
    return tags;
}

std::vector<std::string> derive_tags(const TrainingExample& sample) {
    std::unordered_set<std::string> unique;
    if (sample.provenance.is_object()) {
        const auto& provenance = sample.provenance.as_object();
        if (auto tag_it = provenance.find("tags"); tag_it != provenance.end()) {
            for (const auto& tag : extract_tags_from_json(tag_it->second)) {
                unique.insert(tag);
            }
        }
        if (auto src_it = provenance.find("source"); src_it != provenance.end() && src_it->second.is_string()) {
            unique.insert(std::string("source::") + src_it->second.as_string());
        }
        if (auto prompt_it = provenance.find("prompt_hash"); prompt_it != provenance.end() &&
            prompt_it->second.is_string()) {
            unique.insert(prompt_it->second.as_string());
        }
    }
    if (sample.constraints.is_object()) {
        const auto& constraints = sample.constraints.as_object();
        if (auto tag_it = constraints.find("tags"); tag_it != constraints.end()) {
            for (const auto& tag : extract_tags_from_json(tag_it->second)) {
                unique.insert(tag);
            }
        }
        if (auto curriculum_it = constraints.find("curriculum_tag");
            curriculum_it != constraints.end() && curriculum_it->second.is_string()) {
            unique.insert(curriculum_it->second.as_string());
        }
    }
    if (unique.empty()) {
        unique.insert("curriculum::general");
    }
    return std::vector<std::string>(unique.begin(), unique.end());
}

} // namespace

Autopilot::Autopilot(Trainer& trainer, TokenizerCoordinator& tokenizers)
    : m_trainer(trainer)
    , m_tokenizers(tokenizers)
    , m_tokenizer(tokenizers.bpe())
    , m_training_path("data/training_data.jsonl")
    , m_seed_path("data/training_seed.jsonl")
    , m_eval_path("data/eval_seed.jsonl")
    , m_weights_path("data/student_weights.json")
    , m_mutation_ledger_path("data/mutation_ledger.jsonl") {}

void Autopilot::run() {
    log("Starting autopilot run");
    harvest_from_seed_files();
    warmup_if_needed();
    maybe_train();
    maybe_evaluate();
    log("Autopilot run complete");
}

void Autopilot::warmup_if_needed() {
    if (std::filesystem::exists(m_weights_path)) {
        log("Skipping warmup; existing weights found at " + m_weights_path.string());
        return;
    }
    const auto seed = load_jsonl(m_seed_path);
    if (seed.empty()) {
        log("Skipping warmup; no seed data found at " + m_seed_path.string());
        return;
    }
    log("Running warmup on " + std::to_string(seed.size()) + " seed samples");
    run_warmup_epochs(seed);
    rebuild_retrieval_index(seed);
    log("Warmup complete; retrieval index rebuilt from seed data");
}

void Autopilot::run_warmup_epochs(const std::vector<TrainingExample>& seed_data) {
    if (seed_data.empty()) {
        return;
    }
    auto options = m_trainer.options();
    Trainer::Options warm_options = options;
    warm_options.batch_size = 32;
    m_trainer.set_options(warm_options);
    for (int epoch = 0; epoch < 3; ++epoch) {
        log("Warmup epoch " + std::to_string(epoch + 1) + "/3 started");
        for (std::size_t offset = 0; offset < seed_data.size(); offset += warm_options.batch_size) {
            std::vector<TrainingExample> batch;
            const std::size_t end = std::min(seed_data.size(), offset + warm_options.batch_size);
            batch.insert(batch.end(), seed_data.begin() + static_cast<std::ptrdiff_t>(offset),
                         seed_data.begin() + static_cast<std::ptrdiff_t>(end));
            auto report = m_trainer.train_on_batch(batch);
            std::ostringstream oss;
            oss << "Warmup step " << report.step << " trained on " << report.tokens << " tokens (loss="
                << std::fixed << std::setprecision(4) << report.loss << ", ppl=" << std::setprecision(3)
                << report.perplexity << ')';
            log(oss.str());
        }
        log("Warmup epoch " + std::to_string(epoch + 1) + "/3 finished");
    }
    m_trainer.set_options(options);
    m_trainer.save_checkpoint();
    log("Warmup checkpoint saved");
}

std::vector<TrainingExample> Autopilot::load_jsonl(const std::filesystem::path& path) const {
    std::vector<TrainingExample> data;
    std::ifstream file(path);
    if (!file) {
        return data;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        try {
            Json parsed = Json::parse(line);
            if (!parsed.is_object()) {
                continue;
            }
            const auto& obj = parsed.as_object();
            TrainingExample example;
            if (auto it = obj.find("constraints"); it != obj.end()) {
                example.constraints = it->second;
            } else {
                example.constraints = Json(JsonObject{});
            }
            if (auto it = obj.find("prompt"); it != obj.end() && it->second.is_string()) {
                example.prompt = it->second.as_string();
            }
            if (auto it = obj.find("provenance"); it != obj.end()) {
                example.provenance = it->second;
            } else {
                example.provenance = Json(JsonObject{});
            }
            if (auto it = obj.find("teacher_output"); it != obj.end() && it->second.is_string()) {
                example.teacher_output = it->second.as_string();
            }
            data.push_back(std::move(example));
        } catch (...) {
        }
    }
    return data;
}

void Autopilot::append_training_record(const TrainingExample& sample) {
    std::filesystem::create_directories(m_training_path.parent_path());
    std::ofstream file(m_training_path, std::ios::app);
    if (!file) {
        log("Failed to append training record: unable to open " + m_training_path.string());
        return;
    }
    JsonObject obj;
    obj["constraints"] = sample.constraints;
    obj["prompt"] = Json(sample.prompt);
    obj["provenance"] = sample.provenance;
    obj["teacher_output"] = Json(sample.teacher_output);
    file << Json(obj).dump() << '\n';
}

bool Autopilot::violates_forbidden_regex(const std::string& text) const {
    return std::regex_search(text, forbidden_regex());
}

bool Autopilot::contains_pii(const std::string& text) const {
    return std::regex_search(text, email_regex()) ||
           std::regex_search(text, phone_regex()) ||
           std::regex_search(text, private_key_regex());
}

double Autopilot::max_similarity_against_recent(const std::string& text) const {
    if (m_recent_outputs.empty()) {
        return 0.0;
    }
    const auto candidate = token_set(m_tokenizer, text);
    if (candidate.empty()) {
        return 0.0;
    }
    double max_similarity = 0.0;
    for (const auto& previous : m_recent_outputs) {
        const auto other = token_set(m_tokenizer, previous);
        if (other.empty()) {
            continue;
        }
        std::size_t intersection = 0;
        for (int token : candidate) {
            if (other.find(token) != other.end()) {
                ++intersection;
            }
        }
        std::size_t union_size = candidate.size() + other.size() - intersection;
        if (union_size == 0) {
            continue;
        }
        double similarity = static_cast<double>(intersection) / static_cast<double>(union_size);
        max_similarity = std::max(max_similarity, similarity);
        if (max_similarity > 0.92) {
            break;
        }
    }
    return max_similarity;
}

void Autopilot::remember_output(const std::string& text) {
    if (text.empty()) {
        return;
    }
    m_recent_outputs.push_back(text);
    while (m_recent_outputs.size() > 512) {
        m_recent_outputs.pop_front();
    }
}

std::uint64_t Autopilot::fnv1a_hash(const std::string& text) const {
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : text) {
        hash ^= static_cast<std::uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

Autopilot::GateDecision Autopilot::gate_sample(const TrainingExample& sample) const {
    GateDecision decision;
    decision.accepted = true;

    if (m_policy_governor) {
        decision.governor_report = m_policy_governor->validate_output(sample.teacher_output, sample.constraints);
        if (!decision.governor_report.allowed) {
            decision.accepted = false;
            for (const auto& violation : decision.governor_report.violations) {
                decision.reasons.emplace_back("policy:" + violation);
            }
        }
    }

    decision.regex_violation = violates_forbidden_regex(sample.teacher_output);
    if (decision.regex_violation) {
        decision.accepted = false;
        decision.reasons.emplace_back("safety:forbidden_pattern");
    }

    decision.contains_pii = contains_pii(sample.teacher_output);
    if (decision.contains_pii) {
        decision.accepted = false;
        decision.reasons.emplace_back("safety:pii_detected");
    }

    auto tokens = m_tokenizer.encode(sample.teacher_output);
    tokens.erase(std::remove(tokens.begin(), tokens.end(), BpeTokenizer::PAD_ID), tokens.end());
    tokens.erase(std::remove(tokens.begin(), tokens.end(), BpeTokenizer::EOS_ID), tokens.end());
    decision.filtered_tokens = tokens.size();
    if (decision.filtered_tokens < 24) {
        decision.accepted = false;
        decision.reasons.emplace_back("quality:output_too_short");
    }

    decision.similarity = max_similarity_against_recent(sample.teacher_output);

    const auto tags = sample_tags(sample);
    double similarity_threshold = 0.92;
    if (!m_curriculum_priority.empty()) {
        for (const auto& tag : tags) {
            if (std::find(m_curriculum_priority.begin(), m_curriculum_priority.end(), tag) != m_curriculum_priority.end()) {
                similarity_threshold = 0.96;
                break;
            }
        }
    }
    if (decision.similarity > similarity_threshold) {
        decision.accepted = false;
        decision.reasons.emplace_back("quality:recent_similarity");
    }

    double token_score = decision.filtered_tokens >= 48
                              ? 1.0
                              : static_cast<double>(decision.filtered_tokens) / 48.0;
    double novelty_score = 1.0 - decision.similarity;
    decision.quality_score = std::max(0.0, token_score * novelty_score);
    if (decision.contains_pii || decision.regex_violation || !decision.governor_report.allowed) {
        decision.quality_score = 0.0;
    }

    if (decision.quality_score < m_quality_floor) {
        decision.accepted = false;
        decision.reasons.emplace_back("quality:score_below_floor");
    }

    if (!decision.accepted && decision.reasons.empty()) {
        decision.reasons.emplace_back("quality:unspecified");
    }

    if (!decision.accepted) {
        std::ostringstream oss;
        oss << "Rejected sample during gating: ";
        for (std::size_t i = 0; i < decision.reasons.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << decision.reasons[i];
        }
        log(oss.str());
    }

    return decision;
}

void Autopilot::enqueue_sample(const TrainingExample& sample) {
    const auto ingest = m_tokenizers.ingest_training_pair(m_trainer.model(), sample.prompt, sample.teacher_output);
    if (ingest.word_tokens_added > 0 || ingest.bpe_tokens_added > 0) {
        m_trainer.model().base().save_weights(m_weights_path.string());
        m_tokenizers.persist();
    }
    auto decision = gate_sample(sample);
    record_mutation_decision(sample, decision);
    if (!decision.accepted) {
        return;
    }
    const auto prompt_hash = fnv1a_hash(sample.prompt);
    std::ostringstream hash_stream;
    hash_stream << std::hex << std::uppercase << prompt_hash;
    std::ostringstream oss;
    oss << "Accepted sample 0x" << hash_stream.str()
        << ": enqueueing for training (quality=" << std::fixed << std::setprecision(3) << decision.quality_score
        << ", similarity=" << std::setprecision(3) << decision.similarity << ")";
    log(oss.str());
    m_trainer.append_training_example(sample);
    append_training_record(sample);
    remember_output(sample.teacher_output);
    ingest_into_continuous_learner(sample, decision);
    if (m_retrieval_hook) {
        m_retrieval_hook(sample, decision);
    }
    ++m_pending_since_train;
    maybe_train();
}

void Autopilot::maybe_train() {
    auto batch_size = std::max<std::size_t>(1, m_trainer.options().batch_size);
    while (m_pending_since_train >= 64 && !m_trainer.training_data().empty()) {
        auto batch = select_training_batch(batch_size);
        if (batch.empty()) {
            break;
        }
        auto report = m_trainer.train_on_batch(batch);
        std::ostringstream oss;
        oss << "Training step " << report.step << " processed " << report.tokens << " tokens (loss="
            << std::fixed << std::setprecision(4) << report.loss << ", ppl=" << std::setprecision(3)
            << report.perplexity << ')';
        if (report.checkpoint_saved) {
            oss << " [checkpoint saved]";
            m_tokenizers.persist(report.step);
        }
        log(oss.str());
        if (m_pending_since_train >= 64) {
            m_pending_since_train -= 64;
        } else {
            m_pending_since_train = 0;
        }
    }
}

void Autopilot::maybe_evaluate() {
    if (m_trainer.eval_dataset().empty()) {
        return;
    }
    if (m_trainer.step() < m_last_eval_step + 200) {
        return;
    }
    auto report = m_trainer.evaluate(m_trainer.eval_dataset());
    m_last_eval_step = m_trainer.step();
    if (report.tokens > 0) {
        promote_if_improved(report.perplexity);
        update_curriculum(report);
        std::ostringstream oss;
        oss << "Evaluation at step " << m_last_eval_step << " processed " << report.tokens << " tokens (loss="
            << std::fixed << std::setprecision(4) << report.loss << ", ppl=" << std::setprecision(3)
            << report.perplexity << ')';
        log(oss.str());
    }
}

void Autopilot::promote_if_improved(double perplexity) {
    if (!std::isfinite(perplexity)) {
        return;
    }
    if (m_best_eval_perplexity == std::numeric_limits<double>::infinity() ||
        perplexity <= m_best_eval_perplexity * 0.98) {
        m_best_eval_perplexity = perplexity;
        m_trainer.save_checkpoint();
        std::ostringstream oss;
        oss << "Promoted new best checkpoint with perplexity " << std::fixed << std::setprecision(3) << perplexity;
        log(oss.str());
    }
}

void Autopilot::rebuild_retrieval_index(const std::vector<TrainingExample>& dataset) const {
    refresh_retrieval_index(dataset, m_tokenizer);
}

void Autopilot::harvest_from_seed_files() {
    log("Harvesting training and seed datasets");
    auto existing = load_jsonl(m_training_path);
    log("Loaded " + std::to_string(existing.size()) + " existing training samples");
    const auto seed = load_jsonl(m_seed_path);
    log("Loaded " + std::to_string(seed.size()) + " seed samples");
    if (existing.empty() && !seed.empty()) {
        existing = seed;
        std::ofstream reset(m_training_path, std::ios::trunc);
        for (const auto& sample : existing) {
            JsonObject obj;
            obj["constraints"] = sample.constraints;
            obj["prompt"] = Json(sample.prompt);
            obj["provenance"] = sample.provenance;
            obj["teacher_output"] = Json(sample.teacher_output);
            reset << Json(obj).dump() << '\n';
        }
        log("Bootstrapped training dataset with seed samples");
    }
    for (const auto& sample : existing) {
        const auto added = m_tokenizers.ingest_training_pair(m_trainer.model(), sample.prompt, sample.teacher_output);
        if (added.word_tokens_added > 0 || added.bpe_tokens_added > 0) {
            m_trainer.model().base().save_weights(m_weights_path.string());
            m_tokenizers.persist();
        }
        m_trainer.append_training_example(sample);
        remember_output(sample.teacher_output);
        GateDecision accepted;
        accepted.accepted = true;
        accepted.filtered_tokens = m_tokenizer.encode(sample.teacher_output).size();
        accepted.similarity = 0.0;
        accepted.quality_score = 1.0;
        record_mutation_decision(sample, accepted);
        ingest_into_continuous_learner(sample, accepted);
        if (m_retrieval_hook) {
            m_retrieval_hook(sample, accepted);
        }
    }
    if (!existing.empty()) {
        rebuild_retrieval_index(existing);
        log("Loaded existing training samples into trainer and refreshed retrieval index");
    }
    auto eval_data = load_jsonl(m_eval_path);
    if (!eval_data.empty()) {
        m_trainer.set_eval_dataset(eval_data);
        log("Loaded " + std::to_string(eval_data.size()) + " evaluation samples");
    }

    if (m_teacher) {
        log("Harvesting fresh teacher responses for seed prompts");
        for (const auto& prompt : seed) {
            TrainingExample generated = prompt;
            auto response = m_teacher(prompt);
            if (!response || response->empty()) {
                log("Teacher returned no output for a seed prompt");
                continue;
            }
            generated.teacher_output = *response;
            enqueue_sample(generated);
        }
        log("Finished harvesting teacher responses");
    }
}

void Autopilot::log(std::string_view message) const {
    std::cout << "[Autopilot " << timestamp_now() << "] " << message << std::endl;
}

std::vector<std::string> Autopilot::sample_tags(const TrainingExample& sample) const {
    return derive_tags(sample);
}

std::string Autopilot::derive_prompt_identifier(const TrainingExample& sample, const std::string& fallback) const {
    if (sample.provenance.is_object()) {
        const auto& provenance = sample.provenance.as_object();
        if (auto it = provenance.find("sample_hash"); it != provenance.end() && it->second.is_string()) {
            return it->second.as_string();
        }
        if (auto it = provenance.find("prompt_hash"); it != provenance.end() && it->second.is_string()) {
            return it->second.as_string();
        }
    }
    return fallback;
}

void Autopilot::record_mutation_decision(const TrainingExample& sample, const GateDecision& decision) {
    if (m_mutation_callback) {
        m_mutation_callback(sample, decision);
    }
    if (m_mutation_ledger_path.empty()) {
        return;
    }
    std::filesystem::create_directories(m_mutation_ledger_path.parent_path());
    std::ofstream ledger(m_mutation_ledger_path, std::ios::app);
    if (!ledger) {
        return;
    }

    std::ostringstream fallback_hash;
    fallback_hash << std::hex << std::uppercase << fnv1a_hash(sample.prompt);
    JsonObject entry;
    entry["timestamp"] = Json(timestamp_now());
    entry["accepted"] = Json(decision.accepted);
    entry["quality_score"] = Json(decision.quality_score);
    entry["similarity"] = Json(decision.similarity);
    entry["filtered_tokens"] = Json(static_cast<double>(decision.filtered_tokens));
    entry["pii_detected"] = Json(decision.contains_pii);
    entry["regex_violation"] = Json(decision.regex_violation);
    entry["prompt_hash"] = Json(derive_prompt_identifier(sample, fallback_hash.str()));

    JsonArray tags_json;
    for (const auto& tag : sample_tags(sample)) {
        tags_json.emplace_back(Json(tag));
    }
    entry["tags"] = Json(tags_json);

    if (sample.provenance.is_object()) {
        const auto& provenance = sample.provenance.as_object();
        if (auto source_it = provenance.find("source"); source_it != provenance.end() && source_it->second.is_string()) {
            entry["teacher_source"] = Json(source_it->second.as_string());
        }
    }

    JsonArray reasons_json;
    for (const auto& reason : decision.reasons) {
        reasons_json.emplace_back(Json(reason));
    }
    entry["reasons"] = Json(reasons_json);

    JsonArray governor_json;
    for (const auto& violation : decision.governor_report.violations) {
        governor_json.emplace_back(Json(violation));
    }
    entry["governor_violations"] = Json(governor_json);

    ledger << Json(entry).dump() << '\n';
}

void Autopilot::ingest_into_continuous_learner(const TrainingExample& sample, const GateDecision&) {
    if (!m_continuous_learner) {
        return;
    }
    std::ostringstream fallback_hash;
    fallback_hash << std::hex << std::uppercase << fnv1a_hash(sample.prompt);
    std::string prompt_identifier = derive_prompt_identifier(sample, fallback_hash.str());
    std::string teacher_source;
    if (sample.provenance.is_object()) {
        const auto& provenance = sample.provenance.as_object();
        if (auto source_it = provenance.find("source"); source_it != provenance.end() && source_it->second.is_string()) {
            teacher_source = source_it->second.as_string();
        }
    }
    m_continuous_learner->ingest(sample.prompt,
                                  sample.teacher_output,
                                  sample.constraints,
                                  prompt_identifier,
                                  teacher_source);
}

std::vector<TrainingExample> Autopilot::select_training_batch(std::size_t batch_size) const {
    std::vector<TrainingExample> batch;
    const auto& data = m_trainer.training_data();
    if (batch_size == 0 || data.empty()) {
        return batch;
    }

    std::vector<std::size_t> indices;
    indices.reserve(batch_size);
    std::unordered_set<std::size_t> used;

    if (!m_curriculum_priority.empty()) {
        for (const auto& tag : m_curriculum_priority) {
            if (indices.size() >= batch_size) {
                break;
            }
            for (std::size_t offset = 0; offset < data.size(); ++offset) {
                std::size_t idx = data.size() - 1 - offset;
                if (used.count(idx)) {
                    continue;
                }
                auto tags = derive_tags(data[idx]);
                if (std::find(tags.begin(), tags.end(), tag) != tags.end()) {
                    indices.push_back(idx);
                    used.insert(idx);
                    break;
                }
            }
        }
    }

    for (std::size_t offset = 0; offset < data.size() && indices.size() < batch_size; ++offset) {
        std::size_t idx = data.size() - 1 - offset;
        if (used.count(idx)) {
            continue;
        }
        indices.push_back(idx);
        used.insert(idx);
    }

    std::sort(indices.begin(), indices.end());
    for (std::size_t idx : indices) {
        batch.push_back(data[idx]);
    }
    return batch;
}

void Autopilot::update_curriculum(const EvaluationReport& report) {
    if (report.tokens == 0) {
        return;
    }
    double previous_floor = m_quality_floor;
    double adjusted_floor = m_quality_floor;
    if (std::isfinite(report.perplexity) &&
        m_best_eval_perplexity != std::numeric_limits<double>::infinity()) {
        if (report.perplexity > m_best_eval_perplexity * 1.02) {
            adjusted_floor = std::min(0.6, adjusted_floor + 0.05);
        } else if (report.perplexity < m_best_eval_perplexity * 0.98) {
            adjusted_floor = std::max(0.2, adjusted_floor - 0.02);
        }
    }
    bool floor_changed = std::fabs(adjusted_floor - previous_floor) > 1e-6;
    m_quality_floor = adjusted_floor;

    std::vector<std::pair<std::string, double>> ranked;
    for (const auto& [tag, perplexity] : report.tag_perplexity) {
        auto count_it = report.tag_token_counts.find(tag);
        if (count_it == report.tag_token_counts.end() || count_it->second < 4) {
            continue;
        }
        if (report.perplexity > 0.0 && perplexity > report.perplexity * 1.05) {
            double ratio = perplexity / std::max(report.perplexity, 1e-6);
            ranked.emplace_back(tag, ratio);
        }
    }

    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second > rhs.second;
    });

    std::vector<std::string> prioritized;
    for (const auto& [tag, _] : ranked) {
        prioritized.push_back(tag);
        if (prioritized.size() >= 8) {
            break;
        }
    }

    if (prioritized != m_curriculum_priority) {
        m_curriculum_priority = prioritized;
        std::ostringstream oss;
        if (!m_curriculum_priority.empty()) {
            oss << "Curriculum prioritizing tags: ";
            for (std::size_t i = 0; i < m_curriculum_priority.size(); ++i) {
                if (i > 0) {
                    oss << ", ";
                }
                oss << m_curriculum_priority[i];
            }
        } else {
            oss << "Curriculum reset; no underperforming tags detected";
        }
        oss << " (quality_floor=" << std::fixed << std::setprecision(3) << m_quality_floor << ")";
        log(oss.str());
    } else if (floor_changed) {
        std::ostringstream oss;
        oss << "Quality floor adjusted to " << std::fixed << std::setprecision(3) << m_quality_floor;
        log(oss.str());
    }
}

} // namespace almondai

