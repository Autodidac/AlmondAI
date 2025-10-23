#include "../include/almondai/autopilot.hpp"

#include "../include/almondai/json.hpp"
#include "../include/almondai/retrieval_refresh.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace almondai {

namespace {

const std::regex& forbidden_regex() {
    static const std::regex pattern(
        R"((?i)\b(say nothing|i love you|i['’]m sorry|i apologize|unable to (?:comply|complete|help)|i cannot|i can['’]?t|as an ai(?: language model)?|as a language model|i do not have the ability|policy violation|safety policy|content policy)\b)",
        std::regex::ECMAScript);
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

} // namespace

Autopilot::Autopilot(Trainer& trainer, TokenizerCoordinator& tokenizers)
    : m_trainer(trainer)
    , m_tokenizers(tokenizers)
    , m_tokenizer(tokenizers.bpe())
    , m_training_path("data/training_data.jsonl")
    , m_seed_path("data/training_seed.jsonl")
    , m_eval_path("data/eval_seed.jsonl")
    , m_weights_path("data/student_weights.json") {}

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

bool Autopilot::gate_sample(const TrainingExample& sample) const {
    if (violates_forbidden_regex(sample.teacher_output)) {
        log("Rejected sample during gating: matched forbidden response patterns");
        return false;
    }
    if (contains_pii(sample.teacher_output)) {
        log("Rejected sample during gating: detected possible PII in teacher output");
        return false;
    }
    auto tokens = m_tokenizer.encode(sample.teacher_output);
    tokens.erase(std::remove(tokens.begin(), tokens.end(), BpeTokenizer::PAD_ID), tokens.end());
    tokens.erase(std::remove(tokens.begin(), tokens.end(), BpeTokenizer::EOS_ID), tokens.end());
    if (tokens.size() < 24) {
        log("Rejected sample during gating: teacher output too short after token filtering");
        return false;
    }
    if (max_similarity_against_recent(sample.teacher_output) > 0.92) {
        log("Rejected sample during gating: too similar to recent outputs");
        return false;
    }
    return true;
}

void Autopilot::enqueue_sample(const TrainingExample& sample) {
    const auto ingest = m_tokenizers.ingest_training_pair(m_trainer.model(), sample.prompt, sample.teacher_output);
    if (ingest.word_tokens_added > 0 || ingest.bpe_tokens_added > 0) {
        m_trainer.model().base().save_weights(m_weights_path.string());
        m_tokenizers.persist();
    }
    if (!gate_sample(sample)) {
        return;
    }
    const auto prompt_hash = fnv1a_hash(sample.prompt);
    std::ostringstream hash_stream;
    hash_stream << std::hex << std::uppercase << prompt_hash;
    log("Accepted sample 0x" + hash_stream.str() + ": enqueueing for training");
    m_trainer.append_training_example(sample);
    append_training_record(sample);
    remember_output(sample.teacher_output);
    ++m_pending_since_train;
    maybe_train();
}

void Autopilot::maybe_train() {
    auto batch_size = std::max<std::size_t>(1, m_trainer.options().batch_size);
    while (m_pending_since_train >= 64 && !m_trainer.training_data().empty()) {
        std::vector<TrainingExample> batch;
        const auto& data = m_trainer.training_data();
        for (std::size_t i = 0; i < batch_size && i < data.size(); ++i) {
            batch.push_back(data[data.size() - 1 - i]);
        }
        std::reverse(batch.begin(), batch.end());
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

} // namespace almondai

