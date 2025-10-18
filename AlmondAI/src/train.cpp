#include "../AlmondAI/include/almondai/train.hpp"

#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <sstream>
#include <functional>
#include <system_error>

namespace almondai {

namespace {
const std::filesystem::path kTrainingDataPath{"data/training_data.jsonl"};
const std::filesystem::path kSeedDataPath{"data/training_seed.jsonl"};
const std::filesystem::path kVocabPath{"data/vocab.txt"};
const std::filesystem::path kWeightsPath{"data/student_weights.json"};
}

ContinuousLearner::ContinuousLearner(StudentModel student,
                                     AdapterManager adapters,
                                     WordTokenizer tokenizer,
                                     PolicyGovernor governor)
    : m_student(std::move(student)),
      m_adapters(std::move(adapters)),
      m_tokenizer(std::move(tokenizer)),
      m_retrieval(m_tokenizer),
      m_evaluator(m_tokenizer),
      m_governor(std::move(governor)) {
    m_log_file.open("data/training_log.txt", std::ios::app);
    if (m_log_file.tellp() == 0) {
        m_log_file << "AlmondAI training log\n";
        m_log_file << "Each entry below records a single training or evaluation step in a human-readable format.\n\n";
    }
    load_persistent_data();
}

std::optional<CuratedSample> ContinuousLearner::ingest(const std::string& prompt,
                                                       const std::string& teacher_output,
                                                       Json constraints,
                                                       const std::string& prompt_hash) {
    auto curated = m_curator.curate(prompt, teacher_output, std::move(constraints), prompt_hash);
    if (!curated) {
        return std::nullopt;
    }
    const std::size_t before_vocab = m_tokenizer.vocab().size();
    m_tokenizer.build_vocab({curated->prompt, curated->teacher_output});
    const std::size_t after_vocab = m_tokenizer.vocab().size();
    if (after_vocab > before_vocab) {
        m_student.base().resize_vocab(after_vocab);
        m_tokenizer.save_vocab(kVocabPath.string());
    }
    m_training_data.push_back(*curated);
    if (m_eval_data.size() < 16) {
        m_eval_data.push_back(*curated);
    }
    m_retrieval.ingest_document(prompt_hash, teacher_output);
    persist_sample(*curated);
    return curated;
}

TrainingStats ContinuousLearner::train_step(const CuratedSample& sample) {
    ++m_step;
    TrainingStats stats;
    stats.step = m_step;

    auto tokens = m_tokenizer.encode(sample.prompt);
    auto logits = m_student.forward(tokens);
    auto teacher_tokens = m_tokenizer.encode(sample.teacher_output);
    const int target = teacher_tokens.size() > 1 ? teacher_tokens[1] : 0;
    const double predicted = logits[target % logits.size()];
    const auto max_it = std::max_element(logits.begin(), logits.end());
    const double max_logit = *max_it;
    const double error = max_logit - predicted;

    std::vector<double> gradient(m_student.base().config().hidden_size, error);
    m_student.update(gradient);
    if (Adapter* active = m_adapters.active_adapter()) {
        active->apply_gradient(gradient);
        active->update_statistics(gradient);
        stats.adapter_norm = active->norm();
    }

    std::vector<int> decoded{static_cast<int>(std::distance(logits.begin(), max_it))};
    std::string student_output = m_tokenizer.decode(decoded);
    m_curator.record_student_response(sample.prompt, student_output, sample);

    stats.loss = error * error;
    stats.accuracy = max_logit <= predicted ? 1.0 : 0.0;
    stats.retrieval_hit_rate = m_retrieval.hit_rate();
    log_stats(stats);
    m_student.base().save_weights(kWeightsPath.string());

    return stats;
}

TrainingStats ContinuousLearner::evaluate_canary() {
    TrainingStats stats;
    if (m_eval_data.empty()) {
        return stats;
    }
    const auto metrics = m_evaluator.evaluate(m_student, m_eval_data);
    stats.step = m_step;
    stats.loss = metrics.loss;
    stats.accuracy = metrics.accuracy;
    stats.retrieval_hit_rate = m_retrieval.hit_rate();
    if (const Adapter* adapter = m_adapters.active_adapter()) {
        stats.adapter_norm = adapter->norm();
    }
    log_stats(stats);
    return stats;
}

void ContinuousLearner::fit(const std::string& path,
                            int epochs,
                            int batch,
                            std::function<void(int, double, double, double)> on_batch) {
    (void)path; // Stub implementation does not yet read from disk.

    if (!on_batch) {
        return;
    }

    const int safe_epochs = std::max(1, epochs);
    const int safe_batch = std::max(1, batch);
    const int dataset_size = static_cast<int>(m_training_data.size());
    const int steps_per_epoch = dataset_size > 0
        ? std::max(1, (dataset_size + safe_batch - 1) / safe_batch)
        : 5;

    const int total_steps = safe_epochs * steps_per_epoch;
    double simulated_loss = 2.0;
    const double base_lr = 5e-4;
    double tokens_per_second = 12000.0;

    for (int epoch = 0; epoch < safe_epochs; ++epoch) {
        for (int step_idx = 0; step_idx < steps_per_epoch; ++step_idx) {
            const int global_step = epoch * steps_per_epoch + step_idx + 1;
            simulated_loss = std::max(0.01, simulated_loss * 0.97);
            const double progress = static_cast<double>(global_step) / static_cast<double>(total_steps);
            const double current_lr = base_lr * (0.5 + 0.5 * (1.0 - progress));
            tokens_per_second = 10000.0 + 250.0 * static_cast<double>(global_step);

            on_batch(global_step, simulated_loss, current_lr, tokens_per_second);
        }
    }
}

void ContinuousLearner::promote_adapter(const std::string& name) {
    m_adapters.activate(name);
    if (const Adapter* adapter = m_adapters.active_adapter()) {
        m_student.base().attach_adapter(adapter);
    }
}

void ContinuousLearner::rollback_adapter() {
    m_adapters.deactivate();
    m_student.base().attach_adapter(nullptr);
}

void ContinuousLearner::log_stats(const TrainingStats& stats) {
    if (!m_log_file.is_open()) {
        return;
    }
    m_log_file << "Step " << stats.step
               << " | loss=" << std::fixed << std::setprecision(6) << stats.loss
               << " | accuracy=" << stats.accuracy
               << " | adapter_norm=" << stats.adapter_norm
               << " | retrieval_hit_rate=" << stats.retrieval_hit_rate
               << '\n';
}

void ContinuousLearner::load_persistent_data() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(kTrainingDataPath.parent_path(), ec);

    if (fs::exists(kWeightsPath)) {
        m_student.base().load_weights(kWeightsPath.string());
    }

    if (!fs::exists(kTrainingDataPath) && fs::exists(kSeedDataPath)) {
        fs::copy_file(kSeedDataPath, kTrainingDataPath, fs::copy_options::overwrite_existing, ec);
    }

    load_samples_from_file(kTrainingDataPath);
}

void ContinuousLearner::load_samples_from_file(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        return;
    }
    std::ifstream file(path);
    if (!file) {
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
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
            auto output_it = obj.find("teacher_output");
            if (prompt_it == obj.end() || output_it == obj.end()) {
                continue;
            }
            if (!prompt_it->second.is_string() || !output_it->second.is_string()) {
                continue;
            }
            CuratedSample sample;
            sample.prompt = prompt_it->second.as_string();
            sample.teacher_output = output_it->second.as_string();
            if (auto constraints_it = obj.find("constraints"); constraints_it != obj.end()) {
                sample.constraints = constraints_it->second;
            }
            if (auto provenance_it = obj.find("provenance"); provenance_it != obj.end()) {
                sample.provenance = provenance_it->second;
            } else {
                sample.provenance = JsonObject{};
            }
            const std::size_t before_vocab = m_tokenizer.vocab().size();
            m_tokenizer.build_vocab({sample.prompt, sample.teacher_output});
            if (m_tokenizer.vocab().size() > before_vocab) {
                m_student.base().resize_vocab(m_tokenizer.vocab().size());
            }
            const std::size_t index = m_training_data.size();
            const std::string document_id = derive_document_id(sample, index);
            if (!document_id.empty()) {
                m_curator.mark_seen(document_id);
            }
            m_training_data.push_back(sample);
            if (m_eval_data.size() < 16) {
                m_eval_data.push_back(sample);
            }
            m_retrieval.ingest_document(document_id, sample.teacher_output);
        } catch (...) {
            continue;
        }
    }
    if (!m_training_data.empty()) {
        m_tokenizer.save_vocab(kVocabPath.string());
    }
}

void ContinuousLearner::persist_sample(const CuratedSample& sample) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(kTrainingDataPath.parent_path(), ec);
    std::ofstream file(kTrainingDataPath, std::ios::app);
    if (!file) {
        return;
    }
    JsonObject obj;
    obj["prompt"] = Json(sample.prompt);
    obj["teacher_output"] = Json(sample.teacher_output);
    obj["constraints"] = sample.constraints;
    obj["provenance"] = sample.provenance;
    file << Json(obj).dump() << '\n';
}

std::string ContinuousLearner::derive_document_id(const CuratedSample& sample, std::size_t index) const {
    if (sample.provenance.is_object()) {
        const auto& prov = sample.provenance.as_object();
        if (auto it = prov.find("prompt_hash"); it != prov.end() && it->second.is_string()) {
            const std::string& value = it->second.as_string();
            if (!value.empty()) {
                return value;
            }
        }
    }
    std::hash<std::string> hasher;
    std::ostringstream oss;
    oss << "sample:" << index << ':' << hasher(sample.prompt + sample.teacher_output);
    return oss.str();
}

} // namespace almondai

