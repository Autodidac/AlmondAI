#include "../include/almondai/train.hpp"

#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <sstream>
#include <functional>
#include <system_error>
#include <fstream>
#include <chrono>
#include <random>

namespace almondai {

namespace {
const std::filesystem::path kTrainingDataPath{"data/training_data.jsonl"};
const std::filesystem::path kSeedDataPath{"data/training_seed.jsonl"};
const std::filesystem::path kVocabPath{"data/vocab.txt"};
const std::filesystem::path kWeightsPath{"data/student_weights.json"};
const std::filesystem::path kSeedTextPath{"data/seed.txt"};

std::optional<CuratedSample> parse_sample_line(const std::string& line) {
    if (line.empty()) {
        return std::nullopt;
    }
    try {
        Json record = Json::parse(line);
        if (!record.is_object()) {
            return std::nullopt;
        }
        const auto& obj = record.as_object();
        auto prompt_it = obj.find("prompt");
        auto output_it = obj.find("teacher_output");
        if (prompt_it == obj.end() || output_it == obj.end()) {
            return std::nullopt;
        }
        if (!prompt_it->second.is_string() || !output_it->second.is_string()) {
            return std::nullopt;
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
        return sample;
    } catch (...) {
        return std::nullopt;
    }
}

std::string ensure_seed_text() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(kSeedTextPath.parent_path(), ec);

    bool need_default = true;
    if (fs::exists(kSeedTextPath, ec) && !ec) {
        const auto size = fs::file_size(kSeedTextPath, ec);
        if (!ec && size > 0) {
            need_default = false;
        }
    }

    if (need_default) {
        std::ofstream out(kSeedTextPath, std::ios::trunc);
        out << "AlmondAI is a modular research assistant engineered to blend local learning, retrieval augmented generation, and safety controls in a compact runtime. "
               "It keeps a vocabulary, curated examples, and adapters that can be hot-swapped without downtime. "
               "Respond with a concise, friendly description of AlmondAI's purpose and subsystems.";
    }

    std::ifstream in(kSeedTextPath);
    if (!in) {
        return std::string();
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}
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
                                                       const std::string& prompt_hash,
                                                       const std::string& teacher_source) {
    auto curated = m_curator.curate(prompt, teacher_output, std::move(constraints), prompt_hash, teacher_source);
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
    const std::size_t index = m_training_data.size() - 1;
    if (m_eval_data.size() < 16) {
        m_eval_data.push_back(*curated);
    }
    const std::string document_id = derive_document_id(m_training_data.back(), index);
    if (!document_id.empty()) {
        if (m_training_data.back().provenance.is_object()) {
            auto& prov = m_training_data.back().provenance.as_object();
            if (prov.find("sample_hash") == prov.end()) {
                prov["sample_hash"] = Json(document_id);
            }
        }
        m_retrieval.ingest_document(document_id, teacher_output);
    } else {
        m_retrieval.ingest_document(prompt_hash, teacher_output);
    }
    persist_sample(m_training_data.back());
    return m_training_data.back();
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
    if (sample.provenance.is_object()) {
        const auto& prov = sample.provenance.as_object();
        if (auto src_it = prov.find("source"); src_it != prov.end() && src_it->second.is_string()) {
            stats.teacher_source = src_it->second.as_string();
        }
    }
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
    stats.teacher_source = "evaluation";
    log_stats(stats);
    return stats;
}

void ContinuousLearner::fit(const std::string& path,
                            int epochs,
                            int batch,
                            std::function<void(int, double, double, double)> on_batch) {
    const int safe_epochs = std::max(1, epochs);
    const int safe_batch = std::max(1, batch);

    std::vector<CuratedSample> dataset = m_training_data;

    if (!path.empty()) {
        std::ifstream file(path);
        if (file) {
            std::string line;
            while (std::getline(file, line)) {
                if (auto sample = parse_sample_line(line)) {
                    const std::size_t before_vocab = m_tokenizer.vocab().size();
                    m_tokenizer.build_vocab({sample->prompt, sample->teacher_output});
                    if (m_tokenizer.vocab().size() > before_vocab) {
                        m_student.base().resize_vocab(m_tokenizer.vocab().size());
                        m_tokenizer.save_vocab(kVocabPath.string());
                    }
                    dataset.push_back(*sample);
                }
            }
        }
    }

    if (dataset.empty()) {
        return;
    }

    std::mt19937 rng(std::random_device{}());
    const int steps_per_epoch = std::max(1, static_cast<int>((dataset.size() + safe_batch - 1) / safe_batch));
    const double base_lr = m_student.base().config().learning_rate;

    int global_step = 0;
    for (int epoch = 0; epoch < safe_epochs; ++epoch) {
        std::shuffle(dataset.begin(), dataset.end(), rng);
        for (std::size_t offset = 0; offset < dataset.size(); offset += static_cast<std::size_t>(safe_batch)) {
            const std::size_t end = std::min(dataset.size(), offset + static_cast<std::size_t>(safe_batch));
            if (end <= offset) {
                continue;
            }
            const auto batch_start_time = std::chrono::steady_clock::now();
            double loss_sum = 0.0;
            std::size_t token_count = 0;
            for (std::size_t i = offset; i < end; ++i) {
                token_count += m_tokenizer.encode(dataset[i].prompt).size();
                TrainingStats stats = train_step(dataset[i]);
                loss_sum += stats.loss;
            }
            const auto batch_end_time = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed = batch_end_time - batch_start_time;
            const double tokens_per_second = elapsed.count() > 0.0
                ? static_cast<double>(token_count) / elapsed.count()
                : 0.0;

            ++global_step;
            if (on_batch) {
                const double average_loss = loss_sum / static_cast<double>(end - offset);
                const double schedule = 0.5 + 0.5 * (1.0 - (static_cast<double>(global_step - 1) / static_cast<double>(safe_epochs * steps_per_epoch)));
                const double current_lr = base_lr * schedule;
                on_batch(global_step, average_loss, current_lr, tokens_per_second);
            }
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
               << " | teacher_source=" << (stats.teacher_source.empty() ? std::string{"unknown"} : stats.teacher_source)
               << '\n';
    m_log_file.flush();
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

    if (m_training_data.empty()) {
        const std::string seed_text = ensure_seed_text();
        if (!seed_text.empty()) {
            CuratedSample seed_sample;
            seed_sample.prompt = "Introduce AlmondAI to a new user.";
            seed_sample.teacher_output = seed_text;
            seed_sample.constraints = Json(JsonObject{});
            JsonObject provenance;
            provenance["source"] = Json("seed");
            provenance["prompt_hash"] = Json("seed::bootstrap");
            provenance["teacher_hash"] = Json(std::to_string(std::hash<std::string>{}(seed_sample.teacher_output)));
            seed_sample.provenance = Json(provenance);

            const std::size_t before_vocab = m_tokenizer.vocab().size();
            m_tokenizer.build_vocab({seed_sample.prompt, seed_sample.teacher_output});
            if (m_tokenizer.vocab().size() > before_vocab) {
                m_student.base().resize_vocab(m_tokenizer.vocab().size());
            }

            m_training_data.push_back(seed_sample);
            if (m_eval_data.size() < 16) {
                m_eval_data.push_back(seed_sample);
            }

            const std::size_t index = m_training_data.size() - 1;
            const std::string document_id = derive_document_id(seed_sample, index);
            if (!document_id.empty()) {
                m_curator.mark_seen(document_id);
                if (seed_sample.provenance.is_object()) {
                    seed_sample.provenance.as_object()["sample_hash"] = Json(document_id);
                }
                m_retrieval.ingest_document(document_id, seed_sample.teacher_output);
            }

            m_tokenizer.save_vocab(kVocabPath.string());
            persist_sample(seed_sample);
            train_step(seed_sample);
        }
    }
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
        if (auto sample = parse_sample_line(line)) {
            CuratedSample sample_value = *sample;
            const std::size_t before_vocab = m_tokenizer.vocab().size();
            m_tokenizer.build_vocab({sample_value.prompt, sample_value.teacher_output});
            if (m_tokenizer.vocab().size() > before_vocab) {
                m_student.base().resize_vocab(m_tokenizer.vocab().size());
            }
            const std::size_t index = m_training_data.size();
            const std::string document_id = derive_document_id(sample_value, index);
            if (!document_id.empty()) {
                m_curator.mark_seen(document_id);
                if (sample_value.provenance.is_object()) {
                    auto& prov = sample_value.provenance.as_object();
                    if (prov.find("sample_hash") == prov.end()) {
                        prov["sample_hash"] = Json(document_id);
                    }
                }
            }
            m_training_data.push_back(sample_value);
            if (m_eval_data.size() < 16) {
                m_eval_data.push_back(sample_value);
            }
            if (!document_id.empty()) {
                m_retrieval.ingest_document(document_id, sample_value.teacher_output);
            } else {
                std::hash<std::string> hasher;
                std::ostringstream oss;
                oss << "sample:" << index << ':' << hasher(sample_value.prompt + sample_value.teacher_output);
                m_retrieval.ingest_document(oss.str(), sample_value.teacher_output);
            }
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
        if (auto it = prov.find("sample_hash"); it != prov.end() && it->second.is_string()) {
            const std::string& value = it->second.as_string();
            if (!value.empty()) {
                return value;
            }
        }
        if (auto it = prov.find("prompt_hash"); it != prov.end() && it->second.is_string()) {
            const std::string& value = it->second.as_string();
            if (!value.empty()) {
                if (auto teacher_it = prov.find("teacher_hash"); teacher_it != prov.end() && teacher_it->second.is_string()) {
                    const std::string& teacher_hash = teacher_it->second.as_string();
                    if (!teacher_hash.empty()) {
                        return value + "::" + teacher_hash;
                    }
                }
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

