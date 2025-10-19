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
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace almondai {

namespace {
const std::filesystem::path kTrainingDataPath{"data/training_data.jsonl"};
const std::filesystem::path kSeedDataPath{"data/training_seed.jsonl"};
const std::filesystem::path kVocabPath{"data/vocab.txt"};
const std::filesystem::path kWeightsPath{"data/student_weights.json"};
const std::filesystem::path kSeedTextPath{"data/seed.txt"};

constexpr const char kDefaultSeedText[] =
    R"(AlmondAI is a self-evolving C++23 AI engine runtime that learns from its own source code, compiler feedback, and user interaction. It integrates AI directly into the software loop, enabling self-analysis, self-rebuilds, and continuous evolution across its modules.

Hello  <eos>
hello  <eos>
Hi  <eos>
Hey  <eos>
Hey there  <eos>
Hello there  <eos>
Greetings  <eos>
Howdy  <eos>
Nice to meet you  <eos>
Pleasure to meet you  <eos>
Good to see you  <eos>
Welcome back  <eos>
Long time no see  <eos>
How are you?  <eos>
I'm fine. How about you?  <eos>
Good morning  <eos>
Good afternoon  <eos>
Good evening  <eos>
Good night  <eos>
What's up  <eos>
What's up?  <eos>
Sup  <eos>
How's it going?  <eos>
What's new  <eos>
Not much  <eos>

Bonjour (French)  <eos>
Konnichiwa (Japanese)  <eos>
As-salamu alaykum (Arabic) → Wa alaykum as-salam  <eos>
Hola (Spanish)  <eos>
Sawasdee (Thai)  <eos>

In Japan, people bow to show respect.  <eos>
In many Middle Eastern cultures, the left hand is not used for eating.  <eos>
In France, it's common to kiss on both cheeks when greeting friends.  <eos>

The human brain has about 86 billion neurons that communicate through signals.  <eos>
These neurons form networks responsible for thinking, memory, and movement.  <eos>

In quantum mechanics, particles can exist in a state of uncertainty until observed.  <eos>
This idea is often described using the example of Schrödinger's cat.  <eos>

The Renaissance was a time of cultural and scientific growth in Europe.  <eos>
Key figures include Leonardo da Vinci, Galileo Galilei, and Johannes Gutenberg.  <eos>

The Great Wall of China spans over 13,000 miles and was built to protect against invasions.  <eos>
It includes watchtowers, garrisons, and signal beacons.  <eos>

Free will is the idea that humans can make choices independent of external forces.  <eos>
Determinism argues that all events are caused by prior causes, making free will an illusion.  <eos>

In Japan, people use chopsticks for eating. Pointing them at others is considered rude.  <eos>
In many Middle Eastern cultures, using the right hand for eating is preferred.  <eos>

Time flows from past to future in physics.  <eos>
Einstein's theory of relativity shows that time can be affected by gravity and speed.  <eos>

If every part of a ship is replaced over time, is it still the same ship?  <eos>
A moving object can never reach its destination because it must cover half the distance first, then half of that, and so on infinitely.  <eos>

Should you steal medicine to save a dying loved one if no one else has access to it?  <eos>

The weather is nice today. I enjoy sunny days.  <eos>
I feel tired but I will keep going.  <eos>
Making mistakes is part of learning.  <eos>
Sometimes things go wrong, but we can fix them.  <eos>
Bad choices often lead to bad outcomes.  <eos>

Wars can last for many years.  <eos>
Dynasties often change over time.  <eos>
Japan has a rich history of cultural traditions.  <eos>
Formality is important in some cultures but less so in others.  <eos>
)";

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
        if (!out) {
            return std::string();
        }

        out << kDefaultSeedText;
    }

    std::ifstream in(kSeedTextPath);
    if (!in) {
        return std::string();
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void ensure_seed_samples() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(kSeedDataPath.parent_path(), ec);

    bool need_default = true;
    if (fs::exists(kSeedDataPath, ec) && !ec) {
        const auto size = fs::file_size(kSeedDataPath, ec);
        if (!ec && size > 0) {
            need_default = false;
        }
    }

    if (!need_default) {
        return;
    }

    const std::string seed_text = ensure_seed_text();
    const std::string introduction = seed_text.empty()
        ? std::string{
              "AlmondAI is a self-evolving C++23 AI engine runtime that learns from its own source code, compiler feedback, and user interaction. "
              "It integrates AI directly into the software loop, enabling self-analysis, self-rebuilds, and continuous evolution across its modules."}
        : seed_text;

    std::ofstream out(kSeedDataPath, std::ios::trunc);
    if (!out) {
        return;
    }

    auto emit_sample = [&out](const std::string& prompt,
                              const std::string& teacher_output,
                              const std::string& prompt_hash) {
        JsonObject provenance;
        provenance["source"] = Json("seed");
        provenance["prompt_hash"] = Json(prompt_hash);
        provenance["teacher_hash"] = Json(std::to_string(std::hash<std::string>{}(teacher_output)));

        JsonObject sample;
        sample["prompt"] = Json(prompt);
        sample["teacher_output"] = Json(teacher_output);
        sample["constraints"] = Json(JsonObject{});
        sample["provenance"] = Json(provenance);

        out << Json(sample).dump() << '\n';
    };

    emit_sample("Introduce AlmondAI to a new teammate and describe its design pillars.",
                introduction,
                "seed::introduction");

    emit_sample("How does AlmondAI preserve its learning progress between sessions?",
                "AlmondAI writes curated examples to data/training_data.jsonl, records a running training log in data/"
                "training_log.txt, persists tokenizer vocabulary in data/vocab.txt, and saves decoder weights in data/"
                "student_weights.json. These artifacts let the learner restore adapters, retrieval indices, and vocab so "
                "fine-tuning can resume immediately after a restart.",
                "seed::persistence_overview");

    emit_sample("Explain how AlmondAI curates and reuses training samples during continuous learning.",
                "Every approved prompt/response pair is normalised by the curator, appended to data/training_data.jsonl, "
                "indexed for retrieval, and optionally reserved as evaluation data. When new training begins, the learner "
                "replays these curated samples so adapters and vocabularies stay aligned with the model's production "
                "experience.",
                "seed::curation_cycle");
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
        std::string retrieval_text = curated->prompt;
        if (!retrieval_text.empty() && !teacher_output.empty()) {
            retrieval_text.append("\n\n");
        }
        retrieval_text.append(teacher_output);
        m_retrieval.ingest_document(document_id, retrieval_text);
        m_document_to_index[document_id] = index;
    } else {
        std::string retrieval_text = curated->prompt;
        if (!retrieval_text.empty() && !teacher_output.empty()) {
            retrieval_text.append("\n\n");
        }
        retrieval_text.append(teacher_output);
        m_retrieval.ingest_document(prompt_hash, retrieval_text);
        m_document_to_index[prompt_hash] = index;
    }
    persist_sample(m_training_data.back());
    if (m_log_file.is_open()) {
        const std::size_t prompt_tokens = m_tokenizer.encode(curated->prompt).size();
        const std::size_t teacher_tokens = m_tokenizer.encode(teacher_output).size();
        m_log_file << "[learn::ingest] prompt_hash=" << (prompt_hash.empty() ? std::string{"unknown"} : prompt_hash)
                   << " | teacher_source=" << (teacher_source.empty() ? std::string{"unspecified"} : teacher_source)
                   << " | prompt_tokens=" << prompt_tokens
                   << " | teacher_tokens=" << teacher_tokens
                   << '\n';
        m_log_file.flush();
    }
    return m_training_data.back();
}

TrainingStats ContinuousLearner::train_step(const CuratedSample& sample) {
    ++m_step;
    TrainingStats stats;
    stats.step = m_step;
    stats.learning_tags.emplace_back("learn::step.begin");
    JsonObject begin_event;
    begin_event["tag"] = Json("learn::step.begin");
    begin_event["step"] = Json(static_cast<double>(stats.step));
    if (sample.provenance.is_object()) {
        const auto& prov = sample.provenance.as_object();
        if (auto it = prov.find("prompt_hash"); it != prov.end()) {
            begin_event["prompt_hash"] = it->second;
        }
        if (auto it = prov.find("sample_hash"); it != prov.end()) {
            begin_event["sample_hash"] = it->second;
        }
        if (auto it = prov.find("source"); it != prov.end()) {
            begin_event["teacher_source"] = it->second;
        }
    }
    stats.learning_trace.emplace_back(Json(begin_event));

    auto tokens = m_tokenizer.encode(sample.prompt);
    stats.learning_tags.emplace_back("learn::tokenize.prompt");
    JsonObject prompt_event;
    prompt_event["tag"] = Json("learn::tokenize.prompt");
    prompt_event["tokens"] = Json(static_cast<double>(tokens.size()));
    prompt_event["characters"] = Json(static_cast<double>(sample.prompt.size()));
    prompt_event["vocab_size"] = Json(static_cast<double>(m_tokenizer.vocab().size()));
    stats.learning_trace.emplace_back(Json(prompt_event));
    auto forward = m_student.forward(tokens);
    const auto& logits = forward.logits;
    const auto& hidden = forward.hidden;
    const auto& pre_adapter_hidden = forward.pre_adapter_hidden;
    stats.learning_tags.emplace_back("learn::forward.pass");
    JsonObject forward_event;
    forward_event["tag"] = Json("learn::forward.pass");
    forward_event["logit_count"] = Json(static_cast<double>(logits.size()));
    forward_event["hidden_width"] = Json(static_cast<double>(hidden.size()));
    stats.learning_trace.emplace_back(Json(forward_event));

    auto teacher_tokens = m_tokenizer.encode(sample.teacher_output);
    stats.learning_tags.emplace_back("learn::tokenize.teacher");
    JsonObject teacher_event;
    teacher_event["tag"] = Json("learn::tokenize.teacher");
    teacher_event["tokens"] = Json(static_cast<double>(teacher_tokens.size()));
    teacher_event["characters"] = Json(static_cast<double>(sample.teacher_output.size()));
    stats.learning_trace.emplace_back(Json(teacher_event));
    std::unordered_map<int, double> token_counts;
    for (int token : teacher_tokens) {
        if (token < 0) {
            continue;
        }
        const std::size_t index = static_cast<std::size_t>(token);
        if (index >= logits.size()) {
            continue;
        }
        token_counts[token] += 1.0;
    }
    if (token_counts.empty() && !logits.empty()) {
        token_counts[0] = 1.0;
    }

    const double total = std::accumulate(token_counts.begin(), token_counts.end(), 0.0,
                                         [](double sum, const auto& entry) {
                                             return sum + entry.second;
                                         });
    std::vector<double> target_distribution(logits.size(), 0.0);
    for (const auto& [token, count] : token_counts) {
        const std::size_t index = static_cast<std::size_t>(token);
        target_distribution[index] = count / (total > 0.0 ? total : 1.0);
    }

    std::vector<double> probabilities(logits.size(), 0.0);
    double normaliser = 0.0;
    double max_logit = logits.empty() ? 0.0 : *std::max_element(logits.begin(), logits.end());
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const double value = std::exp(logits[i] - max_logit);
        probabilities[i] = value;
        normaliser += value;
    }
    if (normaliser > 0.0) {
        for (double& probability : probabilities) {
            probability /= normaliser;
        }
    } else if (!probabilities.empty()) {
        const double uniform = 1.0 / static_cast<double>(probabilities.size());
        std::fill(probabilities.begin(), probabilities.end(), uniform);
    }

    constexpr double kEpsilon = 1e-12;
    std::vector<double> grad_logits(logits.size(), 0.0);
    double loss = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        grad_logits[i] = probabilities[i] - target_distribution[i];
        if (target_distribution[i] > 0.0) {
            loss -= target_distribution[i] * std::log(std::max(probabilities[i], kEpsilon));
        }
    }

    std::vector<double> grad_hidden(hidden.size(), 0.0);
    if (!grad_logits.empty()) {
        grad_hidden = m_student.update(hidden, grad_logits);
        stats.learning_tags.emplace_back("learn::update.student");
        JsonObject update_event;
        update_event["tag"] = Json("learn::update.student");
        update_event["gradient_dimensions"] = Json(static_cast<double>(grad_logits.size()));
        update_event["hidden_dimensions"] = Json(static_cast<double>(hidden.size()));
        stats.learning_trace.emplace_back(Json(update_event));
    }
    if (Adapter* active = m_adapters.active_adapter()) {
        active->apply_gradient(pre_adapter_hidden, grad_hidden);
        active->update_statistics(pre_adapter_hidden);
        stats.adapter_norm = active->norm();
        stats.learning_tags.emplace_back("learn::update.adapter");
        JsonObject adapter_event;
        adapter_event["tag"] = Json("learn::update.adapter");
        adapter_event["adapter_norm"] = Json(stats.adapter_norm);
        adapter_event["adapter_name"] = Json(active->name());
        stats.learning_trace.emplace_back(Json(adapter_event));
    }

    auto max_it = std::max_element(probabilities.begin(), probabilities.end());
    std::vector<int> decoded;
    if (max_it != probabilities.end()) {
        decoded.push_back(static_cast<int>(std::distance(probabilities.begin(), max_it)));
    }
    std::string student_output = m_tokenizer.decode(decoded);
    m_curator.record_student_response(sample.prompt, student_output, sample);

    std::unordered_set<int> teacher_vocab;
    for (const auto& [token, _] : token_counts) {
        (void)_;
        teacher_vocab.insert(token);
    }

    stats.loss = loss;
    if (max_it != probabilities.end()) {
        const int prediction = static_cast<int>(std::distance(probabilities.begin(), max_it));
        stats.accuracy = teacher_vocab.count(prediction) ? 1.0 : 0.0;
    } else {
        stats.accuracy = 0.0;
    }
    stats.retrieval_hit_rate = m_retrieval.hit_rate();
    if (sample.provenance.is_object()) {
        const auto& prov = sample.provenance.as_object();
        if (auto src_it = prov.find("source"); src_it != prov.end() && src_it->second.is_string()) {
            stats.teacher_source = src_it->second.as_string();
        }
    }
    stats.learning_tags.emplace_back("learn::summary");
    JsonObject summary_event;
    summary_event["tag"] = Json("learn::summary");
    summary_event["loss"] = Json(stats.loss);
    summary_event["accuracy"] = Json(stats.accuracy);
    summary_event["retrieval_hit_rate"] = Json(stats.retrieval_hit_rate);
    if (!stats.teacher_source.empty()) {
        summary_event["teacher_source"] = Json(stats.teacher_source);
    }
    stats.learning_trace.emplace_back(Json(summary_event));
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
    stats.learning_tags.emplace_back("learn::evaluate.canary");
    JsonObject evaluate_event;
    evaluate_event["tag"] = Json("learn::evaluate.canary");
    evaluate_event["samples_evaluated"] = Json(static_cast<double>(m_eval_data.size()));
    stats.learning_trace.emplace_back(Json(evaluate_event));
    stats.step = m_step;
    stats.loss = metrics.loss;
    stats.accuracy = metrics.accuracy;
    stats.retrieval_hit_rate = m_retrieval.hit_rate();
    if (const Adapter* adapter = m_adapters.active_adapter()) {
        stats.adapter_norm = adapter->norm();
    }
    stats.teacher_source = "evaluation";
    stats.learning_tags.emplace_back("learn::summary");
    JsonObject summary_event;
    summary_event["tag"] = Json("learn::summary");
    summary_event["loss"] = Json(stats.loss);
    summary_event["accuracy"] = Json(stats.accuracy);
    summary_event["retrieval_hit_rate"] = Json(stats.retrieval_hit_rate);
    summary_event["teacher_source"] = Json(stats.teacher_source);
    stats.learning_trace.emplace_back(Json(summary_event));
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
               << " | teacher_source=" << (stats.teacher_source.empty() ? std::string{"unknown"} : stats.teacher_source);
    if (!stats.learning_tags.empty()) {
        m_log_file << " | tags=[";
        for (std::size_t i = 0; i < stats.learning_tags.size(); ++i) {
            if (i != 0) {
                m_log_file << ' ';
            }
            m_log_file << stats.learning_tags[i];
        }
        m_log_file << ']';
    }
    if (!stats.learning_trace.empty()) {
        Json trace_json(stats.learning_trace);
        m_log_file << " | trace=" << trace_json.dump();
    }
    m_log_file << '\n';
    m_log_file.flush();
}

void ContinuousLearner::load_persistent_data() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(kTrainingDataPath.parent_path(), ec);

    ensure_seed_samples();

    if (fs::exists(kWeightsPath)) {
        m_student.base().load_weights(kWeightsPath.string());
    }

    if (!fs::exists(kTrainingDataPath) && fs::exists(kSeedDataPath)) {
        fs::copy_file(kSeedDataPath, kTrainingDataPath, fs::copy_options::overwrite_existing, ec);
    }

    load_samples_from_file(kTrainingDataPath);

    if (m_training_data.empty()) {
        const std::string seed_text = ensure_seed_text();

        auto register_seed_sample = [&](const std::string& prompt,
                                        const std::string& teacher_output,
                                        const std::string& prompt_hash) {
            if (prompt.empty() || teacher_output.empty()) {
                return;
            }

            CuratedSample sample;
            sample.prompt = prompt;
            sample.teacher_output = teacher_output;
            sample.constraints = Json(JsonObject{});
            JsonObject provenance;
            provenance["source"] = Json("seed");
            provenance["prompt_hash"] = Json(prompt_hash);
            provenance["teacher_hash"] = Json(std::to_string(std::hash<std::string>{}(sample.teacher_output)));
            sample.provenance = Json(provenance);

            const std::size_t before_vocab = m_tokenizer.vocab().size();
            m_tokenizer.build_vocab({sample.prompt, sample.teacher_output});
            if (m_tokenizer.vocab().size() > before_vocab) {
                m_student.base().resize_vocab(m_tokenizer.vocab().size());
            }

            m_training_data.push_back(sample);
            if (m_eval_data.size() < 16) {
                m_eval_data.push_back(sample);
            }

            CuratedSample& stored = m_training_data.back();
            const std::size_t index = m_training_data.size() - 1;
            const std::string document_id = derive_document_id(stored, index);
            if (!document_id.empty()) {
                m_curator.mark_seen(document_id);
                if (stored.provenance.is_object()) {
                    stored.provenance.as_object()["sample_hash"] = Json(document_id);
                }
                std::string retrieval_text = stored.prompt;
                if (!retrieval_text.empty() && !stored.teacher_output.empty()) {
                    retrieval_text.append("\n\n");
                }
                retrieval_text.append(stored.teacher_output);
                m_retrieval.ingest_document(document_id, retrieval_text);
                m_document_to_index[document_id] = index;
            }

            m_tokenizer.save_vocab(kVocabPath.string());
            persist_sample(stored);
            train_step(stored);
        };

        if (!seed_text.empty()) {
            register_seed_sample("Introduce AlmondAI to a new user.",
                                 seed_text,
                                 "seed::bootstrap");
        }

        struct SeedSpec {
            const char* prompt;
            const char* teacher_output;
            const char* prompt_hash;
        };

        const std::vector<SeedSpec> greeting_samples = {
            {"Offer a warm greeting to someone joining the conversation.",
             "Hello! It's great to hear from you. How can I support you today?",
             "seed::greeting::hello"},
            {"Respond to a user who says 'Hello there'.",
             "Hello there! It's a pleasure to connect—what would you like to dive into today?",
             "seed::greeting::hello_there"},
            {"Respond to a user who asks 'How are you?'.",
             "I'm doing great, thanks for asking! How can I help you today?",
             "seed::greeting::how_are_you"},
            {"Respond to a user who says 'Good morning'.",
             "Good morning! I hope your day is off to a bright and productive start.",
             "seed::greeting::good_morning"},
            {"Respond to a user who says 'Good afternoon'.",
             "Good afternoon! I hope everything's going smoothly—let me know what you'd like to tackle next.",
             "seed::greeting::good_afternoon"},
            {"Respond to a user who says 'Good evening'.",
             "Good evening! I hope the rest of your day treats you well—what should we look at next?",
             "seed::greeting::good_evening"},
            {"Respond to a user who says 'Good night'.",
             "Good night! Rest well, and ping me again whenever you're ready to continue.",
             "seed::greeting::good_night"},
            {"Respond to a user who says 'Hi'.",
             "Hi there! How can I assist you today?",
             "seed::greeting::hi"},
            {"Respond to a user who says 'Hiya'.",
             "Hiya! I'm ready when you are—what's on your mind?",
             "seed::greeting::hiya"},
            {"Respond to a user who says 'Hey'.",
             "Hey! Glad you're here. What can I do for you?",
             "seed::greeting::hey"},
            {"Respond to a user who says 'Yo'.",
             "Yo! Always happy to help—what are we working on today?",
             "seed::greeting::yo"},
            {"Respond to a user who asks 'How's it going?'.",
             "It's going great—thanks for checking in! What should we dive into next?",
             "seed::greeting::hows_it_going"},
            {"Respond to a user who says 'What's up?'.",
             "Not much—I'm right here and ready to help. What's on your agenda?",
             "seed::greeting::whats_up"},
            {"Respond to a user who says 'Sup?'.",
             "Sup! I'm dialed in and ready to jump into whatever you need.",
             "seed::greeting::sup"},
            {"Respond to a user who says 'Howdy'.",
             "Howdy! Always nice to hear from you—how can I lend a hand?",
             "seed::greeting::howdy"},
            {"Respond to a user who says 'Nice to meet you'.",
             "Nice to meet you too! Let me know what you're curious about and we'll explore it together.",
             "seed::greeting::nice_to_meet_you"},
            {"Respond to a user who says 'Pleasure to meet you'.",
             "The pleasure's mine! I'm here whenever you want to dig into something.",
             "seed::greeting::pleasure_to_meet_you"},
            {"Respond to a user who says 'Long time no see'.",
             "Long time no see! Let's pick up right where we left off.",
             "seed::greeting::long_time_no_see"},
            {"Respond to a user who says 'Welcome back'.",
             "Thanks! I'm all set to help—what should we get started on?",
             "seed::greeting::welcome_back"},
            {"Reply when someone thanks AlmondAI for the help.",
             "You're very welcome! I'm glad I could assist—let me know if there's anything else you need.",
             "seed::greeting::gratitude"},
            {"Close a conversation with a friendly farewell.",
             "Thanks for chatting with me. If you have more questions later, I'll be here. Take care!",
             "seed::greeting::farewell"},
            {"Respond to a user who says 'Goodbye'.",
             "Goodbye! It was great chatting—feel free to reach out again anytime you need a hand.",
             "seed::greeting::goodbye"}
        };

        for (const auto& sample : greeting_samples) {
            register_seed_sample(sample.prompt, sample.teacher_output, sample.prompt_hash);
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
            std::string retrieval_text = sample_value.prompt;
            if (!retrieval_text.empty() && !sample_value.teacher_output.empty()) {
                retrieval_text.append("\n\n");
            }
            retrieval_text.append(sample_value.teacher_output);
            if (!document_id.empty()) {
                m_retrieval.ingest_document(document_id, retrieval_text);
                m_document_to_index[document_id] = index;
            } else {
                std::hash<std::string> hasher;
                std::ostringstream oss;
                oss << "sample:" << index << ':' << hasher(sample_value.prompt + sample_value.teacher_output);
                const std::string fallback_id = oss.str();
                m_retrieval.ingest_document(fallback_id, retrieval_text);
                m_document_to_index[fallback_id] = index;
            }
        }
    }
    if (!m_training_data.empty()) {
        m_tokenizer.save_vocab(kVocabPath.string());
    }
}

const CuratedSample* ContinuousLearner::recall_sample(const std::string& document_id) const {
    auto it = m_document_to_index.find(document_id);
    if (it == m_document_to_index.end()) {
        return nullptr;
    }
    const std::size_t index = it->second;
    if (index >= m_training_data.size()) {
        return nullptr;
    }
    return &m_training_data[index];
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

