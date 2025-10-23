#include "../include/almondai/trainer.hpp"

#include "../include/almondai/json.hpp"
#include "../include/almondai/adapter.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <limits>

namespace almondai {

namespace {

std::vector<int> trim_pad(const std::vector<int>& tokens) {
    std::vector<int> trimmed;
    trimmed.reserve(tokens.size());
    for (int token : tokens) {
        if (token == BpeTokenizer::PAD_ID) {
            continue;
        }
        trimmed.push_back(token);
    }
    return trimmed;
}

void truncate_context(std::vector<int>& context, std::size_t limit) {
    if (limit == 0 || context.size() <= limit) {
        return;
    }
    context.erase(context.begin(), context.end() - static_cast<std::ptrdiff_t>(limit));
}

} // namespace

std::vector<std::string> tags_from_json(const Json& value) {
    std::vector<std::string> tags;
    if (!value.is_array()) {
        return tags;
    }
    for (const auto& item : value.as_array()) {
        if (item.is_string()) {
            tags.push_back(item.as_string());
        }
    }
    return tags;
}

std::vector<std::string> evaluation_tags(const TrainingExample& sample) {
    std::unordered_set<std::string> unique;
    if (sample.provenance.is_object()) {
        const auto& provenance = sample.provenance.as_object();
        if (auto it = provenance.find("tags"); it != provenance.end()) {
            for (const auto& tag : tags_from_json(it->second)) {
                unique.insert(tag);
            }
        }
        if (auto src_it = provenance.find("source"); src_it != provenance.end() && src_it->second.is_string()) {
            unique.insert(std::string("source::") + src_it->second.as_string());
        }
        if (auto prompt_it = provenance.find("prompt_hash"); prompt_it != provenance.end() && prompt_it->second.is_string()) {
            unique.insert(prompt_it->second.as_string());
        }
    }
    if (sample.constraints.is_object()) {
        const auto& constraints = sample.constraints.as_object();
        if (auto it = constraints.find("tags"); it != constraints.end()) {
            for (const auto& tag : tags_from_json(it->second)) {
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

Trainer::Trainer(StudentModel& model,
                 BpeTokenizer& tokenizer,
                 AdamWOptimizer optimizer,
                 WarmupCosineScheduler scheduler)
    : m_model(model)
    , m_tokenizer(tokenizer)
    , m_optimizer(std::move(optimizer))
    , m_scheduler(std::move(scheduler)) {
    m_checkpoint_path = std::filesystem::path("data") / "student_weights.json";
    const auto& projection = m_model.base().output_projection();
    m_optimizer.reset(projection.vector().size());
}

void Trainer::set_checkpoint_path(std::filesystem::path path) {
    m_checkpoint_path = std::move(path);
}

void Trainer::set_eval_dataset(std::vector<TrainingExample> dataset) {
    m_eval_dataset = std::move(dataset);
}

void Trainer::append_training_example(const TrainingExample& example) {
    m_training_data.push_back(example);
}

Trainer::BatchTensor Trainer::prepare_batch(const std::vector<TrainingExample>& batch) const {
    BatchTensor tensor;
    if (batch.empty()) {
        return tensor;
    }

    std::vector<std::vector<int>> inputs;
    std::vector<std::vector<int>> targets;
    inputs.reserve(batch.size());
    targets.reserve(batch.size());

    std::size_t max_input = 0;
    std::size_t max_target = 0;
    for (const auto& example : batch) {
        auto input_tokens = m_tokenizer.encode(example.prompt);
        auto target_tokens = m_tokenizer.encode(example.teacher_output);
        target_tokens.erase(
            std::remove(target_tokens.begin(), target_tokens.end(), BpeTokenizer::EOS_ID),
            target_tokens.end());
        target_tokens.push_back(BpeTokenizer::EOS_ID);

        max_input = std::max(max_input, input_tokens.size());
        max_target = std::max(max_target, target_tokens.size());

        inputs.push_back(std::move(input_tokens));
        targets.push_back(std::move(target_tokens));
    }

    tensor.inputs.resize(batch.size());
    tensor.targets.resize(batch.size());
    tensor.masks.resize(batch.size());

    for (std::size_t i = 0; i < batch.size(); ++i) {
        auto& input = inputs[i];
        auto& target = targets[i];

        if (input.size() < max_input) {
            input.resize(max_input, BpeTokenizer::PAD_ID);
        }
        if (target.size() < max_target) {
            target.resize(max_target, BpeTokenizer::PAD_ID);
        }

        std::vector<double> mask(max_target, 0.0);
        for (std::size_t j = 0; j < target.size(); ++j) {
            if (target[j] != BpeTokenizer::PAD_ID) {
                mask[j] = 1.0;
                tensor.token_count += 1;
            }
        }

        tensor.inputs[i] = std::move(input);
        tensor.targets[i] = std::move(target);
        tensor.masks[i] = std::move(mask);
    }
    return tensor;
}

std::vector<double> Trainer::compute_logits_gradient(const std::vector<double>& logits,
                                                     int target_id,
                                                     double label_smoothing,
                                                     double& loss_accumulator) const {
    const std::size_t vocab = logits.size();
    std::vector<double> probs(vocab, 0.0);
    double max_logit = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (std::size_t i = 0; i < vocab; ++i) {
        double value = std::exp(logits[i] - max_logit);
        probs[i] = value;
        sum += value;
    }
    if (sum <= 0.0) {
        sum = 1.0;
    }
    for (double& p : probs) {
        p /= sum;
        p = std::max(p, 1e-12);
    }

    std::vector<double> grad(vocab, 0.0);
    const double off_value = (vocab > 1) ? (label_smoothing / static_cast<double>(vocab - 1)) : 0.0;
    const double on_value = 1.0 - label_smoothing;

    for (std::size_t i = 0; i < vocab; ++i) {
        const double target_prob = (static_cast<int>(i) == target_id) ? on_value : off_value;
        grad[i] = probs[i] - target_prob;
        loss_accumulator += -target_prob * std::log(probs[i]);
    }
    return grad;
}

TrainingReport Trainer::train_on_batch(const std::vector<TrainingExample>& batch) {
    TrainingReport report;
    if (batch.empty()) {
        return report;
    }
    const auto prepared = prepare_batch(batch);
    if (prepared.token_count == 0) {
        return report;
    }

    const auto& config = m_model.base().config();
    const std::size_t vocab = config.vocab_size;
    const std::size_t hidden = config.hidden_size;

    std::vector<double> grad_projection(hidden * vocab, 0.0);
    double total_loss = 0.0;
    std::size_t total_tokens = 0;

    for (std::size_t i = 0; i < prepared.inputs.size(); ++i) {
        std::vector<int> context = trim_pad(prepared.inputs[i]);
        for (std::size_t t = 0; t < prepared.targets[i].size(); ++t) {
            if (prepared.masks[i][t] == 0.0) {
                continue;
            }
            int target_id = prepared.targets[i][t];
            truncate_context(context, config.context_length);
            auto forward = m_model.forward(context);

            double step_loss = 0.0;
            auto grad_logits = compute_logits_gradient(
                forward.logits,
                target_id,
                m_options.label_smoothing,
                step_loss);

            // Gradient clipping per token
            double norm = 0.0;
            for (double value : grad_logits) {
                norm += value * value;
            }
            norm = std::sqrt(norm);
            if (norm > m_options.gradient_clip && norm > 0.0) {
                const double scale = m_options.gradient_clip / norm;
                for (double& value : grad_logits) {
                    value *= scale;
                }
            }

            for (std::size_t h = 0; h < hidden; ++h) {
                const double hidden_value = forward.hidden[h];
                for (std::size_t v = 0; v < vocab; ++v) {
                    grad_projection[h * vocab + v] += hidden_value * grad_logits[v];
                }
            }

            total_loss += step_loss;
            ++total_tokens;
            context.push_back(target_id);
        }
    }

    if (total_tokens == 0) {
        return report;
    }

    const double inv_tokens = 1.0 / static_cast<double>(total_tokens);
    for (double& value : grad_projection) {
        value *= inv_tokens;
    }

    double grad_norm = 0.0;
    for (double value : grad_projection) {
        grad_norm += value * value;
    }
    grad_norm = std::sqrt(grad_norm);
    if (grad_norm > m_options.gradient_clip && grad_norm > 0.0) {
        const double scale = m_options.gradient_clip / grad_norm;
        for (double& value : grad_projection) {
            value *= scale;
        }
    }

    double lr_scale = m_scheduler.learning_rate_scale(m_step);
    auto& projection = m_model.base().output_projection().vector();
    m_optimizer.step(projection, grad_projection, lr_scale);

    ++m_step;
    m_tokens_trained += total_tokens;

    report.step = m_step;
    report.tokens = total_tokens;
    report.loss = total_loss * inv_tokens;
    report.perplexity = std::exp(report.loss);
    if (m_options.save_every > 0 && (m_step % m_options.save_every) == 0) {
        report.checkpoint_saved = save_checkpoint();
    }
    maybe_retune_scheduler(total_tokens, report.loss);
    return report;
}

EvaluationReport Trainer::evaluate(const std::vector<TrainingExample>& dataset) const {
    EvaluationReport report;
    if (dataset.empty()) {
        return report;
    }
    const auto prepared = prepare_batch(dataset);
    if (prepared.token_count == 0) {
        return report;
    }

    const auto& config = m_model.base().config();
    double total_loss = 0.0;
    std::size_t total_tokens = 0;
    std::vector<double> sample_loss(dataset.size(), 0.0);
    std::vector<std::size_t> sample_tokens(dataset.size(), 0);

    for (std::size_t i = 0; i < prepared.inputs.size(); ++i) {
        std::vector<int> context = trim_pad(prepared.inputs[i]);
        for (std::size_t t = 0; t < prepared.targets[i].size(); ++t) {
            if (prepared.masks[i][t] == 0.0) {
                continue;
            }
            int target_id = prepared.targets[i][t];
            truncate_context(context, config.context_length);
            auto forward = m_model.forward(context);
            double step_loss = 0.0;
            auto grad_logits = compute_logits_gradient(
                forward.logits,
                target_id,
                m_options.label_smoothing,
                step_loss);
            (void)grad_logits;
            total_loss += step_loss;
            ++total_tokens;
            sample_loss[i] += step_loss;
            ++sample_tokens[i];
            context.push_back(target_id);
        }
    }

    if (total_tokens == 0) {
        return report;
    }
    report.tokens = total_tokens;
    report.loss = total_loss / static_cast<double>(total_tokens);
    report.perplexity = std::exp(report.loss);

    std::unordered_map<std::string, double> tag_loss;
    std::unordered_map<std::string, std::size_t> tag_tokens;
    for (std::size_t i = 0; i < dataset.size(); ++i) {
        if (sample_tokens[i] == 0) {
            continue;
        }
        const auto tags = evaluation_tags(dataset[i]);
        for (const auto& tag : tags) {
            tag_loss[tag] += sample_loss[i];
            tag_tokens[tag] += sample_tokens[i];
        }
    }

    for (const auto& [tag, tokens] : tag_tokens) {
        if (tokens == 0) {
            continue;
        }
        double avg_loss = tag_loss[tag] / static_cast<double>(tokens);
        report.tag_perplexity[tag] = std::exp(avg_loss);
        report.tag_token_counts[tag] = tokens;
    }

    auto compute_token_set = [this](const std::string& text) {
        std::unordered_set<int> set;
        auto tokens = m_tokenizer.encode(text);
        for (int id : tokens) {
            if (id <= BpeTokenizer::PAD_ID || id == BpeTokenizer::EOS_ID) {
                continue;
            }
            set.insert(id);
        }
        return set;
    };

    double current_hit_rate = 0.0;
    if (dataset.size() >= 2) {
        std::size_t hits = 0;
        for (std::size_t i = 0; i < dataset.size(); ++i) {
            const auto prompt_tokens = compute_token_set(dataset[i].prompt);
            if (prompt_tokens.empty()) {
                continue;
            }
            double best = 0.0;
            for (std::size_t j = 0; j < dataset.size(); ++j) {
                if (i == j) {
                    continue;
                }
                const auto other_tokens = compute_token_set(dataset[j].prompt);
                if (other_tokens.empty()) {
                    continue;
                }
                std::size_t intersection = 0;
                for (int token : prompt_tokens) {
                    if (other_tokens.find(token) != other_tokens.end()) {
                        ++intersection;
                    }
                }
                const std::size_t union_size = prompt_tokens.size() + other_tokens.size() - intersection;
                if (union_size == 0) {
                    continue;
                }
                const double score = static_cast<double>(intersection) / static_cast<double>(union_size);
                best = std::max(best, score);
            }
            if (best > 0.0) {
                ++hits;
            }
        }
        current_hit_rate = static_cast<double>(hits) / static_cast<double>(dataset.size());
    }

    if (std::isfinite(current_hit_rate)) {
        double previous_hit_rate = m_retrieval_hit_rate_history.empty() ? current_hit_rate
                                                                        : m_retrieval_hit_rate_history.back();
        record_retrieval_hit_rate(current_hit_rate);
        report.retrieval_hit_rate = current_hit_rate;
        report.retrieval_hit_rate_delta = current_hit_rate - previous_hit_rate;
        report.retrieval_hit_rate_history.assign(m_retrieval_hit_rate_history.begin(),
                                                 m_retrieval_hit_rate_history.end());
    }

    double adapter_norm = 0.0;
    if (const Adapter* adapter = m_model.base().active_adapter()) {
        adapter_norm = adapter->norm();
    }
    report.current_adapter_norm = adapter_norm;
    record_adapter_norm(adapter_norm);
    report.adapter_norm_history.assign(m_adapter_norm_history.begin(), m_adapter_norm_history.end());

    report.policy_incident_history.assign(m_policy_incident_history.begin(), m_policy_incident_history.end());
    std::size_t recent_incidents = 0;
    for (std::size_t incidents : m_policy_incident_history) {
        recent_incidents += incidents;
    }
    report.recent_policy_incident_count = recent_incidents;

    return report;
}

bool Trainer::save_checkpoint() const {
    if (m_checkpoint_path.empty()) {
        return false;
    }
    std::filesystem::create_directories(m_checkpoint_path.parent_path());
    return m_model.base().save_weights(m_checkpoint_path.string());
}

void Trainer::record_retrieval_hit_rate(double hit_rate) const {
    if (!std::isfinite(hit_rate)) {
        return;
    }
    m_retrieval_hit_rate_history.push_back(hit_rate);
    while (m_retrieval_hit_rate_history.size() > kTelemetryWindow) {
        m_retrieval_hit_rate_history.pop_front();
    }
}

void Trainer::record_adapter_norm(double norm) const {
    if (!std::isfinite(norm)) {
        norm = 0.0;
    }
    m_adapter_norm_history.push_back(norm);
    while (m_adapter_norm_history.size() > kTelemetryWindow) {
        m_adapter_norm_history.pop_front();
    }
}

void Trainer::record_policy_incidents(std::size_t incidents) const {
    m_policy_incident_history.push_back(incidents);
    while (m_policy_incident_history.size() > kTelemetryWindow) {
        m_policy_incident_history.pop_front();
    }
}

void Trainer::log_scheduler_event(const std::string& message) const {
    std::cout << "[Trainer] " << message << std::endl;
}

void Trainer::maybe_retune_scheduler(std::size_t tokens, double loss) {
    constexpr std::size_t kWindow = 8;
    constexpr double kPlateauTolerance = 0.01; // 1% improvement threshold
    constexpr double kThroughputDrop = 0.75;    // 25% drop triggers retune
    constexpr std::size_t kRetuneCooldown = 200;

    m_recent_losses.push_back(loss);
    if (m_recent_losses.size() > kWindow) {
        m_recent_losses.pop_front();
    }
    m_recent_throughput.push_back(tokens);
    if (m_recent_throughput.size() > kWindow) {
        m_recent_throughput.pop_front();
    }

    if (m_step < m_last_scheduler_retune_step + kRetuneCooldown) {
        return;
    }

    auto relative_improvement = [this]() {
        if (m_recent_losses.size() < 2) {
            return 1.0;
        }
        const double first = m_recent_losses.front();
        const double last = m_recent_losses.back();
        if (!std::isfinite(first) || !std::isfinite(last)) {
            return 1.0;
        }
        if (std::fabs(first) < 1e-9) {
            return 1.0;
        }
        return (first - last) / std::fabs(first);
    }();

    if (m_recent_losses.size() >= kWindow && relative_improvement < kPlateauTolerance) {
        auto params = m_optimizer.params();
        params.learning_rate *= 0.9;
        m_optimizer.set_params(params);
        m_model.base().set_learning_rate(params.learning_rate);
        double new_min_ratio = std::max(0.01, m_scheduler.min_ratio() * 0.8);
        m_scheduler.set_min_ratio(new_min_ratio);
        std::size_t adjusted_total = std::max(m_scheduler.total_steps(), m_step + 1000);
        m_scheduler.set_total_steps(adjusted_total);
        log_scheduler_event("Loss plateau detected; reduced learning rate to " +
                            std::to_string(params.learning_rate) +
                            ", min_lr_ratio=" + std::to_string(new_min_ratio));
        m_recent_losses.clear();
        m_recent_throughput.clear();
        m_last_scheduler_retune_step = m_step;
        return;
    }

    if (m_recent_throughput.size() >= kWindow) {
        const std::size_t half = m_recent_throughput.size() / 2;
        if (half > 0) {
            double early_avg = 0.0;
            for (std::size_t i = 0; i < half; ++i) {
                early_avg += static_cast<double>(m_recent_throughput[i]);
            }
            early_avg /= static_cast<double>(half);
            double recent_avg = 0.0;
            for (std::size_t i = half; i < m_recent_throughput.size(); ++i) {
                recent_avg += static_cast<double>(m_recent_throughput[i]);
            }
            recent_avg /= static_cast<double>(m_recent_throughput.size() - half);
            if (early_avg > 0.0 && recent_avg < early_avg * kThroughputDrop) {
                auto params = m_optimizer.params();
                params.learning_rate *= 1.05;
                m_optimizer.set_params(params);
                m_model.base().set_learning_rate(params.learning_rate);
                std::size_t new_warmup = std::max<std::size_t>(1, m_scheduler.warmup_steps() / 2);
                m_scheduler.set_warmup_steps(new_warmup);
                std::size_t new_total = std::max(m_scheduler.total_steps(), m_step + 2000);
                m_scheduler.set_total_steps(new_total);
                log_scheduler_event("Token throughput dropped; boosted learning rate to " +
                                    std::to_string(params.learning_rate) +
                                    ", warmup=" + std::to_string(new_warmup));
                m_recent_losses.clear();
                m_recent_throughput.clear();
                m_last_scheduler_retune_step = m_step;
            }
        }
    }
}

} // namespace almondai

