#include "../include/almondai/trainer.hpp"

#include "../include/almondai/json.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

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
            context.push_back(target_id);
        }
    }

    if (total_tokens == 0) {
        return report;
    }
    report.tokens = total_tokens;
    report.loss = total_loss / static_cast<double>(total_tokens);
    report.perplexity = std::exp(report.loss);
    return report;
}

bool Trainer::save_checkpoint() const {
    if (m_checkpoint_path.empty()) {
        return false;
    }
    std::filesystem::create_directories(m_checkpoint_path.parent_path());
    return m_model.base().save_weights(m_checkpoint_path.string());
}

} // namespace almondai

