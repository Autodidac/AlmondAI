#include "almondai/train.hpp"

#include <algorithm>
#include <iomanip>

namespace almondai {

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
    m_log_file.open("data/training_log.csv", std::ios::app);
    if (m_log_file.tellp() == 0) {
        m_log_file << "step,loss,accuracy,adapter_norm,retrieval_hit_rate\n";
    }
}

std::optional<CuratedSample> ContinuousLearner::ingest(const std::string& prompt,
                                                       const std::string& teacher_output,
                                                       Json constraints,
                                                       const std::string& prompt_hash) {
    auto curated = m_curator.curate(prompt, teacher_output, std::move(constraints), prompt_hash);
    if (!curated) {
        return std::nullopt;
    }
    m_tokenizer.build_vocab({curated->prompt, curated->teacher_output});
    m_tokenizer.save_vocab("data/vocab.txt");
    m_training_data.push_back(*curated);
    if (m_eval_data.size() < 16) {
        m_eval_data.push_back(*curated);
    }
    m_retrieval.ingest_document(prompt_hash, teacher_output);
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
    m_log_file << stats.step << ','
               << std::fixed << std::setprecision(6) << stats.loss << ','
               << stats.accuracy << ','
               << stats.adapter_norm << ','
               << stats.retrieval_hit_rate << '\n';
}

} // namespace almondai

