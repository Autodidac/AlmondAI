#pragma once

#include "model.hpp"
#include "ingest.hpp"
#include "tokenizer_word.hpp"

#include <vector>

namespace almondai {

struct EvalMetrics {
    double loss = 0.0;
    double accuracy = 0.0;
};

class Evaluator {
public:
    explicit Evaluator(const WordTokenizer& tokenizer);

    EvalMetrics evaluate(StudentModel& model, const std::vector<CuratedSample>& dataset) const;

private:
    const WordTokenizer& m_tokenizer;
};

} // namespace almondai

