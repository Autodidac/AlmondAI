#pragma once

#include "trainer.hpp"

#include <vector>

namespace almondai {

void refresh_retrieval_index(const std::vector<TrainingExample>& dataset, BpeTokenizer& tokenizer);

} // namespace almondai

