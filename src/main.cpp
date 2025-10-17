#include <iostream>

#include "almondai/adapter.hpp"
#include "almondai/governor.hpp"
#include "almondai/model.hpp"
#include "almondai/retrieval.hpp"
#include "almondai/serve.hpp"
#include "almondai/tokenizer_word.hpp"
#include "almondai/train.hpp"

int main() {
    using namespace almondai;

    TokenizerWord tokenizer({true, false});
    std::vector<std::string> bootstrap = {
        "Hello world from AlmondAI",
        "Continuous learning adapters",
        "Retrieval augmented generation",
    };
    tokenizer.build(bootstrap);
    tokenizer.save_vocab("bin/vocab.txt");

    RetrievalIndex retrieval;
    retrieval.add_document("doc1", "AlmondAI enables local adapters for rapid iteration.", tokenizer);
    retrieval.add_document("doc2", "Retrieval provides grounding evidence for responses.", tokenizer);

    AdapterRegistry registry;
    PolicyGovernor governor;

    ModelConfig config;
    config.vocab_size = tokenizer.vocab_size();
    config.hidden_dim = 32;
    config.output_dim = 32;
    config.max_seq_len = 128;

    DecoderModel model(config, registry);
    Trainer trainer(model, registry, tokenizer, retrieval, governor);

    std::vector<LabeledExample> raw_examples = {
        {"Explain adapters", "Adapters enable rapid specialization.", "{}", {"gpt", "", ""}},
        {"Summarize retrieval", "Retrieval surfaces contextual facts.", "{}", {"gpt", "", ""}},
    };
    std::vector<std::string> student_outputs = {
        "Adapters change weights.",
        "Retrieval finds documents.",
    };
    std::vector<std::string> canary_prompts = {
        "How do adapters help?",
        "Why use retrieval?",
    };

    trainer.continuous_learning_cycle(raw_examples, student_outputs, canary_prompts, "bootstrap-adapter");

    ConsoleApplication app(model, tokenizer, retrieval, governor, trainer, registry);
    app.run();
    return 0;
}
