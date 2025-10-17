#include "../AlmondAI/include/almondai/serve.hpp"
#include "../AlmondAI/include/almondai/adapter.hpp"
#include "../AlmondAI/include/almondai/tokenizer_word.hpp"

#include <filesystem>
#include <iostream>

int main() {
    using namespace almondai;

    WordTokenizer tokenizer;
    const std::filesystem::path vocab_path = "data/vocab.txt";
    if (std::filesystem::exists(vocab_path)) {
        tokenizer.load_vocab(vocab_path.string());
    } else {
        tokenizer.save_vocab(vocab_path.string());
    }

    ModelConfig config;
    config.vocab_size = tokenizer.vocab().size();
    config.hidden_size = 64;
    config.num_layers = 2;

    BaseDecoder base(config);
    StudentModel student(std::move(base));

    AdapterManager adapter_manager;
    AdapterConfig adapter_config;
    adapter_manager.register_adapter(Adapter("default", config.hidden_size, adapter_config));
    adapter_manager.activate("default");

    PolicyGovernor governor;
    governor.set_blocklist({"forbidden", "classified"});

    ContinuousLearner learner(std::move(student), std::move(adapter_manager), std::move(tokenizer), std::move(governor));
    learner.promote_adapter("default");

    MCPBridge bridge;
    Service service(learner, bridge);
    service.run(std::cin, std::cout);
    return 0;
}

