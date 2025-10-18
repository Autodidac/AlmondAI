#include "../AlmondAI/include/almondai/serve.hpp"
#include "../AlmondAI/include/almondai/adapter.hpp"
#include "../AlmondAI/include/almondai/tokenizer_word.hpp"
#include "../AlmondAI/include/almondai/json.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>

namespace {

void extend_tokenizer_from_file(almondai::WordTokenizer& tokenizer, const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        return;
    }
    std::ifstream file(path);
    if (!file) {
        return;
    }
    std::vector<std::string> documents;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        try {
            almondai::Json record = almondai::Json::parse(line);
            if (!record.is_object()) {
                continue;
            }
            const auto& obj = record.as_object();
            if (auto it = obj.find("prompt"); it != obj.end() && it->second.is_string()) {
                documents.push_back(it->second.as_string());
            }
            if (auto it = obj.find("teacher_output"); it != obj.end() && it->second.is_string()) {
                documents.push_back(it->second.as_string());
            }
        } catch (...) {
            continue;
        }
    }
    if (!documents.empty()) {
        tokenizer.build_vocab(documents);
    }
}

} // namespace

int main() {
    using namespace almondai;

    WordTokenizer tokenizer;
    const std::filesystem::path vocab_path = "data/vocab.txt";
    if (std::filesystem::exists(vocab_path)) {
        tokenizer.load_vocab(vocab_path.string());
    }
    extend_tokenizer_from_file(tokenizer, "data/training_seed.jsonl");
    extend_tokenizer_from_file(tokenizer, "data/training_data.jsonl");
    tokenizer.save_vocab(vocab_path.string());

    ModelConfig config;
    config.vocab_size = tokenizer.vocab().size();
    config.hidden_size = 64;
    config.num_layers = 2;

    BaseDecoder base(config);
    const std::filesystem::path weights_path = "data/student_weights.json";
    if (std::filesystem::exists(weights_path)) {
        base.load_weights(weights_path.string());
    }
    config = base.config();
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

