#ifndef ALMONDAI_SERVE_HPP
#define ALMONDAI_SERVE_HPP

#include <memory>
#include <string>
#include <vector>

#include "almondai/adapter.hpp"
#include "almondai/mcp.hpp"

namespace almondai {

class DecoderModel;
class TokenizerWord;
class RetrievalIndex;
class PolicyGovernor;
class Trainer;

class ConsoleApplication {
public:
    ConsoleApplication(DecoderModel &model,
                       TokenizerWord &tokenizer,
                       RetrievalIndex &retrieval,
                       PolicyGovernor &governor,
                       Trainer &trainer,
                       AdapterRegistry &registry);

    void run();

private:
    std::string handle_generate(const std::string &params_json);
    std::string handle_retrieval(const std::string &params_json);
    std::string handle_build(const std::string &params_json);
    std::string handle_hot_swap(const std::string &params_json);

    DecoderModel &m_model;
    TokenizerWord &m_tokenizer;
    RetrievalIndex &m_retrieval;
    PolicyGovernor &m_governor;
    Trainer &m_trainer;
    AdapterRegistry &m_registry;
    MCPServer m_server;
};

} // namespace almondai

#endif // ALMONDAI_SERVE_HPP
