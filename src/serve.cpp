#include "almondai/serve.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>

#include "almondai/buildparse.hpp"
#include "almondai/model.hpp"
#include "almondai/retrieval.hpp"
#include "almondai/tokenizer_word.hpp"
#include "almondai/governor.hpp"
#include "almondai/train.hpp"

namespace almondai {

namespace {
std::string extract_string_field(const std::string &json, const std::string &key) {
    auto pos = json.find('"' + key + '"');
    if (pos == std::string::npos) {
        return {};
    }
    auto colon = json.find(':', pos);
    auto quote_start = json.find('"', colon + 1);
    auto quote_end = json.find('"', quote_start + 1);
    if (quote_start == std::string::npos || quote_end == std::string::npos) {
        return {};
    }
    return json.substr(quote_start + 1, quote_end - quote_start - 1);
}

std::vector<std::string> extract_array(const std::string &json, const std::string &key) {
    std::vector<std::string> values;
    auto pos = json.find('"' + key + '"');
    if (pos == std::string::npos) {
        return values;
    }
    auto start = json.find('[', pos);
    auto end = json.find(']', start);
    if (start == std::string::npos || end == std::string::npos || end <= start) {
        return values;
    }
    std::string slice = json.substr(start + 1, end - start - 1);
    std::string current;
    bool in_string = false;
    for (char ch : slice) {
        if (ch == '"') {
            if (in_string) {
                values.push_back(current);
                current.clear();
            }
            in_string = !in_string;
        } else if (in_string) {
            current.push_back(ch);
        }
    }
    return values;
}

std::string escape(const std::string &text) {
    std::ostringstream oss;
    for (char ch : text) {
        if (ch == '"' || ch == '\\') {
            oss << '\\' << ch;
        } else if (ch == '\n') {
            oss << "\\n";
        } else {
            oss << ch;
        }
    }
    return oss.str();
}
}

ConsoleApplication::ConsoleApplication(DecoderModel &model,
                                       TokenizerWord &tokenizer,
                                       RetrievalIndex &retrieval,
                                       PolicyGovernor &governor,
                                       Trainer &trainer,
                                       AdapterRegistry &registry)
    : m_model(model),
      m_tokenizer(tokenizer),
      m_retrieval(retrieval),
      m_governor(governor),
      m_trainer(trainer),
      m_registry(registry) {
    m_server.register_handler("model.generate", [this](const std::string &params) { return handle_generate(params); });
    m_server.register_handler("retrieval.query", [this](const std::string &params) { return handle_retrieval(params); });
    m_server.register_handler("compiler.build", [this](const std::string &params) { return handle_build(params); });
    m_server.register_handler("admin.hot_swap", [this](const std::string &params) { return handle_hot_swap(params); });
}

void ConsoleApplication::run() {
    m_server.serve(std::cin, std::cout);
}

std::string ConsoleApplication::handle_generate(const std::string &params_json) {
    GenerationOptions options;
    std::string prompt = extract_string_field(params_json, "prompt");
    auto schema = extract_string_field(params_json, "schema");
    if (!schema.empty()) {
        options.schema_json = schema;
    }
    options.blocklist_tokens = extract_array(params_json, "blocklist");
    auto output = m_model.generate(prompt, m_tokenizer, m_retrieval, m_governor, options);
    return std::string("{\"output\":\"") + escape(output) + "\"}";
}

std::string ConsoleApplication::handle_retrieval(const std::string &params_json) {
    std::string query = extract_string_field(params_json, "query");
    auto results = m_retrieval.query(query, m_tokenizer, 3);
    std::ostringstream oss;
    oss << "{\"results\":[";
    for (std::size_t i = 0; i < results.size(); ++i) {
        oss << "{\"score\":" << results[i].score << ",\"document\":\"" << escape(results[i].document) << "\"}";
        if (i + 1 < results.size()) {
            oss << ',';
        }
    }
    oss << "]}";
    return oss.str();
}

std::string ConsoleApplication::handle_build(const std::string &params_json) {
    std::string type = extract_string_field(params_json, "type");
    std::string path = extract_string_field(params_json, "path");
    std::vector<Diagnostic> diagnostics;
    if (type == "clang") {
        diagnostics = parse_clang_diagnostics_json(path);
    } else {
        diagnostics = parse_msbuild_log(path);
    }
    return build_diagnostics_response(diagnostics);
}

std::string ConsoleApplication::handle_hot_swap(const std::string &params_json) {
    std::string adapter_name = extract_string_field(params_json, "adapter");
    if (!adapter_name.empty()) {
        m_registry.promote(adapter_name);
    }
    return std::string("{\"active\":\"") + escape(adapter_name) + "\"}";
}

} // namespace almondai
