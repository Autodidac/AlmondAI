/**************************************************************
 *   █████╗ ██╗     ███╗   ███╗   ███╗   ██╗    ██╗██████╗    *
 *  ██╔══██╗██║     ████╗ ████║ ██╔═══██╗████╗  ██║██╔══██╗   *
 *  ███████║██║     ██╔████╔██║ ██║   ██║██╔██╗ ██║██║  ██║   *
 *  ██╔══██║██║     ██║╚██╔╝██║ ██║   ██║██║╚██╗██║██║  ██║   *
 *  ██║  ██║███████╗██║ ╚═╝ ██║ ╚██████╔╝██║ ╚████║██████╔╝   *
 *  ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═════╝ ╚═╝  ╚═══╝╚═════╝    *
 *                                                            *
 *   This file is part of the Almond Project.                 *
 *   AlmondAI - Modular C++ Framework                       *
 *                                                            *
 *   SPDX-License-Identifier: LicenseRef-MIT-NoSell           *
 *                                                            *
 *   Provided "AS IS", without warranty of any kind.          *
 *   Use permitted for non-commercial purposes only           *
 *   without prior commercial licensing agreement.            *
 *                                                            *
 *   Redistribution allowed with this notice.                 *
 *   No obligation to disclose modifications.                 *
 *   See LICENSE file for full terms.                         *
 **************************************************************/
 // main.cpp - AlmondAI runtime console demonstration

#include "../../../AlmondAI/include/almondai/serve.hpp"
#include "../../../AlmondAI/include/almondai/adapter.hpp"
#include "../../../AlmondAI/include/almondai/tokenizer_word.hpp"
#include "../../../AlmondAI/include/almondai/json.hpp"
#include "../../../AlmondAI/include/almondai/chat/backend.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>
#include <unordered_set>
#include <streambuf>

namespace {

std::optional<std::string> read_env(std::string_view name) {
    std::string name_str{name};
#if defined(_WIN32)
    size_t length = 0;
    char* buffer = nullptr;
    if (_dupenv_s(&buffer, &length, name_str.c_str()) != 0 || buffer == nullptr) {
        return std::nullopt;
    }
    std::string value(buffer);
    free(buffer);
    return value;
#else
    const char* value = std::getenv(name_str.c_str());
    if (!value) {
        return std::nullopt;
    }
    return std::string(value);
#endif
}

std::string read_env_string(std::string_view name) {
    auto value = read_env(name);
    return value ? *value : std::string();
}

class CallbackStreambuf : public std::streambuf {
public:
    using ChunkHandler = std::function<void(std::string_view)>;

    explicit CallbackStreambuf(ChunkHandler handler)
        : m_handler(std::move(handler)) {}

protected:
    int_type overflow(int_type ch) override {
        if (traits_type::eq_int_type(ch, traits_type::eof())) {
            flush_buffer();
            return traits_type::not_eof(ch);
        }

        m_buffer.push_back(static_cast<char>(ch));
        if (ch == '\n') {
            flush_buffer();
        }
        return ch;
    }

    int sync() override {
        flush_buffer();
        return 0;
    }

private:
    void flush_buffer() {
        if (m_buffer.empty()) {
            return;
        }
        if (m_handler) {
            m_handler(std::string_view(m_buffer));
        }
        m_buffer.clear();
    }

    std::string m_buffer;
    ChunkHandler m_handler;
};

} // namespace

int main() {
    using namespace almondai;

    WordTokenizer tokenizer;
    const std::filesystem::path vocab_path = "data/vocab.txt";
    if (std::filesystem::exists(vocab_path)) {
        tokenizer.load_vocab(vocab_path.string());
    }
    else {
        tokenizer.save_vocab(vocab_path.string());
    }

    ModelConfig config;
    config.vocab_size = tokenizer.vocab().size();
    config.hidden_size = 64;
    config.num_layers = 2;

    auto trim_copy = [](std::string value) {
        auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
        value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
        value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
        return value;
    };
    auto lowercase_copy = [](std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return value;
    };

    const std::string fast_env = trim_copy(read_env_string("ALMONDAI_FAST_LEARNING"));
    const std::string rate_env = trim_copy(read_env_string("ALMONDAI_LEARNING_RATE"));

    auto bool_from_env = [&](const std::string& value) {
        const std::string lowered = lowercase_copy(value);
        return lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on";
    };

    const bool fast_learning = !fast_env.empty() && bool_from_env(fast_env);
    if (!rate_env.empty()) {
        char* end = nullptr;
        const double parsed = std::strtod(rate_env.c_str(), &end);
        if (end != rate_env.c_str() && std::isfinite(parsed) && parsed > 0.0) {
            config.learning_rate = parsed;
            std::ostringstream msg;
            msg << std::fixed << std::setprecision(6) << parsed;
            std::cout << "Learning rate set to " << msg.str()
                      << " via ALMONDAI_LEARNING_RATE.\n";
        }
        else {
            std::cout << "Ignoring invalid ALMONDAI_LEARNING_RATE value: '"
                      << rate_env << "'.\n";
        }
    }
    else if (fast_learning) {
        const double fast_rate = std::max(config.learning_rate, 5e-3);
        config.learning_rate = fast_rate;
        std::ostringstream msg;
        msg << std::fixed << std::setprecision(6) << fast_rate;
        std::cout << "ALMONDAI_FAST_LEARNING enabled (learning rate "
                  << msg.str() << ").\n";
    }

    BaseDecoder base(config);
    StudentModel student(std::move(base));

    AdapterManager adapter_manager;
    AdapterConfig adapter_config;
    adapter_manager.register_adapter(Adapter("default", config.hidden_size, adapter_config));
    adapter_manager.activate("default");

    PolicyGovernor governor;
    governor.set_blocklist({ "forbidden", "classified" });

    bool boot_ready = false;
    auto load_status_logger = [&](const LoadStatus& status) {
        if (status.phase == "ready") {
            boot_ready = true;
            std::cout << "\033[2J\033[H";
            std::cout << "[engine] " << status.phase;
            if (!status.detail.empty()) {
                std::cout << ": " << status.detail;
            }
            if (status.total > 0) {
                std::cout << " (" << status.completed << '/' << status.total << ')';
            }
            else if (status.completed > 0) {
                std::cout << " (" << status.completed << ')';
            }
            std::cout << '\n';
            std::cout.flush();
            return;
        }

        if (!boot_ready) {
            std::cout << "[engine] " << status.phase;
            if (!status.detail.empty()) {
                std::cout << ": " << status.detail;
            }
            if (status.total > 0) {
                std::cout << " (" << status.completed << '/' << status.total << ')';
            }
            else if (status.completed > 0) {
                std::cout << " (" << status.completed << ')';
            }
            std::cout << '\n';
            std::cout.flush();
        }
    };

    ContinuousLearner learner(std::move(student),
        std::move(adapter_manager),
        std::move(tokenizer),
        std::move(governor),
        load_status_logger);
    learner.promote_adapter("default");

    MCPBridge bridge;
    Service service(learner, bridge);

    almondai::chat::BackendPtr chat_backend;
    std::optional<almondai::chat::Kind> chat_kind;
    std::string chat_route_label;
    bool auto_train_remote = false;

    auto to_lower = [](std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return value;
    };

    auto normalize_endpoint = [](std::string value) {
        auto trim_in_place = [](std::string& text) {
            auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
            text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
            text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
        };

        auto strip_query_fragment = [](std::string& text) {
            const std::size_t pos = text.find_first_of("?#");
            if (pos != std::string::npos) {
                text.erase(pos);
            }
        };

        auto remove_trailing_slashes = [](std::string& text) {
            while (!text.empty() && text.back() == '/') {
                text.pop_back();
            }
        };

        auto lower_copy = [](const std::string& text) {
            std::string lower(text);
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return lower;
        };

        trim_in_place(value);
        if (value.empty()) {
            return value;
        }

        if (value.rfind("http://", 0) != 0 && value.rfind("https://", 0) != 0) {
            value = "http://" + value;
        }

        strip_query_fragment(value);
        remove_trailing_slashes(value);

        const std::string completions = "/v1/chat/completions";
        std::string lower = lower_copy(value);
        const std::size_t completions_pos = lower.find(completions);
        if (completions_pos != std::string::npos) {
            value = value.substr(0, completions_pos + completions.size());
            remove_trailing_slashes(value);
            return value;
        }

        const std::array<std::string, 3> patterns = {"/v1/chat/completions", "/v1/models", "/v1"};
        for (const auto& pattern : patterns) {
            lower = lower_copy(value);
            const std::size_t pos = lower.find(pattern);
            if (pos != std::string::npos) {
                value = value.substr(0, pos);
                remove_trailing_slashes(value);
            }
        }

        remove_trailing_slashes(value);
        if (!value.empty()) {
            value += completions;
        }
        return value;
    };

    auto is_openai_compatible = [](almondai::chat::Kind kind) {
        switch (kind) {
        case almondai::chat::Kind::OpenAICompat:
        case almondai::chat::Kind::H2O:
        case almondai::chat::Kind::HuggingFace:
        case almondai::chat::Kind::OpenRouter:
        case almondai::chat::Kind::TogetherAI:
        case almondai::chat::Kind::DeepInfra:
            return true;
        default:
            return false;
        }
    };

    std::string env_kind = read_env_string("ALMONDAI_CHAT_KIND");
    std::string env_endpoint = read_env_string("ALMONDAI_ENDPOINT");
    std::string env_model = read_env_string("ALMONDAI_MODEL");
    std::string env_key = read_env_string("ALMONDAI_API_KEY");

    const std::string gpt_endpoint = read_env_string("ALMONDAI_GPT_ENDPOINT");
    const std::string gpt_model = read_env_string("ALMONDAI_GPT_MODEL");
    const std::string gpt_key = read_env_string("ALMONDAI_GPT_API_KEY");

    if (env_kind.empty() && (!gpt_endpoint.empty() || !gpt_model.empty() || !gpt_key.empty())) {
        env_kind = "openai";
        if (env_endpoint.empty()) {
            env_endpoint = gpt_endpoint.empty() ? "https://api.openai.com/v1/chat/completions" : gpt_endpoint;
        }
        if (env_model.empty()) {
            env_model = gpt_model.empty() ? "gpt-4o-mini" : gpt_model;
        }
        if (env_key.empty()) {
            env_key = gpt_key;
        }
    }

    if (!env_kind.empty()) {
        const std::string lowered_kind = to_lower(env_kind);
        if (lowered_kind == "lmstudio") {
            if (env_endpoint.empty()) {
                env_endpoint = "http://127.0.0.1:1234/v1/chat/completions";
            }
            if (env_model.empty()) {
                env_model = "lmstudio";
            }
        }
        try {
            almondai::chat::Kind parsed = almondai::chat::parse_kind(env_kind);
            if (is_openai_compatible(parsed) && !env_endpoint.empty()) {
                env_endpoint = normalize_endpoint(env_endpoint);
            }
            const std::string route_label = lowered_kind == "lmstudio"
                ? std::string{"lmstudio"}
                : almondai::chat::kind_to_string(parsed);
            chat_backend = almondai::chat::make_backend(parsed, env_endpoint, env_model, env_key);
            chat_kind = parsed;
            chat_route_label = route_label;
            auto_train_remote = (lowered_kind == "lmstudio");
            service.set_chat_backend(chat_backend.get(), route_label);
            std::cout << "Connected to " << route_label
                      << " chat backend from environment configuration.\n";
            if (auto_train_remote) {
                std::cout << "LM Studio auto-training enabled. Remote replies will update the local student.\n";
            }
        }
        catch (const std::exception& ex) {
            std::cout << "Failed to configure chat backend from environment: " << ex.what() << "\n";
        }
    }

    std::cout << "AlmondAI interactive console\n"
        "Type plain text to chat with the active model or backend.\n"
        "Prefix commands with '/' or use the keywords below (e.g. 'help').\n"
        "Type 'help' to list available commands or 'exit' to quit.\n"
        "Use '\\' before a message when you need to send text that matches a command name.\n"
        "Use 'chat use' to connect to an external backend or 'chat clear' to return to local inference.\n"
        "Use 'chat use lmstudio' to auto-connect to a local LM Studio instance.\n"
        "Remote LM Studio replies automatically train the student model.\n";

    auto trim = [](std::string& text) {
        auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
        text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
        text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
        };

    // Robust service bridge:
    // 1) Try parsing the entire buffer as one JSON object.
    // 2) If that fails, scan all non-empty lines and parse the last valid JSON.
    // 3) Fall back to treating any remaining text as a "text" field so legacy
    //    services that only stream plain strings still echo something useful.
    auto invoke_service = [&service](const std::string& method, Json params) -> std::optional<Json> {
        JsonObject request;
        request["id"] = Json("cli");
        request["method"] = Json(method); // If backend expects "generate"/"retrieve"/"hot-swap", adjust call sites below.
        request["params"] = std::move(params);

        std::istringstream input(Json(request).dump() + "\n");
        std::ostringstream output;
        service.run(input, output);

        const std::string buf = output.str();
        if (buf.empty()) {
            return std::nullopt;
        }

        auto wrap_text_result = [](std::string text) -> Json {
            JsonObject obj;
            obj["text"] = Json(std::move(text));
            return Json(obj);
        };

        auto try_extract_result = [&wrap_text_result](const Json& response) -> std::optional<Json> {
            if (response.is_object()) {
                const auto& obj = response.as_object();
                if (auto err = obj.find("error"); err != obj.end()) {
                    const Json& payload = err->second;
                    if (payload.is_object()) {
                        const auto& err_obj = payload.as_object();
                        if (auto it = err_obj.find("message"); it != err_obj.end() && it->second.is_string()) {
                            throw std::runtime_error(it->second.as_string());
                        }
                    }
                    throw std::runtime_error("service returned an error");
                }
                if (auto res = obj.find("result"); res != obj.end()) {
                    return res->second;
                }
                if (obj.find("output") != obj.end() || obj.find("text") != obj.end()) {
                    return Json(response);
                }
                return std::nullopt;
            }
            if (response.is_array()) {
                return response;
            }
            if (response.is_string()) {
                return wrap_text_result(response.as_string());
            }
            return std::nullopt;
        };

        std::string fallback_text;

        // Strategy A: whole buffer is one JSON
        try {
            Json whole = Json::parse(buf);
            if (auto r = try_extract_result(whole)) return r;
        }
        catch (...) {
            // fall through
        }

        // Strategy B: scan lines for last valid JSON with "result"
        std::istringstream resp(buf);
        std::string line;
        std::optional<Json> last_result;
        while (std::getline(resp, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            try {
                Json j = Json::parse(line);
                if (auto r = try_extract_result(j)) last_result = std::move(r);
            }
            catch (...) {
                if (!fallback_text.empty()) fallback_text.append("\n");
                fallback_text.append(line);
            }
        }
        if (last_result) {
            return last_result;
        }
        if (!fallback_text.empty()) {
            return wrap_text_result(fallback_text);
        }
        return std::nullopt;
        };

    auto invoke_service_streaming = [&service](const std::string& method, Json params, Json* out_result) -> bool {
        JsonObject request;
        request["id"] = Json("cli");
        request["method"] = Json(method);
        request["params"] = std::move(params);

        auto parse_int = [](const Json& value, int fallback) -> int {
            return std::visit([
                                   fallback
                               ](const auto& raw) -> int {
                using T = std::decay_t<decltype(raw)>;
                if constexpr (std::is_same_v<T, double>) {
                    return static_cast<int>(raw);
                }
                else if constexpr (std::is_same_v<T, bool>) {
                    return raw ? 1 : 0;
                }
                else if constexpr (std::is_same_v<T, std::string>) {
                    try {
                        std::size_t idx = 0;
                        int parsed = std::stoi(raw, &idx);
                        if (idx == raw.size()) {
                            return parsed;
                        }
                    }
                    catch (...) {
                    }
                    return fallback;
                }
                else {
                    return fallback;
                }
            }, value.value());
        };

        auto parse_double = [](const Json& value, double fallback) -> double {
            return std::visit([
                                   fallback
                               ](const auto& raw) -> double {
                using T = std::decay_t<decltype(raw)>;
                if constexpr (std::is_same_v<T, double>) {
                    return raw;
                }
                else if constexpr (std::is_same_v<T, bool>) {
                    return raw ? 1.0 : 0.0;
                }
                else if constexpr (std::is_same_v<T, std::string>) {
                    try {
                        std::size_t idx = 0;
                        double parsed = std::stod(raw, &idx);
                        if (idx == raw.size()) {
                            return parsed;
                        }
                    }
                    catch (...) {
                    }
                    return fallback;
                }
                else {
                    return fallback;
                }
            }, value.value());
        };

        auto parse_label = [](const JsonObject& obj, std::string fallback = "train") {
            if (auto it = obj.find("label"); it != obj.end() && it->second.is_string()) {
                std::string label = it->second.as_string();
                if (!label.empty()) {
                    return label;
                }
            }
            return fallback;
        };

        std::istringstream input(Json(request).dump() + "\n");
        std::size_t last_width = 0;
        bool saw_batch = false;
        bool saw_result = false;
        double final_loss = 0.0;
        int final_step = 0;
        std::string fallback_text;
        std::string pending;
        bool received_any = false;
        std::optional<Json> captured_result;

        auto handle_line = [&](std::string raw) {
            if (!raw.empty() && raw.back() == '\r') raw.pop_back();
            if (raw.empty()) {
                return;
            }

            Json parsed;
            try {
                parsed = Json::parse(raw);
            }
            catch (...) {
                if (!fallback_text.empty()) fallback_text.append("\n");
                fallback_text.append(raw);
                return;
            }

            if (!parsed.is_object()) {
                if (!fallback_text.empty()) fallback_text.append("\n");
                fallback_text.append(raw);
                return;
            }

            const auto& obj = parsed.as_object();
            if (auto err_it = obj.find("error"); err_it != obj.end()) {
                if (err_it->second.is_object()) {
                    const auto& err_obj = err_it->second.as_object();
                    if (auto msg_it = err_obj.find("message"); msg_it != err_obj.end() && msg_it->second.is_string()) {
                        throw std::runtime_error(msg_it->second.as_string());
                    }
                }
                throw std::runtime_error("service returned an error");
            }

            if (auto event_it = obj.find("event"); event_it != obj.end() && event_it->second.is_string()) {
                const std::string event = event_it->second.as_string();
                if (event == "batch") {
                    int step_value = 0;
                    double loss_value = 0.0;
                    if (auto step_it = obj.find("step"); step_it != obj.end()) {
                        step_value = parse_int(step_it->second, step_value);
                    }
                    if (auto loss_it = obj.find("loss"); loss_it != obj.end()) {
                        loss_value = parse_double(loss_it->second, loss_value);
                    }
                    double acc_value = 0.0;
                    bool have_accuracy = false;
                    if (auto acc_it = obj.find("accuracy"); acc_it != obj.end()) {
                        acc_value = parse_double(acc_it->second, acc_value);
                        have_accuracy = true;
                    }
                    const std::string label = parse_label(obj);
                    final_step = step_value;
                    final_loss = loss_value;
                    std::ostringstream progress;
                    progress << '[' << label << "] step " << step_value
                             << " loss " << std::fixed << std::setprecision(4) << loss_value;
                    if (have_accuracy) {
                        progress << " acc " << std::fixed << std::setprecision(4) << acc_value;
                    }
                    const std::string text = progress.str();
                    std::cout << '\r' << text;
                    if (last_width > text.size()) {
                        std::cout << std::string(last_width - text.size(), ' ');
                    }
                    std::cout << std::flush;
                    last_width = text.size();
                    saw_batch = true;
                    return;
                }
                if (event == "info") {
                    std::string message;
                    if (auto msg_it = obj.find("message"); msg_it != obj.end() && msg_it->second.is_string()) {
                        message = msg_it->second.as_string();
                    }
                    if (!message.empty()) {
                        if (saw_batch && last_width > 0) {
                            std::cout << '\r' << std::string(last_width, ' ') << '\r';
                            last_width = 0;
                        }
                        const std::string label = parse_label(obj);
                        std::cout << '[' << label << "] " << message << '\n';
                        std::cout.flush();
                    }
                    return;
                }
            }

            if (auto result_it = obj.find("result"); result_it != obj.end() && result_it->second.is_object()) {
                captured_result = result_it->second;
                const auto& result = result_it->second.as_object();
                if (auto loss_it = result.find("final_loss"); loss_it != result.end()) {
                    final_loss = parse_double(loss_it->second, final_loss);
                }
                if (auto steps_it = result.find("steps"); steps_it != result.end()) {
                    final_step = parse_int(steps_it->second, final_step);
                }
                saw_result = true;
            }
        };

        CallbackStreambuf buffer([&](std::string_view chunk) {
            received_any = true;
            pending.append(chunk.begin(), chunk.end());
            std::size_t start = 0;
            while (true) {
                const std::size_t pos = pending.find('\n', start);
                if (pos == std::string::npos) {
                    break;
                }
                std::string line = pending.substr(start, pos - start);
                handle_line(std::move(line));
                start = pos + 1;
            }
            if (start > 0) {
                pending.erase(0, start);
            }
        });

        std::ostream output(&buffer);
        service.run(input, output);
        output.flush();
        buffer.pubsync();

        if (!pending.empty()) {
            handle_line(std::move(pending));
            pending.clear();
        }

        if (!received_any) {
            return false;
        }

        if (saw_result) {
            if (saw_batch && last_width > 0) {
                std::cout << '\r' << std::string(last_width, ' ') << '\r';
            }
            std::string summary_text;
            std::string summary_label = "train";
            if (captured_result && captured_result->is_object()) {
                const auto& result_obj = captured_result->as_object();
                summary_label = parse_label(result_obj, summary_label);
                if (auto out_it = result_obj.find("output"); out_it != result_obj.end() && out_it->second.is_string()) {
                    summary_text = out_it->second.as_string();
                }
            }
            if (summary_text.empty()) {
                std::ostringstream summary;
                summary << "Training complete (loss=" << std::fixed << std::setprecision(4) << final_loss
                        << ", steps=" << final_step << ')';
                summary_text = summary.str();
            }
            if (summary_label != "train") {
                std::cout << '[' << summary_label << "] " << summary_text << '\n';
            } else {
                std::cout << summary_text << '\n';
            }
            std::cout.flush();
            if (out_result && captured_result) {
                *out_result = *captured_result;
            }
        }
        else if (saw_batch) {
            if (last_width > 0) {
                std::cout << '\r' << std::string(last_width, ' ') << '\r';
            }
            std::cout.flush();
            return true;
        }

        if (!fallback_text.empty()) {
            std::cout << fallback_text << '\n';
            std::cout.flush();
            return true;
        }

        return saw_result || saw_batch;
    };

    struct UsageStats {
        std::optional<int> prompt_tokens;
        std::optional<int> completion_tokens;
        std::optional<int> total_tokens;
    };

    struct TurnRecord {
        std::size_t turn_id = 0;
        std::string prompt;
        std::string response;
        std::string route;
        std::string backend;
        bool remote_route = false;
        bool allowed_for_training = true;
        bool auto_training_attempted = false;
        bool auto_training_success = false;
        bool implicit = false;
        std::vector<std::string> notes;
        UsageStats usage;
    };

    auto parse_optional_int = [](const Json& value) -> std::optional<int> {
        return std::visit(
            [](const auto& raw) -> std::optional<int> {
                using T = std::decay_t<decltype(raw)>;
                if constexpr (std::is_same_v<T, double>) {
                    return static_cast<int>(raw);
                }
                else if constexpr (std::is_same_v<T, bool>) {
                    return raw ? std::optional<int>{1} : std::optional<int>{0};
                }
                else if constexpr (std::is_same_v<T, std::string>) {
                    try {
                        std::size_t idx = 0;
                        int parsed = std::stoi(raw, &idx);
                        if (idx == raw.size()) {
                            return parsed;
                        }
                    }
                    catch (...) {
                    }
                    return std::nullopt;
                }
                else {
                    return std::nullopt;
                }
            },
            value.value());
    };

    auto parse_optional_double = [](const Json& value) -> std::optional<double> {
        return std::visit(
            [](const auto& raw) -> std::optional<double> {
                using T = std::decay_t<decltype(raw)>;
                if constexpr (std::is_same_v<T, double>) {
                    return raw;
                }
                else if constexpr (std::is_same_v<T, bool>) {
                    return raw ? std::optional<double>{1.0} : std::optional<double>{0.0};
                }
                else if constexpr (std::is_same_v<T, std::string>) {
                    try {
                        std::size_t idx = 0;
                        double parsed = std::stod(raw, &idx);
                        if (idx == raw.size()) {
                            return parsed;
                        }
                    }
                    catch (...) {
                    }
                    return std::nullopt;
                }
                else {
                    return std::nullopt;
                }
            },
            value.value());
    };

    auto parse_bool_value = [](const Json& value, bool fallback) -> bool {
        return std::visit(
            [fallback](const auto& raw) -> bool {
                using T = std::decay_t<decltype(raw)>;
                if constexpr (std::is_same_v<T, bool>) {
                    return raw;
                }
                else if constexpr (std::is_same_v<T, double>) {
                    return raw != 0.0;
                }
                else if constexpr (std::is_same_v<T, std::string>) {
                    std::string lowered = raw;
                    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
                        return static_cast<char>(std::tolower(ch));
                    });
                    if (lowered == "true" || lowered == "1" || lowered == "yes") {
                        return true;
                    }
                    if (lowered == "false" || lowered == "0" || lowered == "no") {
                        return false;
                    }
                    return fallback;
                }
                else {
                    return fallback;
                }
            },
            value.value());
    };

    auto format_usage = [](const UsageStats& usage) -> std::string {
        std::ostringstream oss;
        bool first = true;
        if (usage.prompt_tokens) {
            oss << (first ? "" : ", ") << "prompt=" << *usage.prompt_tokens;
            first = false;
        }
        if (usage.completion_tokens) {
            oss << (first ? "" : ", ") << "completion=" << *usage.completion_tokens;
            first = false;
        }
        if (usage.total_tokens) {
            oss << (first ? "" : ", ") << "total=" << *usage.total_tokens;
            first = false;
        }
        return oss.str();
    };

    auto print_block = [](const std::string& label, const std::string& text) {
        if (text.empty()) {
            return;
        }
        std::cout << label << '\n';
        std::istringstream stream(text);
        std::string line;
        while (std::getline(stream, line)) {
            std::cout << "  " << line << '\n';
        }
    };

    std::vector<TurnRecord> conversation;
    std::size_t turn_counter = 1;

    auto handle_generation = [&](const std::string& prompt_text, bool implicit_chat) {
        std::string prompt = prompt_text;
        if (prompt.empty()) {
            std::cout << "Prompt cannot be empty.\n";
            return;
        }

        JsonObject params;
        params["prompt"] = Json(prompt);
        auto result = invoke_service("model.generate", Json(params));
        if (!result) {
            result = invoke_service("generate", Json(params));
        }

        TurnRecord record;
        record.turn_id = turn_counter++;
        record.prompt = prompt;
        record.implicit = implicit_chat;

        bool remote_route_used = false;
        std::string backend_label_for_training;
        std::string captured_output;
        bool have_output_text = false;
        bool allowed_for_training = true;
        std::string observed_route;
        std::vector<std::string> notes;

        if (!result) {
            notes.push_back("no response from service");
        }
        else if (result->is_object()) {
            const auto& obj = result->as_object();
            if (auto it = obj.find("output"); it != obj.end() && it->second.is_string()) {
                captured_output = it->second.as_string();
                have_output_text = true;
            }
            else if (auto t = obj.find("text"); t != obj.end() && t->second.is_string()) {
                captured_output = t->second.as_string();
                have_output_text = true;
            }
            if (auto route_it = obj.find("route"); route_it != obj.end() && route_it->second.is_string()) {
                observed_route = route_it->second.as_string();
                if (!observed_route.empty()) {
                    if (observed_route == "remote") {
                        remote_route_used = true;
                    }
                }
            }
            if (auto backend_it = obj.find("backend"); backend_it != obj.end() && backend_it->second.is_string()) {
                backend_label_for_training = backend_it->second.as_string();
            }
            if (auto allowed_it = obj.find("allowed"); allowed_it != obj.end()) {
                allowed_for_training = parse_bool_value(allowed_it->second, allowed_for_training);
            }
            if (auto err_it = obj.find("remote_error"); err_it != obj.end() && err_it->second.is_string()) {
                notes.push_back("remote error: " + err_it->second.as_string());
            }
            if (auto fallback_it = obj.find("fallback"); fallback_it != obj.end() && fallback_it->second.is_object()) {
                const auto& fb = fallback_it->second.as_object();
                if (auto fb_msg = fb.find("output"); fb_msg != fb.end() && fb_msg->second.is_string()) {
                    notes.push_back("fallback: " + fb_msg->second.as_string());
                }
            }
            if (auto usage_it = obj.find("usage"); usage_it != obj.end() && usage_it->second.is_object()) {
                const auto& usage_obj = usage_it->second.as_object();
                if (auto pt = usage_obj.find("prompt_tokens"); pt != usage_obj.end()) {
                    record.usage.prompt_tokens = parse_optional_int(pt->second);
                }
                if (auto ct = usage_obj.find("completion_tokens"); ct != usage_obj.end()) {
                    record.usage.completion_tokens = parse_optional_int(ct->second);
                }
                if (auto tt = usage_obj.find("total_tokens"); tt != usage_obj.end()) {
                    record.usage.total_tokens = parse_optional_int(tt->second);
                }
            }
            if (!have_output_text) {
                notes.push_back("service responded without textual output");
            }
        }
        else if (result->is_string()) {
            captured_output = result->as_string();
            have_output_text = !captured_output.empty();
            if (!have_output_text) {
                notes.push_back("empty response");
            }
        }
        else {
            notes.push_back("unexpected response format");
        }

        record.response = captured_output;
        record.route = observed_route;
        record.backend = backend_label_for_training;
        record.remote_route = remote_route_used;
        record.allowed_for_training = allowed_for_training;

        if (auto_train_remote && remote_route_used) {
            record.auto_training_attempted = true;
            if (allowed_for_training && have_output_text) {
                JsonObject train_params;
                train_params["prompt"] = Json(prompt);
                train_params["teacher_output"] = Json(captured_output);
                if (!backend_label_for_training.empty()) {
                    train_params["teacher_source"] = Json(backend_label_for_training);
                }
                else if (!chat_route_label.empty()) {
                    train_params["teacher_source"] = Json(chat_route_label);
                }
                else {
                    train_params["teacher_source"] = Json("lmstudio");
                }

                try {
                    auto train_result = invoke_service("train.step", Json(train_params));
                    if (train_result && train_result->is_object()) {
                        const auto& train_obj = train_result->as_object();
                        std::string status_text = "unknown";
                        if (auto status_it = train_obj.find("status"); status_it != train_obj.end() && status_it->second.is_string()) {
                            status_text = status_it->second.as_string();
                        }
                        bool have_loss_metric = false;
                        double loss_metric = 0.0;
                        if (auto loss_it = train_obj.find("loss"); loss_it != train_obj.end()) {
                            if (auto parsed = parse_optional_double(loss_it->second)) {
                                loss_metric = *parsed;
                                have_loss_metric = true;
                            }
                        }
                        bool have_accuracy_metric = false;
                        double accuracy_metric = 0.0;
                        if (auto acc_it = train_obj.find("accuracy"); acc_it != train_obj.end()) {
                            if (auto parsed = parse_optional_double(acc_it->second)) {
                                accuracy_metric = *parsed;
                                have_accuracy_metric = true;
                            }
                        }

                        std::ostringstream msg;
                        if (status_text == "trained") {
                            msg << "auto-train updated student";
                            record.auto_training_success = true;
                        }
                        else if (status_text == "skipped") {
                            msg << "auto-train skipped by curator";
                        }
                        else {
                            msg << "auto-train status: " << status_text;
                        }
                        std::vector<std::string> metrics;
                        if (have_loss_metric) {
                            std::ostringstream metric;
                            metric << std::fixed << std::setprecision(4) << "loss=" << loss_metric;
                            metrics.push_back(metric.str());
                        }
                        if (have_accuracy_metric) {
                            std::ostringstream metric;
                            metric << std::fixed << std::setprecision(3) << "acc=" << accuracy_metric;
                            metrics.push_back(metric.str());
                        }
                        if (!metrics.empty()) {
                            msg << " (";
                            for (std::size_t i = 0; i < metrics.size(); ++i) {
                                if (i != 0) {
                                    msg << ", ";
                                }
                                msg << metrics[i];
                            }
                            msg << ")";
                        }
                        notes.push_back(msg.str());
                    }
                    else {
                        notes.push_back("auto-train: no response");
                    }
                }
                catch (const std::exception& ex) {
                    notes.push_back(std::string("auto-train failed: ") + ex.what());
                }
            }
            else if (!allowed_for_training) {
                notes.push_back("auto-train skipped: output blocked by governor");
            }
            else if (!have_output_text) {
                notes.push_back("auto-train skipped: empty response");
            }
        }

        std::string route_display;
        if (!observed_route.empty()) {
            route_display = observed_route;
        }
        else {
            route_display = remote_route_used ? "remote" : "local";
        }
        if (remote_route_used) {
            if (!backend_label_for_training.empty()) {
                route_display += " · backend=" + backend_label_for_training;
            }
            else if (!chat_route_label.empty()) {
                route_display += " · backend=" + chat_route_label;
            }
        }

        std::cout << "\n=== Turn " << record.turn_id << " (" << (implicit_chat ? "chat" : "generate") << ") ===\n";
        print_block("[you]", prompt);
        if (have_output_text) {
            print_block("[assistant]", captured_output);
        }
        else {
            print_block("[assistant]", have_output_text ? captured_output : std::string("(no response)"));
        }
        std::cout << "[route] " << route_display << '\n';
        if (auto usage_text = format_usage(record.usage); !usage_text.empty()) {
            std::cout << "[usage] " << usage_text << '\n';
        }
        if (!notes.empty()) {
            for (const auto& note : notes) {
                std::cout << "[note] " << note << '\n';
            }
        }
        std::cout << "===============================\n";

        if (record.backend.empty() && remote_route_used) {
            record.backend = chat_route_label;
        }
        if (record.route.empty()) {
            record.route = remote_route_used ? "remote" : "local";
        }
        record.notes = notes;
        conversation.push_back(std::move(record));
    };

    std::string line;
    while (std::cout << "AlmondAI> " && std::getline(std::cin, line)) {
        trim(line);
        if (line.empty()) continue;

        bool force_prompt = false;
        if (!line.empty() && line.front() == '\\') {
            force_prompt = true;
            line.erase(line.begin());
            trim(line);
        }

        bool slash_command = false;
        if (!force_prompt && !line.empty() && line.front() == '/') {
            slash_command = true;
            line.erase(line.begin());
            trim(line);
        }

        if (line.empty()) {
            if (force_prompt) {
                std::cout << "Prompt cannot be empty.\n";
            }
            continue;
        }

        std::string original_line = line;
        std::istringstream iss(line);
        std::string command;
        iss >> command;
        if (command.empty()) {
            if (force_prompt) {
                handle_generation(original_line, true);
            }
            continue;
        }

        std::string lowered_command = to_lower(command);

        if (force_prompt) {
            handle_generation(original_line, true);
            continue;
        }

        if (lowered_command == "exit" || lowered_command == "quit") break;

        if (lowered_command == "help") {
            std::cout << R"(Available commands:
  help                    Show this message.
  generate <prompt>       Generate a completion and report the route/backend used.
  retrieve <query>        Search the retrieval index for relevant samples.
  reader [file] [limit] [offset]
                          Preview JSONL records (defaults to data/training_data.jsonl).
  directory [training]    Show absolute paths for the data/training files.
  train <file> [epochs=1] [batch=32]
                          Run batched training against a JSONL file.
  self-learn [loops=1] [delay_ms=0] [options]
                          Loop through seed prompts, ask the teacher, and train automatically.
                          Options: shuffle/random, ordered, force, keep/dedupe, tags=tag1,tag2.
  auto [loops=1] [delay_ms=0]
                          Quick self-learning loop with shuffle + dedupe defaults. Options: limit=N, tags=tag1,tag2.
  hot-swap [name]         Promote adapter <name> or rollback when omitted.
  chat use <kind> [endpoint] [model] [key]
                          Switch to an external chat backend. 'lmstudio' pre-fills
                          http://127.0.0.1:1234/v1/chat/completions and model 'lmstudio'.
                          Other kinds fall back to ALMONDAI_* environment variables.
                          With 'lmstudio' active, remote replies auto-train the student.
  chat clear              Return to local student model responses.
  history [clear]         Show or reset recorded turns and telemetry.
  \<text>                Force-send text that would otherwise look like a command.
  (chat)                  Type any other text to prompt the active model.
  exit | quit             Quit the console.
)";
            continue;
        }

        try {
            auto split_tags = [&](const std::string& value) {
                std::vector<std::string> parsed;
                std::unordered_set<std::string> seen;
                std::size_t start = 0;
                while (start <= value.size()) {
                    const std::size_t end = value.find(',', start);
                    std::string token = end == std::string::npos
                        ? value.substr(start)
                        : value.substr(start, end - start);
                    trim(token);
                    if (!token.empty()) {
                        std::string lowered = to_lower(token);
                        if (seen.insert(lowered).second) {
                            parsed.push_back(lowered);
                        }
                    }
                    if (end == std::string::npos) {
                        break;
                    }
                    start = end + 1;
                }
                return parsed;
            };

            auto run_self_loop = [&](int loops,
                                     int delay_ms,
                                     bool shuffle,
                                     bool force_new,
                                     std::optional<int> limit,
                                     const std::vector<std::string>& tags) {
                JsonObject params;
                params["loops"] = Json(loops);
                params["delay_ms"] = Json(delay_ms);
                params["shuffle"] = Json(shuffle);
                params["force_new"] = Json(force_new);
                if (limit && *limit > 0) {
                    params["limit"] = Json(*limit);
                }
                if (!tags.empty()) {
                    JsonArray tag_array;
                    tag_array.reserve(tags.size());
                    for (const auto& tag : tags) {
                        tag_array.emplace_back(Json(tag));
                    }
                    params["tags"] = Json(tag_array);
                }

                auto process_self_loop_result = [&](const JsonObject& obj, bool skip_output_line) {
                    if (!skip_output_line) {
                        if (auto out_it = obj.find("output"); out_it != obj.end() && out_it->second.is_string()) {
                            std::cout << out_it->second.as_string() << "\n";
                        }
                    }

                    auto get_int_field = [&](const char* key) -> std::optional<int> {
                        if (auto it = obj.find(key); it != obj.end()) {
                            if (auto parsed = parse_optional_int(it->second)) {
                                return parsed;
                            }
                        }
                        return std::nullopt;
                    };

                    auto get_double_field = [&](const char* key) -> std::optional<double> {
                        if (auto it = obj.find(key); it != obj.end()) {
                            return parse_optional_double(it->second);
                        }
                        return std::nullopt;
                    };

                    std::optional<int> loops_completed = get_int_field("loops_completed");
                    std::optional<int> loops_requested = get_int_field("loops_requested");
                    std::optional<int> processed = get_int_field("processed");
                    std::optional<int> trained = get_int_field("trained");
                    std::optional<int> skipped = get_int_field("skipped");
                    std::optional<int> unavailable = get_int_field("teacher_unavailable");
                    std::optional<double> avg_loss = get_double_field("average_loss");
                    std::optional<double> avg_accuracy = get_double_field("average_accuracy");

                    bool printed_summary = false;
                    if (loops_completed || loops_requested || processed || trained || skipped || unavailable) {
                        std::cout << "Summary: ";
                        printed_summary = true;
                        bool first_field = true;
                        if (loops_completed || loops_requested) {
                            std::cout << "loops=";
                            if (loops_completed) {
                                std::cout << *loops_completed;
                            }
                            if (loops_requested) {
                                if (!loops_completed || *loops_requested != *loops_completed) {
                                    std::cout << '/' << *loops_requested;
                                }
                            }
                            first_field = false;
                        }
                        auto append_field = [&](const char* label, const std::optional<int>& value) {
                            if (!value) {
                                return;
                            }
                            if (!first_field) {
                                std::cout << ", ";
                            }
                            first_field = false;
                            std::cout << label << '=' << *value;
                        };
                        append_field("processed", processed);
                        append_field("trained", trained);
                        append_field("skipped", skipped);
                        append_field("unavailable", unavailable);
                        std::cout << '\n';
                    }
                    if (avg_loss || avg_accuracy) {
                        std::vector<std::string> metrics;
                        if (avg_loss) {
                            std::ostringstream oss;
                            oss << std::fixed << std::setprecision(4) << *avg_loss;
                            metrics.push_back("loss=" + oss.str());
                        }
                        if (avg_accuracy) {
                            std::ostringstream oss;
                            oss << std::fixed << std::setprecision(3) << *avg_accuracy;
                            metrics.push_back("acc=" + oss.str());
                        }
                        if (!metrics.empty()) {
                            std::cout << "Averages: ";
                            for (std::size_t i = 0; i < metrics.size(); ++i) {
                                if (i != 0) {
                                    std::cout << ", ";
                                }
                                std::cout << metrics[i];
                            }
                            std::cout << '\n';
                        }
                    }

                    if (auto tags_it = obj.find("requested_tags"); tags_it != obj.end() && tags_it->second.is_array()) {
                        const auto& arr = tags_it->second.as_array();
                        if (!arr.empty()) {
                            std::cout << "Tags: ";
                            for (std::size_t idx = 0; idx < arr.size(); ++idx) {
                                if (idx != 0) {
                                    std::cout << ", ";
                                }
                                if (arr[idx].is_string()) {
                                    std::cout << arr[idx].as_string();
                                }
                            }
                            std::cout << '\n';
                        }
                    }

                    auto print_event_log = [&]() {
                        auto events_it = obj.find("events");
                        if (events_it == obj.end() || !events_it->second.is_array()) {
                            return;
                        }
                        const auto& events = events_it->second.as_array();
                        if (events.empty()) {
                            return;
                        }
                        std::cout << "Event log:\n";
                        for (const auto& entry : events) {
                            if (!entry.is_object()) {
                                continue;
                            }
                            const auto& event = entry.as_object();
                            std::optional<int> iteration = std::nullopt;
                            if (auto it = event.find("iteration"); it != event.end()) {
                                iteration = parse_optional_int(it->second);
                            }
                            std::string status;
                            if (auto it = event.find("status"); it != event.end() && it->second.is_string()) {
                                status = it->second.as_string();
                            }
                            std::string route;
                            if (auto it = event.find("teacher_route"); it != event.end() && it->second.is_string()) {
                                route = it->second.as_string();
                            }
                            std::string source;
                            if (auto it = event.find("teacher_source"); it != event.end() && it->second.is_string()) {
                                source = it->second.as_string();
                            }
                            std::string prompt_text;
                            if (auto it = event.find("prompt"); it != event.end() && it->second.is_string()) {
                                prompt_text = it->second.as_string();
                            }
                            std::string snippet = prompt_text;
                            const std::size_t max_snippet = 80;
                            if (snippet.size() > max_snippet) {
                                snippet = snippet.substr(0, max_snippet - 3) + "...";
                            }
                            auto loss = event.find("loss");
                            std::optional<double> loss_value = (loss != event.end()) ? parse_optional_double(loss->second) : std::nullopt;
                            auto acc = event.find("accuracy");
                            std::optional<double> acc_value = (acc != event.end()) ? parse_optional_double(acc->second) : std::nullopt;

                            std::cout << "- ";
                            if (iteration) {
                                std::cout << "[#" << *iteration << "] ";
                            }
                            if (!status.empty()) {
                                std::cout << status;
                            } else {
                                std::cout << "(unknown)";
                            }
                            if (!route.empty()) {
                                std::cout << " via " << route;
                            }
                            if (!source.empty()) {
                                std::cout << " source=" << source;
                            }
                            if (auto req_it = event.find("requested_tags"); req_it != event.end() && req_it->second.is_array()) {
                                const auto& arr = req_it->second.as_array();
                                if (!arr.empty()) {
                                    std::cout << " tags=[";
                                    for (std::size_t idx = 0; idx < arr.size(); ++idx) {
                                        if (idx != 0) {
                                            std::cout << ',';
                                        }
                                        if (arr[idx].is_string()) {
                                            std::cout << arr[idx].as_string();
                                        }
                                    }
                                    std::cout << ']';
                                }
                            }
                            if (loss_value) {
                                std::ostringstream metric;
                                metric << std::fixed << std::setprecision(4) << *loss_value;
                                std::cout << " loss=" << metric.str();
                            }
                            if (acc_value) {
                                std::ostringstream metric;
                                metric << std::fixed << std::setprecision(3) << *acc_value;
                                std::cout << " acc=" << metric.str();
                            }
                            if (auto sem_it = event.find("semantic_tags"); sem_it != event.end() && sem_it->second.is_array()) {
                                const auto& arr = sem_it->second.as_array();
                                if (!arr.empty()) {
                                    std::cout << " sample_tags=[";
                                    for (std::size_t idx = 0; idx < arr.size(); ++idx) {
                                        if (idx != 0) {
                                            std::cout << ',';
                                        }
                                        if (arr[idx].is_string()) {
                                            std::cout << arr[idx].as_string();
                                        }
                                    }
                                    std::cout << ']';
                                }
                            }
                            if (!snippet.empty()) {
                                std::cout << " | " << snippet;
                            }
                            std::cout << '\n';
                        }
                        if (auto truncated_it = obj.find("events_truncated"); truncated_it != obj.end()) {
                            if (parse_bool_value(truncated_it->second, false)) {
                                std::cout << "(event log truncated)\n";
                            }
                        }
                    };

                    print_event_log();

                    if (!printed_summary && !avg_loss && !avg_accuracy) {
                        std::cout << "Self-learning request completed.\n";
                    }
                };

                Json stream_result;
                if (invoke_service_streaming("train.self_loop", Json(params), &stream_result)) {
                    if (stream_result.is_object()) {
                        process_self_loop_result(stream_result.as_object(), true);
                    }
                    return;
                }

                auto response = invoke_service("train.self_loop", Json(params));
                if (!response) {
                    std::cout << "(no response)\n";
                    return;
                }

                if (!response->is_object()) {
                    if (response->is_string()) {
                        std::cout << response->as_string() << "\n";
                    }
                    else {
                        std::cout << "Unexpected response format.\n";
                    }
                    return;
                }

                process_self_loop_result(response->as_object(), false);
            };

            auto is_integer_token = [](const std::string& text) {
                if (text.empty()) {
                    return false;
                }
                std::size_t start = 0;
                if (text.front() == '+' || text.front() == '-') {
                    if (text.size() == 1) {
                        return false;
                    }
                    start = 1;
                }
                const auto offset = static_cast<std::string::difference_type>(start);
                return std::all_of(text.begin() + offset, text.end(), [](unsigned char ch) {
                    return std::isdigit(ch) != 0;
                });
            };

            if (lowered_command == "history") {
                std::string subcommand;
                if (iss >> subcommand) {
                    std::string lowered_sub = to_lower(subcommand);
                    if (lowered_sub == "clear") {
                        conversation.clear();
                        turn_counter = 1;
                        std::cout << "Chat history cleared.\n";
                    }
                    else {
                        std::cout << "Unknown history command.\n";
                    }
                }
                else {
                    if (conversation.empty()) {
                        std::cout << "No turns recorded yet.\n";
                    }
                    else {
                        for (const auto& turn : conversation) {
                            std::cout << "#" << turn.turn_id << " (" << (turn.implicit ? "chat" : "generate") << ")";
                            std::cout << " route=" << (turn.route.empty() ? "n/a" : turn.route);
                            if (!turn.backend.empty()) {
                                std::cout << " backend=" << turn.backend;
                            }
                            std::cout << '\n';
                            print_block("  [you]", turn.prompt);
                            if (!turn.response.empty()) {
                                print_block("  [assistant]", turn.response);
                            }
                            if (auto usage_text = format_usage(turn.usage); !usage_text.empty()) {
                                std::cout << "  [usage] " << usage_text << '\n';
                            }
                            if (!turn.notes.empty()) {
                                for (const auto& note : turn.notes) {
                                    std::cout << "  [note] " << note << '\n';
                                }
                            }
                            std::cout << '\n';
                        }
                    }
                }
                continue;
            }

            if (lowered_command == "directory" || lowered_command == "dir") {
                std::string scope;
                iss >> scope;
                std::string lowered_scope = to_lower(scope);
                if (!scope.empty() && lowered_scope != "training") {
                    std::cout << "Unknown directory scope. Try 'directory' or 'directory training'.\n";
                    continue;
                }

                auto resolve_path = [](const std::filesystem::path& candidate) {
                    std::error_code ec;
                    std::filesystem::path resolved = std::filesystem::weakly_canonical(candidate, ec);
                    if (ec) {
                        resolved = std::filesystem::absolute(candidate, ec);
                        if (ec) {
                            return candidate;
                        }
                    }
                    return resolved;
                };

                auto print_entry = [&](const std::string& label, const std::filesystem::path& relative) {
                    const std::filesystem::path resolved = resolve_path(relative);
                    std::error_code exists_ec;
                    const bool exists = std::filesystem::exists(relative, exists_ec);
                    std::cout << "  " << label << ": " << resolved.string();
                    if (!exists) {
                        std::cout << " (missing)";
                    }
                    std::cout << '\n';
                };

                const std::filesystem::path data_dir{"data"};
                std::cout << "Training files are stored here:\n";
                print_entry("data", data_dir);
                print_entry("training_data.jsonl", data_dir / "training_data.jsonl");
                print_entry("training_seed.jsonl", data_dir / "training_seed.jsonl");
                print_entry("training_log.txt", data_dir / "training_log.txt");
                continue;
            }

            if (lowered_command == "chat") {
                std::string subcommand;
                iss >> subcommand;
                std::string lowered_sub = to_lower(subcommand);
                if (lowered_sub == "use") {
                    std::vector<std::string> args;
                    std::string token;
                    while (iss >> token) {
                        args.push_back(token);
                    }
                    if (args.empty()) {
                        std::cout << "Usage: chat use <kind> <endpoint> [model] [key]\n";
                        continue;
                    }
                    std::string lowered_kind = to_lower(args[0]);
                    const bool allow_defaults = lowered_kind == "lmstudio";

                    std::string endpoint = args.size() >= 2 ? args[1] : std::string();
                    std::string model = args.size() >= 3 ? args[2] : std::string();
                    std::string api_key = args.size() >= 4 ? args[3] : std::string();

                    if (endpoint.empty()) {
                        if (allow_defaults) {
                            endpoint = "http://127.0.0.1:1234/v1/chat/completions";
                        }
                        else {
                            std::cout << "Endpoint required for chat backend.\n";
                            continue;
                        }
                    }

                    if (model.empty()) {
                        if (allow_defaults) {
                            model = "lmstudio";
                        }
                        else {
                            model = trim_copy(read_env_string("ALMONDAI_MODEL"));
                            if (model.empty()) {
                                std::cout << "Model required for chat backend.\n";
                                continue;
                            }
                        }
                    }

                    if (api_key.empty()) {
                        api_key = trim_copy(read_env_string("ALMONDAI_API_KEY"));
                    }

                    try {
                        almondai::chat::Kind kind = almondai::chat::parse_kind(args[0]);
                        if (is_openai_compatible(kind)) {
                            endpoint = normalize_endpoint(endpoint);
                        }
                        const std::string route_label = allow_defaults
                            ? std::string{"lmstudio"}
                            : almondai::chat::kind_to_string(kind);
                        chat_backend = almondai::chat::make_backend(kind, std::move(endpoint), std::move(model), std::move(api_key));
                        chat_kind = kind;
                        chat_route_label = route_label;
                        auto_train_remote = (lowered_kind == "lmstudio");
                        service.set_chat_backend(chat_backend.get(), route_label);
                        if (allow_defaults) {
                            std::cout << "Using lmstudio (OpenAI-compatible) chat backend.\n";
                        }
                        else {
                            std::cout << "Using " << almondai::chat::kind_to_string(kind)
                                      << " chat backend.\n";
                        }
                        if (auto_train_remote) {
                            std::cout << "LM Studio auto-training enabled. Remote replies will update the local student.\n";
                        }
                    }
                    catch (const std::exception& ex) {
                        std::cout << "Failed to configure chat backend: " << ex.what() << "\n";
                    }
                    continue;
                }
                if (lowered_sub == "clear") {
                    if (chat_kind) {
                        const std::string label = chat_route_label.empty()
                            ? almondai::chat::kind_to_string(*chat_kind)
                            : chat_route_label;
                        std::cout << "Chat backend '" << label
                                  << "' cleared. Local model enabled.\n";
                    }
                    else {
                        std::cout << "Chat backend cleared. Local model enabled.\n";
                    }
                    chat_backend.reset();
                    chat_kind.reset();
                    chat_route_label.clear();
                    auto_train_remote = false;
                    service.set_chat_backend(nullptr);
                    continue;
                }
                std::cout << "Usage: chat use <kind> <endpoint> [model] [key] or chat clear\n";
                continue;
            }

            if (lowered_command == "generate") {
                std::string prompt;
                std::getline(iss, prompt);
                trim(prompt);
                handle_generation(prompt, false);
                continue;
            }

            if (lowered_command == "retrieve") {
                std::string query;
                std::getline(iss, query);
                trim(query);
                if (query.empty()) { std::cout << "Query cannot be empty.\n"; continue; }

                JsonObject params;
                params["query"] = Json(query);

                auto result = invoke_service("retrieval.query", Json(params));
                if (!result) {
                    result = invoke_service("retrieve", Json(params));
                }
                if (result) {
                    if (result->is_object()) {
                        const auto& obj = result->as_object();
                        bool printed_summary = false;
                        if (auto out = obj.find("output"); out != obj.end() && out->second.is_string()) {
                            std::cout << out->second.as_string() << "\n";
                            printed_summary = true;
                        }
                        if (auto hits_it = obj.find("hits"); hits_it != obj.end() && hits_it->second.is_array()) {
                            const auto& arr = hits_it->second.as_array();
                            if (!printed_summary) {
                                if (arr.empty()) {
                                    std::cout << "No retrieval hits.\n";
                                } else {
                                    for (const auto& item : arr) {
                                        if (!item.is_object()) {
                                            continue;
                                        }
                                        const auto& hit = item.as_object();
                                        std::string id = hit.count("document_id") ? hit.at("document_id").as_string() : "<no-id>";
                                        double score = 0.0;
                                        if (auto score_it = hit.find("score"); score_it != hit.end()) {
                                            const auto& v = score_it->second.value();
                                            std::visit([&](const auto& x) {
                                                using T = std::decay_t<decltype(x)>;
                                                if constexpr (std::is_arithmetic_v<T>) {
                                                    score = static_cast<double>(x);
                                                } else if constexpr (std::is_same_v<T, std::string>) {
                                                    try { score = std::stod(x); } catch (...) {}
                                                }
                                            }, v);
                                        }
                                        std::cout << "- " << id << " (score: " << score << ")\n";
                                    }
                                }
                            }
                        } else if (!printed_summary) {
                            std::cout << "Unexpected response format.\n";
                        }
                    }
                    else if (result->is_array()) {
                        const auto& arr = result->as_array();
                        if (arr.empty()) {
                            std::cout << "No retrieval hits.\n";
                        }
                        else {
                            for (const auto& item : arr) {
                                const auto& obj = item.as_object();
                                std::string id = obj.count("document_id") ? obj.at("document_id").as_string() : "<no-id>";
                                double score = 0.0;
                                if (auto it = obj.find("score"); it != obj.end()) {
                                    const auto& v = it->second.value();
                                    std::visit([&](const auto& x) {
                                        using T = std::decay_t<decltype(x)>;
                                        if constexpr (std::is_arithmetic_v<T>) {
                                            score = static_cast<double>(x);
                                        } else if constexpr (std::is_same_v<T, std::string>) {
                                            try { score = std::stod(x); } catch (...) {}
                                        }
                                    }, v);
                                }
                                std::cout << "- " << id << " (score: " << score << ")\n";
                            }
                        }
                    }
                    else {
                        std::cout << "Unexpected response format.\n";
                    }
                }
                else {
                    std::cout << "(no response)\n";
                }
                continue;
            }

            if (lowered_command == "reader") {
                std::vector<std::string> args;
                std::string token;
                while (iss >> token) {
                    args.push_back(token);
                }

                std::string file = "data/training_data.jsonl";
                int limit = 5;
                int offset = 0;
                bool file_set = false;
                bool limit_set = false;
                bool offset_set = false;
                bool invalid = false;

                auto set_limit = [&](const std::string& value) -> bool {
                    try {
                        limit = std::stoi(value);
                        limit_set = true;
                        return true;
                    }
                    catch (...) {
                        std::cout << "Invalid limit value.\n";
                        return false;
                    }
                };

                auto set_offset = [&](const std::string& value) -> bool {
                    try {
                        offset = std::stoi(value);
                        offset_set = true;
                        return true;
                    }
                    catch (...) {
                        std::cout << "Invalid offset value.\n";
                        return false;
                    }
                };

                for (const auto& arg : args) {
                    std::string lowered = to_lower(arg);
                    if (lowered.rfind("limit=", 0) == 0) {
                        if (!set_limit(arg.substr(6))) {
                            invalid = true;
                            break;
                        }
                        continue;
                    }
                    if (lowered.rfind("offset=", 0) == 0) {
                        if (!set_offset(arg.substr(7))) {
                            invalid = true;
                            break;
                        }
                        continue;
                    }
                    if (is_integer_token(arg)) {
                        if (!limit_set) {
                            if (!set_limit(arg)) {
                                invalid = true;
                                break;
                            }
                            continue;
                        }
                        if (!offset_set) {
                            if (!set_offset(arg)) {
                                invalid = true;
                                break;
                            }
                            continue;
                        }
                        std::cout << "Unexpected numeric argument '" << arg << "'.\n";
                        invalid = true;
                        break;
                    }
                    if (!file_set) {
                        if (lowered == "seed" || lowered == "training_seed") {
                            file = "data/training_seed.jsonl";
                        }
                        else if (lowered == "data" || lowered == "training_data") {
                            file = "data/training_data.jsonl";
                        }
                        else {
                            file = arg;
                        }
                        file_set = true;
                        continue;
                    }
                    std::cout << "Unknown reader option '" << arg << "'.\n";
                    invalid = true;
                    break;
                }

                if (invalid) {
                    continue;
                }

                if (limit < 0) {
                    limit = 0;
                }
                if (offset < 0) {
                    offset = 0;
                }

                JsonObject params;
                params["file"] = Json(file);
                params["offset"] = Json(offset);
                params["limit"] = Json(limit);

                auto result = invoke_service("data.read", Json(params));
                if (!result) {
                    result = invoke_service("reader", Json(params));
                }

                if (!result) {
                    std::cout << "(no response)\n";
                    continue;
                }

                if (!result->is_object()) {
                    std::cout << "Unexpected response format.\n";
                    continue;
                }

                const auto& obj = result->as_object();
                if (auto summary = obj.find("output"); summary != obj.end() && summary->second.is_string()) {
                    std::cout << summary->second.as_string() << '\n';
                }

                if (auto records_it = obj.find("records"); records_it != obj.end() && records_it->second.is_array()) {
                    const auto& recs = records_it->second.as_array();
                    if (recs.empty()) {
                        std::cout << "(no entries)\n";
                    }
                    else {
                        for (const auto& item : recs) {
                            if (!item.is_object()) {
                                continue;
                            }
                            const auto& record = item.as_object();
                            int line_number = 0;
                            if (auto line_it = record.find("line"); line_it != record.end()) {
                                if (auto parsed = parse_optional_int(line_it->second)) {
                                    line_number = *parsed;
                                }
                            }

                            std::cout << '\n';
                            if (line_number > 0) {
                                std::cout << "Line " << line_number << ":\n";
                            }
                            else {
                                std::cout << "Record:\n";
                            }

                            bool printed_any = false;
                            if (auto prompt_it = record.find("prompt"); prompt_it != record.end() && prompt_it->second.is_string()) {
                                print_block("  [prompt]", prompt_it->second.as_string());
                                printed_any = true;
                            }
                            if (auto teacher_it = record.find("teacher_output"); teacher_it != record.end()
                                && teacher_it->second.is_string()) {
                                print_block("  [teacher]", teacher_it->second.as_string());
                                printed_any = true;
                            }
                            if (auto tags_it = record.find("tags"); tags_it != record.end()) {
                                std::vector<std::string> tag_strings;
                                if (tags_it->second.is_array()) {
                                    const auto& arr = tags_it->second.as_array();
                                    for (const auto& tag_value : arr) {
                                        if (tag_value.is_string()) {
                                            tag_strings.push_back(tag_value.as_string());
                                        }
                                    }
                                }
                                else if (tags_it->second.is_string()) {
                                    tag_strings.push_back(tags_it->second.as_string());
                                }
                                if (!tag_strings.empty()) {
                                    std::cout << "  [tags] ";
                                    for (std::size_t idx = 0; idx < tag_strings.size(); ++idx) {
                                        if (idx != 0) {
                                            std::cout << ", ";
                                        }
                                        std::cout << tag_strings[idx];
                                    }
                                    std::cout << '\n';
                                    printed_any = true;
                                }
                            }
                            if (auto constraints_it = record.find("constraints"); constraints_it != record.end()) {
                                std::cout << "  [constraints] " << constraints_it->second.dump() << '\n';
                                printed_any = true;
                            }
                            if (auto error_it = record.find("error"); error_it != record.end() && error_it->second.is_string()) {
                                std::cout << "  [error] " << error_it->second.as_string() << '\n';
                                printed_any = true;
                            }
                            if (!printed_any) {
                                if (auto raw_it = record.find("raw"); raw_it != record.end() && raw_it->second.is_string()) {
                                    print_block("  [raw]", raw_it->second.as_string());
                                }
                            }
                        }
                    }
                }
                else {
                    std::cout << "Unexpected response format.\n";
                }

                if (auto truncated_it = obj.find("truncated"); truncated_it != obj.end()) {
                    if (parse_bool_value(truncated_it->second, false)) {
                        std::cout << "\n(more records available - increase limit or adjust offset)\n";
                    }
                }
                if (auto next_it = obj.find("next_offset"); next_it != obj.end()) {
                    if (auto parsed = parse_optional_int(next_it->second)) {
                        std::cout << "Next offset: " << *parsed << "\n";
                    }
                }
                continue;
            }

            if (lowered_command == "auto" || lowered_command == "autopilot") {
                std::vector<std::string> args;
                std::string token;
                while (iss >> token) {
                    args.push_back(token);
                }

                int loops = 1;
                int delay_ms = 0;
                std::optional<int> limit;
                bool loops_set = false;
                bool delay_set = false;
                bool invalid = false;
                std::vector<std::string> tags;

                for (const auto& arg : args) {
                    if (!loops_set && is_integer_token(arg)) {
                        try {
                            loops = std::stoi(arg);
                        }
                        catch (...) {
                            std::cout << "Invalid loops value.\n";
                            invalid = true;
                            break;
                        }
                        loops_set = true;
                        continue;
                    }
                    if (!delay_set && is_integer_token(arg)) {
                        try {
                            delay_ms = std::stoi(arg);
                        }
                        catch (...) {
                            std::cout << "Invalid delay value.\n";
                            invalid = true;
                            break;
                        }
                        delay_set = true;
                        continue;
                    }

                    std::string lowered = to_lower(arg);
                    if (lowered.rfind("limit=", 0) == 0) {
                        try {
                            int parsed = std::stoi(lowered.substr(6));
                            if (parsed < 0) {
                                parsed = 0;
                            }
                            limit = parsed;
                        }
                        catch (...) {
                            std::cout << "Invalid limit value.\n";
                            invalid = true;
                            break;
                        }
                        continue;
                    }

                    if (lowered.rfind("tags=", 0) == 0) {
                        const std::string value = arg.substr(5);
                        auto parsed_tags = split_tags(value);
                        tags.insert(tags.end(), parsed_tags.begin(), parsed_tags.end());
                        continue;
                    }

                    std::cout << "Unknown option '" << arg << "'.\n";
                    invalid = true;
                    break;
                }

                if (invalid) {
                    continue;
                }

                if (!tags.empty()) {
                    std::unordered_set<std::string> dedupe(tags.begin(), tags.end());
                    tags.assign(dedupe.begin(), dedupe.end());
                }

                if (loops <= 0) {
                    std::cout << "Loops must be positive.\n";
                    continue;
                }
                if (delay_ms < 0) {
                    std::cout << "Delay must be non-negative.\n";
                    continue;
                }

                run_self_loop(loops, delay_ms, true, false, limit, tags);
                continue;
            }

            if (lowered_command == "self-learn" || lowered_command == "selflearn") {
                std::vector<std::string> args;
                std::string token;
                while (iss >> token) {
                    args.push_back(token);
                }

                int loops = 1;
                int delay_ms = 0;
                bool shuffle = false;
                bool force_new = false;
                bool loops_set = false;
                bool delay_set = false;
                bool invalid = false;
                std::optional<int> limit;
                std::vector<std::string> tags;

                for (const auto& arg : args) {
                    if (!loops_set && is_integer_token(arg)) {
                        try {
                            loops = std::stoi(arg);
                        }
                        catch (...) {
                            std::cout << "Invalid loops value.\n";
                            invalid = true;
                            break;
                        }
                        loops_set = true;
                        continue;
                    }
                    if (!delay_set && is_integer_token(arg)) {
                        try {
                            delay_ms = std::stoi(arg);
                        }
                        catch (...) {
                            std::cout << "Invalid delay value.\n";
                            invalid = true;
                            break;
                        }
                        delay_set = true;
                        continue;
                    }

                    std::string lowered = to_lower(arg);
                    if (lowered == "shuffle" || lowered == "random") {
                        shuffle = true;
                        continue;
                    }
                    if (lowered == "ordered") {
                        shuffle = false;
                        continue;
                    }
                    if (lowered == "force") {
                        force_new = true;
                        continue;
                    }
                    if (lowered == "dedupe" || lowered == "keep") {
                        force_new = false;
                        continue;
                    }

                    if (lowered.rfind("limit=", 0) == 0) {
                        try {
                            int parsed = std::stoi(lowered.substr(6));
                            if (parsed < 0) {
                                parsed = 0;
                            }
                            limit = parsed;
                        }
                        catch (...) {
                            std::cout << "Invalid limit value.\n";
                            invalid = true;
                            break;
                        }
                        continue;
                    }

                    if (lowered.rfind("tags=", 0) == 0) {
                        const std::string value = arg.substr(5);
                        auto parsed_tags = split_tags(value);
                        tags.insert(tags.end(), parsed_tags.begin(), parsed_tags.end());
                        continue;
                    }

                    std::cout << "Unknown option '" << arg << "'.\n";
                    invalid = true;
                    break;
                }

                if (invalid) {
                    continue;
                }

                if (!tags.empty()) {
                    std::unordered_set<std::string> dedupe(tags.begin(), tags.end());
                    tags.assign(dedupe.begin(), dedupe.end());
                }

                if (loops <= 0) {
                    std::cout << "Loops must be positive.\n";
                    continue;
                }
                if (delay_ms < 0) {
                    std::cout << "Delay must be non-negative.\n";
                    continue;
                }

                run_self_loop(loops, delay_ms, shuffle, force_new, limit, tags);
                continue;
            }

            if (lowered_command == "train") {
                std::vector<std::string> args;
                std::string token;
                while (iss >> token) {
                    args.push_back(token);
                }
                if (args.empty()) {
                    std::cout << "Usage: train <file> [epochs=1] [batch=32]\n";
                    continue;
                }
                if (args.size() > 3) {
                    std::cout << "Too many arguments for train command.\n";
                    continue;
                }

                std::string file = args.front();
                int epochs = 1;
                int batch = 32;

                auto parse_positive = [](const std::string& value, const char* label) -> std::optional<int> {
                    try {
                        int parsed = std::stoi(value);
                        if (parsed <= 0) {
                            std::cout << label << " must be positive.\n";
                            return std::nullopt;
                        }
                        return parsed;
                    }
                    catch (...) {
                        std::cout << "Invalid " << label << " value.\n";
                        return std::nullopt;
                    }
                };

                if (args.size() >= 2) {
                    if (auto parsed = parse_positive(args[1], "epochs")) {
                        epochs = *parsed;
                    }
                    else {
                        continue;
                    }
                }
                if (args.size() == 3) {
                    if (auto parsed = parse_positive(args[2], "batch")) {
                        batch = *parsed;
                    }
                    else {
                        continue;
                    }
                }

                JsonObject params;
                params["file"] = Json(file);
                params["epochs"] = Json(epochs);
                params["batch"] = Json(batch);

                if (!invoke_service_streaming("trainer.fit", Json(params), nullptr)) {
                    if (!invoke_service_streaming("train", Json(params), nullptr)) {
                        std::cout << "(no response)\n";
                    }
                }
                continue;
            }

            if (lowered_command == "hot-swap") {
                std::string name;
                std::getline(iss, name);
                trim(name);

                JsonObject params;
                if (!name.empty()) params["name"] = Json(name);

                auto response = invoke_service("admin.hot_swap", Json(params));
                if (!response) {
                    response = invoke_service("hot-swap", Json(params));
                }
                if (response) {
                    if (response->is_object()) {
                        const auto& obj = response->as_object();
                        if (auto out = obj.find("output"); out != obj.end() && out->second.is_string()) {
                            std::cout << out->second.as_string() << "\n";
                        }
                        else if (name.empty()) {
                            std::cout << "Rolled back to previous adapter.\n";
                        }
                        else {
                            std::cout << "Promoted adapter '" << name << "'.\n";
                        }
                    }
                    else if (name.empty()) {
                        std::cout << "Rolled back to previous adapter.\n";
                    }
                    else {
                        std::cout << "Promoted adapter '" << name << "'.\n";
                    }
                }
                else {
                    std::cout << "(no response)\n";
                }
                continue;
            }

            if (!slash_command) {
                handle_generation(original_line, true);
            }
            else {
                std::cout << "Unknown command. Type 'help' for a list of commands.\n";
            }
            continue;
        }
        catch (const std::exception& ex) {
            std::cout << "Error: " << ex.what() << "\n";
        }
    }

    return 0;
}



//#include "almondshell.hpp"
////#include "../src/aengine.cpp"
////#include "../src/almondshell"
//
//#include <chrono>
//#include <filesystem>
//#include <iostream>
//#include <thread>
//
////void test_linkage() {
////    almondshell::opengl::s_state.shader = 42;  // Or just read a member
////}
//
//// this basically just leaves ninja.zip when commented out, but will be configured better in the future
//#define LEAVE_NO_FILES_ALWAYS_REDOWNLOAD
//
////using namespace almondnamespace::core;
//
//// configuration overrides
//namespace urls {
//    const std::string github_base = "https://github.com/"; // base github url
//    const std::string github_raw_base = "https://raw.githubusercontent.com/"; // raw base github url, for source downloads
//
//    const std::string owner = "Autodidac/"; // github project developer username for url
//    const std::string repo = "Cpp_Ultimate_Project_Updater"; // whatever your github project name is
//    const std::string branch = "main/"; // incase you need a different branch than githubs default branch main
//
//    // It's now using this internal file to fetch update versions internally without version.txt file that can be modified
//    const std::string version_url = github_raw_base + owner + repo + "/" + branch + "/include/config.hpp";
//    //const std::string source_url = github_base + owner + repo + "/archive/refs/heads/main.zip";
//    const std::string binary_url = github_base + owner + repo + "/releases/latest/download/updater.exe";
//}
//
//static void RunEngine() {
//    // 🔄 **Cleanup Restart Script on Restart & Old Files on Update**
//#ifdef LEAVE_NO_FILES_ALWAYS_REDOWNLOAD
//#if defined(_WIN32)
//    system("del /F /Q replace_updater.bat >nul 2>&1");
//    system(("rmdir /s /q \"" + std::string(almondnamespace::updater::REPO.c_str()) + "-main\" >nul 2>&1").c_str());
//#else
//    system("rm -rf replace_updater");
//#endif
//#endif
//
//    if (almondnamespace::updater::check_for_updates(urls::version_url)) {
//        std::cout << "[INFO] New version available!\n";
//        almondnamespace::updater::update_project(urls::version_url, urls::binary_url);
//    }
//    else {
//        // Clear console before showing "No updates available."
//#if defined(_WIN32)
//        system("cls");
//#else
//        system("clear");
//#endif
//        std::cout << "[INFO] No updates available.\n";
//    }
//
//    // Lets Begin
//    //TaskScheduler scheduler;
//
//    std::cout << "[Engine] Starting up...\n";
//
//
//
//
//
//
//
//
//
//
//    //Unleash C++ Scripting!
//    /*
//    std::string scriptName = "editor_launcher";
//
//    if (!almond::load_or_reload_script(scriptName, scheduler)) {
//        std::cerr << "[Engine] Initial script load failed.\n";
//    }
//
//    auto lastCheck = std::filesystem::last_write_time("src/scripts/" + scriptName + ".ascript.cpp");
//    auto start = std::chrono::steady_clock::now();
//
//    while (true) {
//        auto now = std::filesystem::last_write_time("src/scripts/" + scriptName + ".ascript.cpp");
//        if (now != lastCheck) {
//            std::cout << "\n[Engine] Detected change in script source, recompiling...\n";
//            almond::load_or_reload_script(scriptName, scheduler);
//            lastCheck = now;
//        }
//
//        std::this_thread::sleep_for(std::chrono::milliseconds(200));
//        if (std::chrono::duration_cast<std::chrono::minutes>(std::chrono::steady_clock::now() - start).count() > 10) break;
//    }
//*/
//    std::cout << "[Engine] Session ended.\n";
//  //  return 0;
//}
