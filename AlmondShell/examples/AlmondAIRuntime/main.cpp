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
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

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

    BaseDecoder base(config);
    StudentModel student(std::move(base));

    AdapterManager adapter_manager;
    AdapterConfig adapter_config;
    adapter_manager.register_adapter(Adapter("default", config.hidden_size, adapter_config));
    adapter_manager.activate("default");

    PolicyGovernor governor;
    governor.set_blocklist({ "forbidden", "classified" });

    ContinuousLearner learner(std::move(student),
        std::move(adapter_manager),
        std::move(tokenizer),
        std::move(governor));
    learner.promote_adapter("default");

    MCPBridge bridge;
    Service service(learner, bridge);

    almondai::chat::BackendPtr chat_backend;
    std::optional<almondai::chat::Kind> chat_kind;

    auto getenv_string = [](const char* name) -> std::string {
#if defined(_WIN32)
        char* buf = nullptr;
        size_t len = 0;
        if (_dupenv_s(&buf, &len, name) == 0 && buf) {
            std::string value(buf);
            free(buf);
            return value;
        }
        return {};
#else
        const char* value = std::getenv(name);
        return value ? std::string(value) : std::string();
#endif
        };

    const std::string env_kind = getenv_string("ALMONDAI_CHAT_KIND");
    if (!env_kind.empty()) {
        const std::string env_endpoint = getenv_string("ALMONDAI_ENDPOINT");
        const std::string env_model = getenv_string("ALMONDAI_MODEL");
        const std::string env_key = getenv_string("ALMONDAI_API_KEY");
        try {
            almondai::chat::Kind parsed = almondai::chat::parse_kind(env_kind);
            chat_backend = almondai::chat::make_backend(parsed, env_endpoint, env_model, env_key);
            chat_kind = parsed;
            service.set_chat_backend(chat_backend.get(), almondai::chat::kind_to_string(parsed));
            std::cout << "Connected to " << almondai::chat::kind_to_string(parsed)
                      << " chat backend from environment configuration.\n";
        }
        catch (const std::exception& ex) {
            std::cout << "Failed to configure chat backend from environment: " << ex.what() << "\n";
        }
    }

    std::cout << "AlmondAI interactive console\n"
        "Type 'help' to list available commands or 'exit' to quit.\n"
        "Use 'chat use' to connect to an external backend or 'chat clear' to return to local inference.\n";

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

    auto invoke_service_streaming = [&service](const std::string& method, Json params) -> bool {
        JsonObject request;
        request["id"] = Json("cli");
        request["method"] = Json(method);
        request["params"] = std::move(params);

        std::istringstream input(Json(request).dump() + "\n");
        std::ostringstream output;
        service.run(input, output);

        const std::string buffer = output.str();
        if (buffer.empty()) {
            return false;
        }

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

        std::istringstream lines(buffer);
        std::string raw;
        std::size_t last_width = 0;
        bool saw_batch = false;
        bool saw_result = false;
        double final_loss = 0.0;
        int final_step = 0;
        std::string fallback_text;

        while (std::getline(lines, raw)) {
            if (!raw.empty() && raw.back() == '\r') raw.pop_back();
            if (raw.empty()) {
                continue;
            }
            Json parsed;
            try {
                parsed = Json::parse(raw);
            }
            catch (...) {
                if (!fallback_text.empty()) fallback_text.append("\n");
                fallback_text.append(raw);
                continue;
            }
            if (!parsed.is_object()) {
                if (!fallback_text.empty()) fallback_text.append("\n");
                fallback_text.append(raw);
                continue;
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
                if (event_it->second.as_string() == "batch") {
                    int step_value = 0;
                    double loss_value = 0.0;
                    if (auto step_it = obj.find("step"); step_it != obj.end()) {
                        step_value = parse_int(step_it->second, step_value);
                    }
                    if (auto loss_it = obj.find("loss"); loss_it != obj.end()) {
                        loss_value = parse_double(loss_it->second, loss_value);
                    }
                    final_step = step_value;
                    final_loss = loss_value;
                    std::ostringstream progress;
                    progress << "[train] step " << step_value
                             << " loss " << std::fixed << std::setprecision(4) << loss_value;
                    const std::string text = progress.str();
                    std::cout << '\r' << text;
                    if (last_width > text.size()) {
                        std::cout << std::string(last_width - text.size(), ' ');
                    }
                    std::cout << std::flush;
                    last_width = text.size();
                    saw_batch = true;
                    continue;
                }
            }
            if (auto result_it = obj.find("result"); result_it != obj.end() && result_it->second.is_object()) {
                const auto& result = result_it->second.as_object();
                if (auto loss_it = result.find("final_loss"); loss_it != result.end()) {
                    final_loss = parse_double(loss_it->second, final_loss);
                }
                if (auto steps_it = result.find("steps"); steps_it != result.end()) {
                    final_step = parse_int(steps_it->second, final_step);
                }
                saw_result = true;
            }
        }

        if (saw_result) {
            if (saw_batch) {
                std::cout << '\r' << std::string(last_width, ' ') << '\r';
            }
            std::ostringstream summary;
            summary << "Training complete (loss=" << std::fixed << std::setprecision(4) << final_loss
                    << ", steps=" << final_step << ")\n";
            std::cout << summary.str();
        }
        else if (saw_batch) {
            std::cout << '\r' << std::string(last_width, ' ') << '\r' << std::flush;
            return true;
        }

        if (!fallback_text.empty()) {
            std::cout << fallback_text << '\n';
            return true;
        }

        return saw_result || saw_batch;
    };

    std::string line;
    while (std::cout << "AlmondAI> " && std::getline(std::cin, line)) {
        trim(line);
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string command;
        iss >> command;

        if (command == "exit" || command == "quit") break;

        if (command == "help") {
            std::cout <<
                "Available commands:\n"
                "  generate <prompt>      Generate a completion (uses chat backend if configured).\n"
                "  retrieve <query>       Query the retrieval index.\n"
                "  train <file> [epochs] [batch]  Run iterative training over <file>.\n"
                "  hot-swap [name]        Promote adapter <name> or rollback if omitted.\n"
                "  chat use <kind> <endpoint> [model] [key]  Switch to an external chat backend.\n"
                "  chat clear             Return to local model responses.\n"
                "  exit                   Quit the console.\n";
            continue;
        }

        try {
            if (command == "chat") {
                std::string subcommand;
                iss >> subcommand;
                if (subcommand == "use") {
                    std::vector<std::string> args;
                    std::string token;
                    while (iss >> token) {
                        args.push_back(token);
                    }
                    if (args.size() < 2) {
                        std::cout << "Usage: chat use <kind> <endpoint> [model] [key]\n";
                        continue;
                    }
                    try {
                        almondai::chat::Kind kind = almondai::chat::parse_kind(args[0]);
                        std::string a = args[1];
                        std::string b = args.size() >= 3 ? args[2] : std::string();
                        std::string c = args.size() >= 4 ? args[3] : std::string();
                        chat_backend = almondai::chat::make_backend(kind, std::move(a), std::move(b), std::move(c));
                        chat_kind = kind;
                        service.set_chat_backend(chat_backend.get(), almondai::chat::kind_to_string(kind));
                        std::cout << "Using " << almondai::chat::kind_to_string(kind)
                                  << " chat backend.\n";
                    }
                    catch (const std::exception& ex) {
                        std::cout << "Failed to configure chat backend: " << ex.what() << "\n";
                    }
                    continue;
                }
                if (subcommand == "clear") {
                    if (chat_kind) {
                        std::cout << "Chat backend '" << almondai::chat::kind_to_string(*chat_kind)
                                  << "' cleared. Local model enabled.\n";
                    } else {
                        std::cout << "Chat backend cleared. Local model enabled.\n";
                    }
                    chat_backend.reset();
                    chat_kind.reset();
                    service.set_chat_backend(nullptr);
                    continue;
                }
                std::cout << "Usage: chat use <kind> <endpoint> [model] [key] or chat clear\n";
                continue;
            }

            if (command == "generate") {
                std::string prompt;
                std::getline(iss, prompt);
                trim(prompt);
                if (prompt.empty()) { std::cout << "Prompt cannot be empty.\n"; continue; }

                JsonObject params;
                params["prompt"] = Json(prompt);
                auto result = invoke_service("model.generate", Json(params));
                if (!result) {
                    result = invoke_service("generate", Json(params));
                }
                if (result && result->is_object()) {
                    const auto& obj = result->as_object();
                    if (auto it = obj.find("output"); it != obj.end() && it->second.is_string()) {
                        std::cout << it->second.as_string() << "\n";
                    }
                    else if (auto t = obj.find("text"); t != obj.end() && t->second.is_string()) {
                        std::cout << t->second.as_string() << "\n";
                    }
                    else {
                        std::cout << "No output returned.\n";
                    }
                    if (auto route_it = obj.find("route"); route_it != obj.end() && route_it->second.is_string()) {
                        const std::string route = route_it->second.as_string();
                        if (!route.empty()) {
                            std::cout << "[route: " << route;
                            if (route != "local") {
                                if (auto backend_it = obj.find("backend"); backend_it != obj.end() && backend_it->second.is_string()) {
                                    std::cout << ", backend: " << backend_it->second.as_string();
                                }
                            }
                            std::cout << "]\n";
                        }
                    }
                    if (auto err_it = obj.find("remote_error"); err_it != obj.end() && err_it->second.is_string()) {
                        std::cout << "(remote error: " << err_it->second.as_string() << ")\n";
                    }
                    if (auto fallback_it = obj.find("fallback"); fallback_it != obj.end() && fallback_it->second.is_object()) {
                        const auto& fb = fallback_it->second.as_object();
                        if (auto fb_msg = fb.find("output"); fb_msg != fb.end() && fb_msg->second.is_string()) {
                            std::cout << "[fallback notice: " << fb_msg->second.as_string() << "]\n";
                        }
                    }
                }
                else {
                    std::cout << "(no response)\n";
                }
                continue;
            }

            if (command == "retrieve") {
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

            if (command == "train") {
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

                if (!invoke_service_streaming("trainer.fit", Json(params))) {
                    if (!invoke_service_streaming("train", Json(params))) {
                        std::cout << "(no response)\n";
                    }
                }
                continue;
            }

            if (command == "hot-swap") {
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

            std::cout << "Unknown command. Type 'help' for a list of commands.\n";
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
