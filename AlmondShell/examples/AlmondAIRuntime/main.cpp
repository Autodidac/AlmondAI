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

#include <algorithm>
#include <cctype>
#include <charconv>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>

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

    std::cout << "AlmondAI interactive console\n"
        "Type 'help' to list available commands or 'exit' to quit.\n";

    auto trim = [](std::string& text) {
        auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
        text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
        text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
        };

    // Robust service bridge:
    // 1) Try parsing the entire buffer as one JSON object.
    // 2) If that fails, scan all non-empty lines and parse the last valid JSON.
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

        auto try_extract_result = [](const Json& response) -> std::optional<Json> {
            const auto& obj = response.as_object();
            if (auto err = obj.find("error"); err != obj.end()) {
                const auto& err_obj = err->second.as_object();
                if (auto it = err_obj.find("message"); it != err_obj.end() && it->second.is_string())
                    throw std::runtime_error(it->second.as_string());
                throw std::runtime_error("service returned an error");
            }
            if (auto res = obj.find("result"); res != obj.end()) {
                return res->second;
            }
            return std::nullopt;
            };

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
                // ignore non-JSON log lines
            }
        }
        return last_result;
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
                "  generate <prompt>      Generate a completion for the prompt.\n"
                "  retrieve <query>       Query the retrieval index.\n"
                "  hot-swap [name]        Promote adapter <name> or rollback if omitted.\n"
                "  exit                   Quit the console.\n";
            continue;
        }

        try {
            if (command == "generate") {
                std::string prompt;
                std::getline(iss, prompt);
                trim(prompt);
                if (prompt.empty()) { std::cout << "Prompt cannot be empty.\n"; continue; }

                JsonObject params;
                params["prompt"] = Json(prompt);

                // If your Service expects plain "generate", change here:
                // if (auto result = invoke_service("generate", Json(params))) { ... }
                if (auto result = invoke_service("model.generate", Json(params))) {
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

                // If your Service expects plain "retrieve", change here:
                // if (auto result = invoke_service("retrieve", Json(params))) { ... }
                if (auto result = invoke_service("retrieval.query", Json(params))) {
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
                                // Accept any arithmetic type or stringified number.
                                const auto& v = it->second.value(); // variant inside your Json
                                std::visit([&](const auto& x) {
                                    using T = std::decay_t<decltype(x)>;
                                    if constexpr (std::is_arithmetic_v<T>) {
                                        score = static_cast<double>(x);
                                    }
                                    else if constexpr (std::is_same_v<T, std::string>) {
                                        const char* begin = x.data();
                                        const char* end = x.data() + x.size();
                                        // try integer first (fast, no alloc)
                                        long long ll = 0;
                                        if (auto [p, ec] = std::from_chars(begin, end, ll); ec == std::errc{} && p == end) {
                                            score = static_cast<double>(ll);
                                        }
                                        else {
                                            // fallback: double parse
                                            try { score = std::stod(x); }
                                            catch (...) {}
                                        }
                                    }
                                    // else: ignore non-numeric types
                                    }, v);
                            }

                            std::cout << "- " << id << " (score: " << score << ")\n";
                        }
                    }
                }
                else {
                    std::cout << "(no response)\n";
                }
                continue;
            }

            if (command == "hot-swap") {
                std::string name;
                std::getline(iss, name);
                trim(name);

                JsonObject params;
                if (!name.empty()) params["name"] = Json(name);

                // If your Service expects "hot-swap", change here:
                // if (invoke_service("hot-swap", Json(params))) { ... }
                if (invoke_service("admin.hot_swap", Json(params))) {
                    if (name.empty()) std::cout << "Rolled back to previous adapter.\n";
                    else             std::cout << "Promoted adapter '" << name << "'.\n";
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
