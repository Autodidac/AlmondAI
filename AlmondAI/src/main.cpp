//// main.cpp - AlmondAI runtime console demonstration
//
//#include "../../../AlmondAI/include/almondai/serve.hpp"
//#include "../../../AlmondAI/include/almondai/adapter.hpp"
//#include "../../../AlmondAI/include/almondai/tokenizer_word.hpp"
//#include "../../../AlmondAI/include/almondai/json.hpp"
//
//#include <algorithm>
//#include <cctype>
//#include <charconv>
//#include <filesystem>
//#include <iomanip>
//#include <iostream>
//#include <limits>
//#include <optional>
//#include <sstream>
//#include <stdexcept>
//#include <string>
//#include <type_traits>
//#include <variant>
//#include <vector>
//
//int main() {
//    using namespace almondai;
//
//    WordTokenizer tokenizer;
//    const std::filesystem::path vocab_path = "data/vocab.txt";
//    if (std::filesystem::exists(vocab_path)) {
//        tokenizer.load_vocab(vocab_path.string());
//    }
//    else {
//        tokenizer.save_vocab(vocab_path.string());
//    }
//
//    ModelConfig config;
//    config.vocab_size = tokenizer.vocab().size();
//    config.hidden_size = 64;
//    config.num_layers = 2;
//
//    BaseDecoder base(config);
//    StudentModel student(std::move(base));
//
//    AdapterManager adapter_manager;
//    AdapterConfig adapter_config;
//    adapter_manager.register_adapter(Adapter("default", config.hidden_size, adapter_config));
//    adapter_manager.activate("default");
//
//    PolicyGovernor governor;
//    governor.set_blocklist({ "forbidden", "classified" });
//
//    ContinuousLearner learner(std::move(student),
//        std::move(adapter_manager),
//        std::move(tokenizer),
//        std::move(governor));
//    learner.promote_adapter("default");
//
//    MCPBridge bridge;
//    Service service(learner, bridge);
//
//    std::cout << "AlmondAI interactive console\n"
//        "Type 'help' to list available commands or 'exit' to quit.\n";
//
//    auto trim = [](std::string& text) {
//        auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
//        text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
//        text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
//        };
//
//    // Robust service bridge:
//    // 1) Try parsing the entire buffer as one JSON object.
//    // 2) If that fails, scan all non-empty lines and parse the last valid JSON.
//    auto invoke_service = [&service](const std::string& method, Json params) -> std::optional<Json> {
//        JsonObject request;
//        request["id"] = Json("cli");
//        request["method"] = Json(method); // If backend expects "generate"/"retrieve"/"hot-swap", adjust call sites below.
//        request["params"] = std::move(params);
//
//        std::istringstream input(Json(request).dump() + "\n");
//        std::ostringstream output;
//        service.run(input, output);
//
//        const std::string buf = output.str();
//        if (buf.empty()) {
//            return std::nullopt;
//        }
//
//        auto try_extract_result = [](const Json& response) -> std::optional<Json> {
//            const auto& obj = response.as_object();
//            if (auto err = obj.find("error"); err != obj.end()) {
//                const auto& err_obj = err->second.as_object();
//                if (auto it = err_obj.find("message"); it != err_obj.end() && it->second.is_string())
//                    throw std::runtime_error(it->second.as_string());
//                throw std::runtime_error("service returned an error");
//            }
//            if (auto res = obj.find("result"); res != obj.end()) {
//                return res->second;
//            }
//            return std::nullopt;
//            };
//
//        // Strategy A: whole buffer is one JSON
//        try {
//            Json whole = Json::parse(buf);
//            if (auto r = try_extract_result(whole)) return r;
//        }
//        catch (...) {
//            // fall through
//        }
//
//        // Strategy B: scan lines for last valid JSON with "result"
//        std::istringstream resp(buf);
//        std::string line;
//        std::optional<Json> last_result;
//        while (std::getline(resp, line)) {
//            if (!line.empty() && line.back() == '\r') line.pop_back();
//            if (line.empty()) continue;
//            try {
//                Json j = Json::parse(line);
//                if (auto r = try_extract_result(j)) last_result = std::move(r);
//            }
//            catch (...) {
//                // ignore non-JSON log lines
//            }
//        }
//        return last_result;
//        };
//
//    auto invoke_service_streaming = [&service](const std::string& method, Json params) -> bool {
//        JsonObject request;
//        request["id"] = Json("cli");
//        request["method"] = Json(method);
//        request["params"] = std::move(params);
//
//        std::istringstream input(Json(request).dump() + "\n");
//        std::ostringstream output;
//        service.run(input, output);
//
//        const std::string buffer = output.str();
//        if (buffer.empty()) {
//            return false;
//        }
//
//        auto parse_int = [](const Json& value, int fallback) -> int {
//            return std::visit([
//                fallback
//            ](const auto& raw) -> int {
//                    using T = std::decay_t<decltype(raw)>;
//                    if constexpr (std::is_same_v<T, double>) {
//                        return static_cast<int>(raw);
//                    }
//                    else if constexpr (std::is_same_v<T, bool>) {
//                        return raw ? 1 : 0;
//                    }
//                    else if constexpr (std::is_same_v<T, std::string>) {
//                        try {
//                            std::size_t idx = 0;
//                            int parsed = std::stoi(raw, &idx);
//                            if (idx == raw.size()) {
//                                return parsed;
//                            }
//                        }
//                        catch (...) {
//                        }
//                        return fallback;
//                    }
//                    else {
//                        return fallback;
//                    }
//                }, value.value());
//            };
//
//        auto parse_double = [](const Json& value, double fallback) -> double {
//            return std::visit([
//                fallback
//            ](const auto& raw) -> double {
//                    using T = std::decay_t<decltype(raw)>;
//                    if constexpr (std::is_same_v<T, double>) {
//                        return raw;
//                    }
//                    else if constexpr (std::is_same_v<T, bool>) {
//                        return raw ? 1.0 : 0.0;
//                    }
//                    else if constexpr (std::is_same_v<T, std::string>) {
//                        try {
//                            std::size_t idx = 0;
//                            double parsed = std::stod(raw, &idx);
//                            if (idx == raw.size()) {
//                                return parsed;
//                            }
//                        }
//                        catch (...) {
//                        }
//                        return fallback;
//                    }
//                    else {
//                        return fallback;
//                    }
//                }, value.value());
//            };
//
//        std::istringstream lines(buffer);
//        std::string raw;
//        std::size_t last_width = 0;
//        bool saw_batch = false;
//        bool saw_result = false;
//        double final_loss = 0.0;
//        int final_step = 0;
//
//        while (std::getline(lines, raw)) {
//            if (!raw.empty() && raw.back() == '\r') raw.pop_back();
//            if (raw.empty()) {
//                continue;
//            }
//            Json parsed;
//            try {
//                parsed = Json::parse(raw);
//            }
//            catch (...) {
//                continue;
//            }
//            if (!parsed.is_object()) {
//                continue;
//            }
//            const auto& obj = parsed.as_object();
//            if (auto err_it = obj.find("error"); err_it != obj.end()) {
//                if (err_it->second.is_object()) {
//                    const auto& err_obj = err_it->second.as_object();
//                    if (auto msg_it = err_obj.find("message"); msg_it != err_obj.end() && msg_it->second.is_string()) {
//                        throw std::runtime_error(msg_it->second.as_string());
//                    }
//                }
//                throw std::runtime_error("service returned an error");
//            }
//            if (auto event_it = obj.find("event"); event_it != obj.end() && event_it->second.is_string()) {
//                if (event_it->second.as_string() == "batch") {
//                    int step_value = 0;
//                    double loss_value = 0.0;
//                    if (auto step_it = obj.find("step"); step_it != obj.end()) {
//                        step_value = parse_int(step_it->second, step_value);
//                    }
//                    if (auto loss_it = obj.find("loss"); loss_it != obj.end()) {
//                        loss_value = parse_double(loss_it->second, loss_value);
//                    }
//                    final_step = step_value;
//                    final_loss = loss_value;
//                    std::ostringstream progress;
//                    progress << "[train] step " << step_value
//                        << " loss " << std::fixed << std::setprecision(4) << loss_value;
//                    const std::string text = progress.str();
//                    std::cout << '\r' << text;
//                    if (last_width > text.size()) {
//                        std::cout << std::string(last_width - text.size(), ' ');
//                    }
//                    std::cout << std::flush;
//                    last_width = text.size();
//                    saw_batch = true;
//                    continue;
//                }
//            }
//            if (auto result_it = obj.find("result"); result_it != obj.end() && result_it->second.is_object()) {
//                const auto& result = result_it->second.as_object();
//                if (auto loss_it = result.find("final_loss"); loss_it != result.end()) {
//                    final_loss = parse_double(loss_it->second, final_loss);
//                }
//                if (auto steps_it = result.find("steps"); steps_it != result.end()) {
//                    final_step = parse_int(steps_it->second, final_step);
//                }
//                saw_result = true;
//            }
//        }
//
//        if (saw_result) {
//            if (saw_batch) {
//                std::cout << '\r' << std::string(last_width, ' ') << '\r';
//            }
//            std::ostringstream summary;
//            summary << "Training complete (loss=" << std::fixed << std::setprecision(4) << final_loss
//                << ", steps=" << final_step << ")\n";
//            std::cout << summary.str();
//        }
//        else if (saw_batch) {
//            std::cout << '\r' << std::string(last_width, ' ') << '\r' << std::flush;
//        }
//
//        return saw_result || saw_batch;
//        };
//
//    std::string line;
//    while (std::cout << "AlmondAI> " && std::getline(std::cin, line)) {
//        trim(line);
//        if (line.empty()) continue;
//
//        std::istringstream iss(line);
//        std::string command;
//        iss >> command;
//
//        if (command == "exit" || command == "quit") break;
//
//        if (command == "help") {
//            std::cout <<
//                "Available commands:\n"
//                "  generate <prompt>      Generate a completion for the prompt.\n"
//                "  retrieve <query>       Query the retrieval index.\n"
//                "  train <file> [epochs] [batch]  Run iterative training over <file>.\n"
//                "  hot-swap [name]        Promote adapter <name> or rollback if omitted.\n"
//                "  exit                   Quit the console.\n";
//            continue;
//        }
//
//        try {
//            if (command == "generate") {
//                std::string prompt;
//                std::getline(iss, prompt);
//                trim(prompt);
//                if (prompt.empty()) { std::cout << "Prompt cannot be empty.\n"; continue; }
//
//                JsonObject params;
//                params["prompt"] = Json(prompt);
//
//                // If your Service expects plain "generate", change here:
//                // if (auto result = invoke_service("generate", Json(params))) { ... }
//                if (auto result = invoke_service("model.generate", Json(params))) {
//                    const auto& obj = result->as_object();
//                    if (auto it = obj.find("output"); it != obj.end() && it->second.is_string()) {
//                        std::cout << it->second.as_string() << "\n";
//                    }
//                    else if (auto t = obj.find("text"); t != obj.end() && t->second.is_string()) {
//                        std::cout << t->second.as_string() << "\n";
//                    }
//                    else {
//                        std::cout << "No output returned.\n";
//                    }
//                }
//                else {
//                    std::cout << "(no response)\n";
//                }
//                continue;
//            }
//
//            if (command == "retrieve") {
//                std::string query;
//                std::getline(iss, query);
//                trim(query);
//                if (query.empty()) { std::cout << "Query cannot be empty.\n"; continue; }
//
//                JsonObject params;
//                params["query"] = Json(query);
//
//                // If your Service expects plain "retrieve", change here:
//                // if (auto result = invoke_service("retrieve", Json(params))) { ... }
//                if (auto result = invoke_service("retrieval.query", Json(params))) {
//                    const auto& arr = result->as_array();
//                    if (arr.empty()) {
//                        std::cout << "No retrieval hits.\n";
//                    }
//                    else {
//                        for (const auto& item : arr) {
//                            const auto& obj = item.as_object();
//                            std::string id = obj.count("document_id") ? obj.at("document_id").as_string() : "<no-id>";
//
//                            double score = 0.0;
//                            if (auto it = obj.find("score"); it != obj.end()) {
//                                // Accept any arithmetic type or stringified number.
//                                const auto& v = it->second.value(); // variant inside your Json
//                                std::visit([&](const auto& x) {
//                                    using T = std::decay_t<decltype(x)>;
//                                    if constexpr (std::is_arithmetic_v<T>) {
//                                        score = static_cast<double>(x);
//                                    }
//                                    else if constexpr (std::is_same_v<T, std::string>) {
//                                        const char* begin = x.data();
//                                        const char* end = x.data() + x.size();
//                                        // try integer first (fast, no alloc)
//                                        long long ll = 0;
//                                        if (auto [p, ec] = std::from_chars(begin, end, ll); ec == std::errc{} && p == end) {
//                                            score = static_cast<double>(ll);
//                                        }
//                                        else {
//                                            // fallback: double parse
//                                            try { score = std::stod(x); }
//                                            catch (...) {}
//                                        }
//                                    }
//                                    // else: ignore non-numeric types
//                                    }, v);
//                            }
//
//                            std::cout << "- " << id << " (score: " << score << ")\n";
//                        }
//                    }
//                }
//                else {
//                    std::cout << "(no response)\n";
//                }
//                continue;
//            }
//
//            if (command == "train") {
//                std::vector<std::string> args;
//                std::string token;
//                while (iss >> token) {
//                    args.push_back(token);
//                }
//                if (args.empty()) {
//                    std::cout << "Usage: train <file> [epochs=1] [batch=32]\n";
//                    continue;
//                }
//                if (args.size() > 3) {
//                    std::cout << "Too many arguments for train command.\n";
//                    continue;
//                }
//
//                std::string file = args.front();
//                int epochs = 1;
//                int batch = 32;
//
//                auto parse_positive = [](const std::string& value, const char* label) -> std::optional<int> {
//                    try {
//                        int parsed = std::stoi(value);
//                        if (parsed <= 0) {
//                            std::cout << label << " must be positive.\n";
//                            return std::nullopt;
//                        }
//                        return parsed;
//                    }
//                    catch (...) {
//                        std::cout << "Invalid " << label << " value.\n";
//                        return std::nullopt;
//                    }
//                    };
//
//                if (args.size() >= 2) {
//                    if (auto parsed = parse_positive(args[1], "epochs")) {
//                        epochs = *parsed;
//                    }
//                    else {
//                        continue;
//                    }
//                }
//                if (args.size() == 3) {
//                    if (auto parsed = parse_positive(args[2], "batch")) {
//                        batch = *parsed;
//                    }
//                    else {
//                        continue;
//                    }
//                }
//
//                JsonObject params;
//                params["file"] = Json(file);
//                params["epochs"] = Json(epochs);
//                params["batch"] = Json(batch);
//
//                if (!invoke_service_streaming("trainer.fit", Json(params))) {
//                    std::cout << "(no response)\n";
//                }
//                continue;
//            }
//
//            if (command == "hot-swap") {
//                std::string name;
//                std::getline(iss, name);
//                trim(name);
//
//                JsonObject params;
//                if (!name.empty()) params["name"] = Json(name);
//
//                // If your Service expects "hot-swap", change here:
//                // if (invoke_service("hot-swap", Json(params))) { ... }
//                if (invoke_service("admin.hot_swap", Json(params))) {
//                    if (name.empty()) std::cout << "Rolled back to previous adapter.\n";
//                    else             std::cout << "Promoted adapter '" << name << "'.\n";
//                }
//                else {
//                    std::cout << "(no response)\n";
//                }
//                continue;
//            }
//
//            std::cout << "Unknown command. Type 'help' for a list of commands.\n";
//        }
//        catch (const std::exception& ex) {
//            std::cout << "Error: " << ex.what() << "\n";
//        }
//    }
//
//    return 0;
//}
//
