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

#include <filesystem>
#include <iostream>

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

    ContinuousLearner learner(std::move(student), std::move(adapter_manager), std::move(tokenizer), std::move(governor));
    learner.promote_adapter("default");

    MCPBridge bridge;
    Service service(learner, bridge);
    service.run(std::cin, std::cout);
    return 0;
}





#include "almondshell.hpp"
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
