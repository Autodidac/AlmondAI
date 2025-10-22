#include "../include/almondai/train.hpp"

#include <algorithm>
#include <atomic>
#include <iomanip>
#include <filesystem>
#include <sstream>
#include <functional>
#include <system_error>
#include <fstream>
#include <chrono>
#include <random>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <string_view>
#include <cstdlib>

namespace almondai {

namespace {
const std::filesystem::path kTrainingDataPath{"data/training_data.jsonl"};
const std::filesystem::path kSeedDataPath{"data/training_seed.jsonl"};
const std::filesystem::path kVocabPath{"data/vocab.txt"};
const std::filesystem::path kWeightsPath{"data/student_weights.json"};
const std::filesystem::path kSeedTextPath{"data/seed.txt"};
const std::filesystem::path kRetrievalMetadataPath{"data/retrieval_index.json"};

constexpr const char kDefaultSeedText[] =
    R"(AlmondAI is a self-evolving C++23 AI engine runtime that learns from its own source code, compiler feedback, and user interaction. It integrates AI directly into the software loop, enabling self-analysis, self-rebuilds, and continuous evolution across its modules.

Hello  <eos>
hello  <eos>
Hi  <eos>
Hey  <eos>
Hey there  <eos>
Hello there  <eos>
Greetings  <eos>
Howdy  <eos>
Nice to meet you  <eos>
Pleasure to meet you  <eos>
Good to see you  <eos>
Welcome back  <eos>
Long time no see  <eos>
How are you?  <eos>
I'm fine. How about you?  <eos>
Good morning  <eos>
Good afternoon  <eos>
Good evening  <eos>
Good night  <eos>
What's up  <eos>
What's up?  <eos>
Sup  <eos>
How's it going?  <eos>
What's new  <eos>
Not much  <eos>

Bonjour (French)  <eos>
Konnichiwa (Japanese)  <eos>
As-salamu alaykum (Arabic) -> Wa alaykum as-salam  <eos>
Hola (Spanish)  <eos>
Sawasdee (Thai)  <eos>

In Japan, people bow to show respect.  <eos>
In many Middle Eastern cultures, the left hand is not used for eating.  <eos>
In France, it's common to kiss on both cheeks when greeting friends.  <eos>

The human brain has about 86 billion neurons that communicate through signals.  <eos>
These neurons form networks responsible for thinking, memory, and movement.  <eos>

In quantum mechanics, particles can exist in a state of uncertainty until observed.  <eos>
This idea is often described using the example of Schrödinger's cat.  <eos>

The Renaissance was a time of cultural and scientific growth in Europe.  <eos>
Key figures include Leonardo da Vinci, Galileo Galilei, and Johannes Gutenberg.  <eos>

The Great Wall of China spans over 13,000 miles and was built to protect against invasions.  <eos>
It includes watchtowers, garrisons, and signal beacons.  <eos>

Free will is the idea that humans can make choices independent of external forces.  <eos>
Determinism argues that all events are caused by prior causes, making free will an illusion.  <eos>

In Japan, people use chopsticks for eating. Pointing them at others is considered rude.  <eos>
In many Middle Eastern cultures, using the right hand for eating is preferred.  <eos>

Time flows from past to future in physics.  <eos>
Einstein's theory of relativity shows that time can be affected by gravity and speed.  <eos>

If every part of a ship is replaced over time, is it still the same ship?  <eos>
A moving object can never reach its destination because it must cover half the distance first, then half of that, and so on infinitely.  <eos>

Should you steal medicine to save a dying loved one if no one else has access to it?  <eos>

The weather is nice today. I enjoy sunny days.  <eos>
I feel tired but I will keep going.  <eos>
Making mistakes is part of learning.  <eos>
Sometimes things go wrong, but we can fix them.  <eos>
Bad choices often lead to bad outcomes.  <eos>

Wars can last for many years.  <eos>
Dynasties often change over time.  <eos>
Japan has a rich history of cultural traditions.  <eos>
Formality is important in some cultures but less so in others.  <eos>
)";

constexpr const char kCompactSeedText[] =
    R"(AlmondAI speaks clear conversational English, keeps replies concise, and mirrors the user's intent. Respond with short paragraphs unless the user explicitly asks for lists or code. Always acknowledge the request, mention any assumptions, and close with an offer of further help.

Hello! I'm AlmondAI, your local assistant. <eos>
Thanks for reaching out. How can I help? <eos>
Certainly! Could you share a little more detail so I can assist precisely? <eos>
I can summarise documents, explain concepts, or walk through debugging steps. <eos>
)";

struct SeedSpec {
    const char* prompt;
    const char* teacher_output;
    const char* prompt_hash;
    const char* teacher_hash;
};

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

std::string determine_seed_profile() {
    const auto env = read_env("ALMONDAI_SEED_PROFILE");
    if (!env) {
        return "rich";
    }
    std::string profile(*env);
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    profile.erase(profile.begin(), std::find_if(profile.begin(), profile.end(), not_space));
    profile.erase(std::find_if(profile.rbegin(), profile.rend(), not_space).base(), profile.end());
    std::transform(profile.begin(), profile.end(), profile.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (profile == "compact" || profile == "minimal" || profile == "lmstudio") {
        return "compact";
    }
    return "rich";
}

std::string default_seed_text_for_profile(const std::string& profile) {
    if (profile == "compact") {
        return std::string{kCompactSeedText};
    }
    return std::string{kDefaultSeedText};
}

std::vector<SeedSpec> seed_file_curriculum_for_profile(const std::string& profile) {
    (void)profile;
    return {
        {"Greet politely in one line.", "Hello. Nice to meet you.", "seed::eng_greet_1", "1001"},
        {"Reply to: Hello", "Hi there.", "seed::eng_reply_hello", "1002"},
        {"Ask how someone is, short.", "How are you?", "seed::eng_ask_how", "1003"},
        {"Answer: How are you?", "I'm well, thanks. How about you?", "seed::eng_answer_how", "1004"},
        {"Morning greeting.", "Good morning.", "seed::eng_morning", "1005"},
        {"Make a polite request: open the door.", "Please open the door.", "seed::eng_request_open", "1006"},
        {"Fix grammar: 'She don't like it.'", "She doesn't like it.", "seed::eng_grammar_1", "1007"},
        {"Fix grammar: 'They was late.'", "They were late.", "seed::eng_grammar_2", "1008"},
        {"Choose the correct article: '___ apple'.", "an apple", "seed::eng_article_apple", "1009"},
        {"Explain when to use 'a' vs 'an' in one line.", "'A' before consonant sounds; 'an' before vowel sounds.", "seed::eng_article_rule", "1010"},
        {"Pluralize: 'mouse'.", "mice", "seed::eng_plural_mouse", "1011"},
        {"Past tense: 'go'.", "went", "seed::eng_past_go", "1012"},
        {"Paraphrase: 'The system failed to load.'", "The system did not load successfully.", "seed::eng_paraphrase_1", "1013"},
        {"Turn passive to active: 'The file was saved by the user.'", "The user saved the file.", "seed::eng_active_1", "1014"},
        {"Define a noun in one line.", "A noun names a person, place, thing, or idea.", "seed::eng_define_noun", "1015"},
        {"Define a verb in one line.", "A verb expresses an action or a state of being.", "seed::eng_define_verb", "1016"},
        {"Explain a comma in one line.", "A comma separates parts of a sentence to clarify meaning.", "seed::eng_punct_comma", "1017"},
        {"Explain a period in one line.", "A period ends a sentence.", "seed::eng_punct_period", "1018"},
        {"Write a short two-sentence self-intro.", "I'm a software assistant. I try to be clear and concise.", "seed::eng_self_intro", "1019"},
        {"Short encouragement.", "Keep going; you've got this.", "seed::eng_encourage", "1020"},
        {"C++: Print \"Hello, world!\" using iostream.", "#include <iostream>\nint main(){ std::cout << \"Hello, world!\\n\"; }", "seed::cpp_hello", "2001"},
        {"C++: Explain const correctness in one line.", "Const correctness prevents unintended mutation by marking data and member functions as read-only where possible.", "seed::cpp_const", "2002"},
        {"C++: Prefer pass-by-const-reference over pass-by-value for large objects - why?", "It avoids copies and preserves immutability for efficiency and safety.", "seed::cpp_constref", "2003"},
        {"C++: Show a function taking const std::string& and returning size_t.", "#include <string>\nsize_t length_of(const std::string& s){ return s.size(); }", "seed::cpp_func_sig", "2004"},
        {"C++: What is RAII in one sentence?", "RAII binds resource lifetime to object lifetime so acquisition and release happen in constructors and destructors.", "seed::cpp_raii", "2005"},
        {"C++: Show RAII with std::unique_ptr for an int.", "#include <memory>\nstd::unique_ptr<int> p = std::make_unique<int>(42);", "seed::cpp_unique_ptr_basic", "2006"},
        {"C++: unique_ptr vs shared_ptr in one line.", "unique_ptr has sole ownership; shared_ptr uses reference counting for shared ownership.", "seed::cpp_unique_vs_shared", "2007"},
        {"C++: Demonstrate move semantics with std::string.", "#include <string>\nstd::string a = \"data\"; std::string b = std::move(a);", "seed::cpp_move", "2008"},
        {"C++: Explain the Rule of Five in one line.", "If a type manages resources, define or delete copy/move ctor, copy/move assign, and destructor.", "seed::cpp_rule_of_five", "2009"},
        {"C++: Give a noexcept move constructor example signature.", "MyType(MyType&&) noexcept = default;", "seed::cpp_noexcept_move", "2010"},
        {"C++: Show range-based for over std::vector<int>.", "#include <vector>\nstd::vector<int> v{1,2,3}; for(int x : v){ /*...*/ }", "seed::cpp_range_for", "2011"},
        {"C++: Explain enum class vs enum in one line.", "enum class is scoped and doesn't implicitly convert to int; old enum is unscoped and implicitly convertible.", "seed::cpp_enum_class", "2012"},
        {"C++: Provide a constexpr function squaring an int.", "constexpr int sq(int x){ return x*x; }", "seed::cpp_constexpr_square", "2013"},
        {"C++: Capture by reference in a lambda summing a vector.", "#include <vector>\nint sum=0; std::vector<int> v{1,2,3}; auto f=[&]{ for(int x:v) sum+=x; }; f();", "seed::cpp_lambda_capture_ref", "2014"},
        {"C++: Why reserve() on std::vector improves performance?", "It preallocates capacity to reduce reallocations and copies during growth.", "seed::cpp_vector_reserve", "2015"},
        {"C++: Show std::optional<int> usage returning maybe a value.", "#include <optional>\nstd::optional<int> find(bool ok){ return ok? std::optional<int>{42} : std::nullopt; }", "seed::cpp_optional", "2016"},
        {"C++: Explain std::move in one line.", "std::move casts to an rvalue to enable moving; it does not move by itself.", "seed::cpp_std_move", "2017"},
        {"C++: Give a trivial header-only assert-like macro guarded by NDEBUG.", "#ifndef NDEBUG\n#define A_ASSERT(x) do{ if(!(x)) std::abort(); }while(0)\n#else\n#define A_ASSERT(x) do{}while(0)\n#endif", "seed::cpp_assert_macro", "2018"},
        {"C++: Thread-safe atomic increment with memory_order_relaxed.", "#include <atomic>\nstd::atomic<int> c{0}; c.fetch_add(1, std::memory_order_relaxed);", "seed::cpp_atomic_relaxed", "2019"},
        {"C++: Explain memory_order_relaxed in one line.", "It enforces atomicity but provides no ordering guarantees beyond that.", "seed::cpp_relaxed_line", "2020"},
        {"C++: Show a simple exception-safe file reader using ifstream.", "#include <fstream>\nstd::string read(const char* p){ std::ifstream f(p); return {std::istreambuf_iterator<char>(f), {}}; }", "seed::cpp_ifstream", "2021"},
        {"C++: Summarize exceptions vs error-codes in one line.", "Exceptions separate error flow from normal flow; error codes keep it explicit but verbose.", "seed::cpp_exc_vs_codes", "2022"},
        {"C++: Show a minimal header guard pattern.", "#pragma once", "seed::cpp_header_guard", "2023"},
        {"C++: One-line description of RAII for file handles.", "Wrap the handle in an object so the destructor reliably closes it.", "seed::cpp_raii_file", "2024"},
        {"Explain in one line: what is a sentence?", "A sentence is a complete thought that starts with a capital letter and ends with terminal punctuation.", "seed::eng_sentence_def", "1030"},
        {"Form a conditional in one sentence.", "If it rains, we will stay inside.", "seed::eng_conditional", "1031"},
        {"Give an example of a list with commas.", "We bought apples, oranges, and grapes.", "seed::eng_list_commas", "1032"},
        {"Correct capitalization: 'monday in april'.", "Monday in April", "seed::eng_caps", "1033"},
        {"Translate to plain English: 'utilize'.", "use", "seed::eng_plain_utilize", "1034"},
        {"Close a conversation politely.", "Thanks for your time. Goodbye.", "seed::eng_close", "1035"}
    };
}

std::vector<SeedSpec> bootstrap_curriculum_for_profile(const std::string& profile) {
    if (profile == "compact") {
        return {
            {"Offer a warm greeting to someone joining the conversation.",
             "Hello! It's great to hear from you. How can I support you today?",
             "seed::compact::greeting", nullptr},
            {"Respond to a user who asks for help debugging some code.",
             "I'd be happy to help debug the code. Tell me what language you're using, share the error message, and we can step through it together.",
             "seed::compact::debug", nullptr},
            {"Respond to a user who wants a summary of an article.",
             "Sure thing! Drop in the key points - or the article itself - and I'll return a concise English summary with any crucial details highlighted.",
             "seed::compact::summary", nullptr},
            {"Respond to a user who says 'Thank you!'.",
             "You're very welcome! If you need anything else, just let me know and we'll pick it up right away.",
             "seed::compact::thanks", nullptr},
            {"Respond to a user who says 'Goodbye'.",
             "Goodbye! It was great working with you - ping me anytime you want to continue.",
             "seed::compact::goodbye", nullptr},
            {"Show how AlmondAI formats a quick checklist in Markdown.",
             "Here you go!\n- [x] Acknowledge the request\n- [ ] Gather important context\n- [ ] Present a concise answer\n- [ ] Offer follow-up help",
             "seed::compact::checklist", nullptr}
        };
    }
    return {
        {"Offer a warm greeting to someone joining the conversation.",
         "Hello! It's great to hear from you. How can I support you today?",
         "seed::greeting::hello", nullptr},
        {"Respond to a user who says 'Hello there'.",
         "Hello there! It's a pleasure to connect - what would you like to dive into today?",
         "seed::greeting::hello_there", nullptr},
        {"Respond to a user who asks 'How are you?'.",
         "I'm doing great, thanks for asking! How can I help you today?",
         "seed::greeting::how_are_you", nullptr},
        {"Respond to a user who says 'Good morning'.",
         "Good morning! I hope your day is off to a bright and productive start.",
         "seed::greeting::good_morning", nullptr},
        {"Respond to a user who says 'Good afternoon'.",
         "Good afternoon! I hope everything's going smoothly - let me know what you'd like to tackle next.",
         "seed::greeting::good_afternoon", nullptr},
        {"Respond to a user who says 'Good evening'.",
         "Good evening! I hope the rest of your day treats you well - what should we look at next?",
         "seed::greeting::good_evening", nullptr},
        {"Respond to a user who says 'Good night'.",
         "Good night! Rest well, and ping me again whenever you're ready to continue.",
         "seed::greeting::good_night", nullptr},
        {"Respond to a user who says 'Hi'.",
         "Hi there! How can I assist you today?",
         "seed::greeting::hi", nullptr},
        {"Respond to a user who says 'Hiya'.",
         "Hiya! I'm ready when you are - what's on your mind?",
         "seed::greeting::hiya", nullptr},
        {"Respond to a user who says 'Hey'.",
         "Hey! Glad you're here. What can I do for you?",
         "seed::greeting::hey", nullptr},
        {"Respond to a user who says 'Yo'.",
         "Yo! Always happy to help - what are we working on today?",
         "seed::greeting::yo", nullptr},
        {"Respond to a user who asks 'How's it going?'.",
         "It's going great - thanks for checking in! What should we dive into next?",
         "seed::greeting::hows_it_going", nullptr},
        {"Respond to a user who says 'What's up?'.",
         "Not much - I'm right here and ready to help. What's on your agenda?",
         "seed::greeting::whats_up", nullptr},
        {"Respond to a user who says 'Sup?'.",
         "Sup! I'm dialed in and ready to jump into whatever you need.",
         "seed::greeting::sup", nullptr},
        {"Respond to a user who says 'Howdy'.",
         "Howdy! Always nice to hear from you - how can I lend a hand?",
         "seed::greeting::howdy", nullptr},
        {"Respond to a user who says 'Nice to meet you'.",
         "Nice to meet you too! Let me know what you're curious about and we'll explore it together.",
         "seed::greeting::nice_to_meet_you", nullptr},
        {"Respond to a user who says 'Pleasure to meet you'.",
         "The pleasure's mine! I'm here whenever you want to dig into something.",
         "seed::greeting::pleasure_to_meet_you", nullptr},
        {"Respond to a user who says 'Long time no see'.",
         "Long time no see! Let's pick up right where we left off.",
         "seed::greeting::long_time_no_see", nullptr},
        {"Respond to a user who says 'Welcome back'.",
         "Thanks! I'm all set to help - what should we get started on?",
         "seed::greeting::welcome_back", nullptr},
        {"Reply when someone thanks AlmondAI for the help.",
         "You're very welcome! I'm glad I could assist - let me know if there's anything else you need.",
         "seed::greeting::gratitude", nullptr},
        {"Close a conversation with a friendly farewell.",
         "Thanks for chatting with me. If you have more questions later, I'll be here. Take care!",
         "seed::greeting::farewell", nullptr},
        {"Respond to a user who says 'Goodbye'.",
         "Goodbye! It was great chatting - feel free to reach out again anytime you need a hand.",
         "seed::greeting::goodbye", nullptr}
    };
}

std::optional<CuratedSample> parse_sample_line(const std::string& line) {
    if (line.empty()) {
        return std::nullopt;
    }
    try {
        Json record = Json::parse(line);
        if (!record.is_object()) {
            return std::nullopt;
        }
        const auto& obj = record.as_object();
        auto prompt_it = obj.find("prompt");
        auto output_it = obj.find("teacher_output");
        if (prompt_it == obj.end() || output_it == obj.end()) {
            return std::nullopt;
        }
        if (!prompt_it->second.is_string() || !output_it->second.is_string()) {
            return std::nullopt;
        }
        CuratedSample sample;
        sample.prompt = prompt_it->second.as_string();
        sample.teacher_output = output_it->second.as_string();
        if (auto constraints_it = obj.find("constraints"); constraints_it != obj.end()) {
            sample.constraints = constraints_it->second;
        }
        if (auto provenance_it = obj.find("provenance"); provenance_it != obj.end()) {
            sample.provenance = provenance_it->second;
        } else {
            sample.provenance = JsonObject{};
        }
        if (auto tags_it = obj.find("semantic_tags"); tags_it != obj.end() && tags_it->second.is_array()) {
            const auto& tags_array = tags_it->second.as_array();
            sample.semantic_tags.reserve(tags_array.size());
            for (const auto& entry : tags_array) {
                if (entry.is_string()) {
                    sample.semantic_tags.push_back(entry.as_string());
                }
            }
        }
        return sample;
    } catch (...) {
        return std::nullopt;
    }
}

std::mt19937 make_training_rng() {
    static std::atomic<std::uint64_t> counter{0};
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::uint64_t seed = static_cast<std::uint64_t>(now)
        ^ (counter.fetch_add(1, std::memory_order_relaxed) + 0x9e3779b97f4a7c15ULL);
    std::seed_seq seq{
        static_cast<std::seed_seq::result_type>(seed & 0xffffffffu),
        static_cast<std::seed_seq::result_type>((seed >> 32) & 0xffffffffu)
    };
    return std::mt19937(seq);
}

std::string ensure_seed_text() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(kSeedTextPath.parent_path(), ec);

    const std::string profile = determine_seed_profile();
    const std::string desired_text = default_seed_text_for_profile(profile);

    std::string existing_text;
    if (fs::exists(kSeedTextPath, ec) && !ec) {
        std::ifstream in(kSeedTextPath);
        if (in) {
            std::ostringstream buffer;
            buffer << in.rdbuf();
            existing_text = buffer.str();
        }
    }

    auto should_overwrite = [&]() {
        if (existing_text.empty()) {
            return true;
        }
        if (existing_text == desired_text) {
            return false;
        }
        if (existing_text == std::string{kDefaultSeedText} || existing_text == std::string{kCompactSeedText}) {
            return true;
        }
        return false;
    }();

    if (should_overwrite) {
        std::ofstream out(kSeedTextPath, std::ios::trunc);
        if (!out) {
            return existing_text;
        }
        out << desired_text;
        existing_text = desired_text;
    }

    if (existing_text.empty()) {
        std::ifstream in(kSeedTextPath);
        if (!in) {
            return std::string();
        }
        std::ostringstream buffer;
        buffer << in.rdbuf();
        existing_text = buffer.str();
    }

    return existing_text;
}

void ensure_seed_samples() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(kSeedDataPath.parent_path(), ec);

    bool need_default = true;
    if (fs::exists(kSeedDataPath, ec) && !ec) {
        const auto size = fs::file_size(kSeedDataPath, ec);
        if (!ec && size > 0) {
            need_default = false;
        }
    }

    if (!need_default) {
        return;
    }

    (void)ensure_seed_text();

    std::ofstream out(kSeedDataPath, std::ios::trunc);
    if (!out) {
        return;
    }

    auto emit_sample = [&out](const SeedSpec& spec) {
        const std::string prompt = spec.prompt ? std::string{spec.prompt} : std::string{};
        const std::string teacher_output = spec.teacher_output ? std::string{spec.teacher_output} : std::string{};
        const std::string prompt_hash = spec.prompt_hash ? std::string{spec.prompt_hash} : std::string{};

        JsonObject provenance;
        provenance["source"] = Json("seed");
        provenance["prompt_hash"] = Json(prompt_hash);
        if (spec.teacher_hash && spec.teacher_hash[0] != '\0') {
            provenance["teacher_hash"] = Json(spec.teacher_hash);
        } else {
            provenance["teacher_hash"] = Json(std::to_string(std::hash<std::string>{}(teacher_output)));
        }

        JsonObject sample;
        sample["prompt"] = Json(prompt);
        sample["teacher_output"] = Json(teacher_output);
        sample["constraints"] = Json(JsonObject{});
        sample["provenance"] = Json(provenance);

        out << Json(sample).dump() << '\n';
    };

    const std::string profile = determine_seed_profile();
    const auto file_seeds = seed_file_curriculum_for_profile(profile);
    if (file_seeds.empty()) {
        return;
    }

    for (const auto& spec : file_seeds) {
        if (std::string(spec.prompt).empty() || std::string(spec.teacher_output).empty()) {
            continue;
        }
        emit_sample(spec);
    }

}

std::string normalise_tag_value(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

void append_tag(std::vector<std::string>& tags, std::unordered_set<std::string>& seen, std::string tag) {
    if (tag.empty()) {
        return;
    }
    if (seen.insert(tag).second) {
        tags.push_back(std::move(tag));
    }
}

std::vector<std::string> compute_semantic_tags(const CuratedSample& sample) {
    std::vector<std::string> tags;
    std::unordered_set<std::string> seen;
    tags.reserve(sample.semantic_tags.size() + 6);
    for (const auto& existing : sample.semantic_tags) {
        const std::string normalised = normalise_tag_value(existing);
        append_tag(tags, seen, normalised);
    }

    auto add_prefixed = [&](std::string prefix, const std::string& raw) {
        std::string normalised = normalise_tag_value(raw);
        if (normalised.empty()) {
            return;
        }
        append_tag(tags, seen, prefix + ':' + normalised);
        const std::size_t delim = normalised.find("::");
        if (delim != std::string::npos && delim > 0) {
            append_tag(tags, seen, prefix + ':' + normalised.substr(0, delim));
        }
    };

    if (sample.provenance.is_object()) {
        const auto& prov = sample.provenance.as_object();
        if (auto it = prov.find("source"); it != prov.end() && it->second.is_string()) {
            const std::string value = it->second.as_string();
            add_prefixed("source", value);
            if (normalise_tag_value(value) == "seed") {
                append_tag(tags, seen, "curriculum:seed");
            }
        }
        if (auto it = prov.find("status"); it != prov.end() && it->second.is_string()) {
            add_prefixed("status", it->second.as_string());
        }
        if (auto it = prov.find("teacher_source"); it != prov.end() && it->second.is_string()) {
            add_prefixed("teacher", it->second.as_string());
        }
        else if (auto it = prov.find("backend"); it != prov.end() && it->second.is_string()) {
            add_prefixed("teacher", it->second.as_string());
        }
        if (auto it = prov.find("backend"); it != prov.end() && it->second.is_string()) {
            add_prefixed("backend", it->second.as_string());
        }
    }

    if (!sample.prompt.empty()) {
        std::string trimmed = sample.prompt;
        trimmed.erase(trimmed.begin(), std::find_if_not(trimmed.begin(), trimmed.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }));
        if (!trimmed.empty()) {
            std::string first_word;
            for (char ch : trimmed) {
                if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
                    break;
                }
                first_word.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
            }
            if (!first_word.empty()) {
                append_tag(tags, seen, "prompt:" + first_word);
            }
        }
    }

    return tags;
}

std::vector<std::string> merge_semantic_tags(std::vector<std::string> base,
                                             const std::vector<std::string>& existing) {
    std::unordered_set<std::string> seen(base.begin(), base.end());
    for (const auto& tag : existing) {
        std::string normalised = normalise_tag_value(tag);
        if (normalised.empty()) {
            continue;
        }
        if (seen.insert(normalised).second) {
            base.push_back(normalised);
        }
    }
    return base;
}
}

ContinuousLearner::ContinuousLearner(StudentModel student,
                                     AdapterManager adapters,
                                     WordTokenizer tokenizer,
                                     PolicyGovernor governor,
                                     LoadStatusCallback load_callback)
    : m_student(std::move(student)),
      m_adapters(std::move(adapters)),
      m_tokenizer(std::move(tokenizer)),
      m_retrieval(m_tokenizer),
      m_evaluator(m_tokenizer),
      m_governor(std::move(governor)),
      m_load_status_callback(std::move(load_callback)) {
    m_log_file.open("data/training_log.txt", std::ios::app);
    if (m_log_file.tellp() == 0) {
        m_log_file << "AlmondAI training log\n";
        m_log_file << "Each entry below records a single training or evaluation step in a human-readable format.\n\n";
    }
    load_persistent_data();
}

std::optional<CuratedSample> ContinuousLearner::ingest(const std::string& prompt,
                                                       const std::string& teacher_output,
                                                       Json constraints,
                                                       const std::string& prompt_hash,
                                                       const std::string& teacher_source) {
    auto curated = m_curator.curate(prompt, teacher_output, std::move(constraints), prompt_hash, teacher_source);
    if (!curated) {
        return std::nullopt;
    }
    curated->semantic_tags = compute_semantic_tags(*curated);
    const std::size_t before_vocab = m_tokenizer.vocab().size();
    m_tokenizer.build_vocab({curated->prompt, curated->teacher_output});
    const std::size_t after_vocab = m_tokenizer.vocab().size();
    if (after_vocab > before_vocab) {
        m_student.base().resize_vocab(after_vocab);
        m_tokenizer.save_vocab(kVocabPath.string());
    }
    m_training_data.push_back(*curated);
    const std::size_t index = m_training_data.size() - 1;
    if (m_eval_data.size() < 16) {
        m_eval_data.push_back(*curated);
    }
    const std::string document_id = derive_document_id(m_training_data.back(), index);
    std::vector<std::string> merged_tags;
    if (!document_id.empty()) {
        merged_tags = merge_semantic_tags(m_training_data.back().semantic_tags, m_retrieval.tags_for(document_id));
    } else {
        merged_tags = merge_semantic_tags(m_training_data.back().semantic_tags, m_retrieval.tags_for(prompt_hash));
    }
    m_training_data.back().semantic_tags = merged_tags;
    curated->semantic_tags = merged_tags;
    if (!document_id.empty()) {
        if (m_training_data.back().provenance.is_object()) {
            auto& prov = m_training_data.back().provenance.as_object();
            if (prov.find("sample_hash") == prov.end()) {
                prov["sample_hash"] = Json(document_id);
            }
        }
        std::string retrieval_text = curated->prompt;
        if (!retrieval_text.empty() && !teacher_output.empty()) {
            retrieval_text.append("\n\n");
        }
        retrieval_text.append(teacher_output);
        m_retrieval.ingest_document(document_id, retrieval_text, m_training_data.back().semantic_tags);
        m_document_to_index[document_id] = index;
    } else {
        std::string retrieval_text = curated->prompt;
        if (!retrieval_text.empty() && !teacher_output.empty()) {
            retrieval_text.append("\n\n");
        }
        retrieval_text.append(teacher_output);
        m_retrieval.ingest_document(prompt_hash, retrieval_text, m_training_data.back().semantic_tags);
        m_document_to_index[prompt_hash] = index;
    }
    m_retrieval.save_metadata(kRetrievalMetadataPath);
    persist_sample(m_training_data.back());
    if (m_log_file.is_open()) {
        const std::size_t prompt_tokens = m_tokenizer.encode(curated->prompt).size();
        const std::size_t teacher_tokens = m_tokenizer.encode(teacher_output).size();
        m_log_file << "[learn::ingest] prompt_hash=" << (prompt_hash.empty() ? std::string{"unknown"} : prompt_hash)
                   << " | teacher_source=" << (teacher_source.empty() ? std::string{"unspecified"} : teacher_source)
                   << " | prompt_tokens=" << prompt_tokens
                   << " | teacher_tokens=" << teacher_tokens
                   << '\n';
        m_log_file.flush();
    }
    return m_training_data.back();
}

TrainingStats ContinuousLearner::train_step(const CuratedSample& sample) {
    ++m_step;
    TrainingStats stats;
    stats.step = m_step;
    stats.learning_tags.emplace_back("learn::step.begin");
    JsonObject begin_event;
    begin_event["tag"] = Json("learn::step.begin");
    begin_event["step"] = Json(static_cast<double>(stats.step));
    if (sample.provenance.is_object()) {
        const auto& prov = sample.provenance.as_object();
        if (auto it = prov.find("prompt_hash"); it != prov.end()) {
            begin_event["prompt_hash"] = it->second;
        }
        if (auto it = prov.find("sample_hash"); it != prov.end()) {
            begin_event["sample_hash"] = it->second;
        }
        if (auto it = prov.find("source"); it != prov.end()) {
            begin_event["teacher_source"] = it->second;
        }
    }
    stats.learning_trace.emplace_back(Json(begin_event));

    auto tokens = m_tokenizer.encode(sample.prompt);
    stats.learning_tags.emplace_back("learn::tokenize.prompt");
    JsonObject prompt_event;
    prompt_event["tag"] = Json("learn::tokenize.prompt");
    prompt_event["tokens"] = Json(static_cast<double>(tokens.size()));
    prompt_event["characters"] = Json(static_cast<double>(sample.prompt.size()));
    prompt_event["vocab_size"] = Json(static_cast<double>(m_tokenizer.vocab().size()));
    stats.learning_trace.emplace_back(Json(prompt_event));
    auto forward = m_student.forward(tokens);
    const auto& logits = forward.logits;
    const auto& hidden = forward.hidden;
    const auto& pre_adapter_hidden = forward.pre_adapter_hidden;
    stats.learning_tags.emplace_back("learn::forward.pass");
    JsonObject forward_event;
    forward_event["tag"] = Json("learn::forward.pass");
    forward_event["logit_count"] = Json(static_cast<double>(logits.size()));
    forward_event["hidden_width"] = Json(static_cast<double>(hidden.size()));
    stats.learning_trace.emplace_back(Json(forward_event));

    auto teacher_tokens = m_tokenizer.encode(sample.teacher_output);
    stats.learning_tags.emplace_back("learn::tokenize.teacher");
    JsonObject teacher_event;
    teacher_event["tag"] = Json("learn::tokenize.teacher");
    teacher_event["tokens"] = Json(static_cast<double>(teacher_tokens.size()));
    teacher_event["characters"] = Json(static_cast<double>(sample.teacher_output.size()));
    stats.learning_trace.emplace_back(Json(teacher_event));
    std::unordered_map<int, double> token_counts;
    for (int token : teacher_tokens) {
        if (token < 0) {
            continue;
        }
        const std::size_t index = static_cast<std::size_t>(token);
        if (index >= logits.size()) {
            continue;
        }
        token_counts[token] += 1.0;
    }
    if (token_counts.empty() && !logits.empty()) {
        token_counts[0] = 1.0;
    }

    const double total = std::accumulate(token_counts.begin(), token_counts.end(), 0.0,
                                         [](double sum, const auto& entry) {
                                             return sum + entry.second;
                                         });
    std::vector<double> target_distribution(logits.size(), 0.0);
    for (const auto& [token, count] : token_counts) {
        const std::size_t index = static_cast<std::size_t>(token);
        target_distribution[index] = count / (total > 0.0 ? total : 1.0);
    }

    std::vector<double> probabilities(logits.size(), 0.0);
    double normaliser = 0.0;
    double max_logit = logits.empty() ? 0.0 : *std::max_element(logits.begin(), logits.end());
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const double value = std::exp(logits[i] - max_logit);
        probabilities[i] = value;
        normaliser += value;
    }
    if (normaliser > 0.0) {
        for (double& probability : probabilities) {
            probability /= normaliser;
        }
    } else if (!probabilities.empty()) {
        const double uniform = 1.0 / static_cast<double>(probabilities.size());
        std::fill(probabilities.begin(), probabilities.end(), uniform);
    }

    constexpr double kEpsilon = 1e-12;
    std::vector<double> grad_logits(logits.size(), 0.0);
    double loss = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        grad_logits[i] = probabilities[i] - target_distribution[i];
        if (target_distribution[i] > 0.0) {
            loss -= target_distribution[i] * std::log(std::max(probabilities[i], kEpsilon));
        }
    }

    std::vector<double> grad_hidden(hidden.size(), 0.0);
    if (!grad_logits.empty()) {
        grad_hidden = m_student.update(hidden, grad_logits);
        stats.learning_tags.emplace_back("learn::update.student");
        JsonObject update_event;
        update_event["tag"] = Json("learn::update.student");
        update_event["gradient_dimensions"] = Json(static_cast<double>(grad_logits.size()));
        update_event["hidden_dimensions"] = Json(static_cast<double>(hidden.size()));
        stats.learning_trace.emplace_back(Json(update_event));
    }
    if (Adapter* active = m_adapters.active_adapter()) {
        active->apply_gradient(pre_adapter_hidden, grad_hidden);
        active->update_statistics(pre_adapter_hidden);
        stats.adapter_norm = active->norm();
        stats.learning_tags.emplace_back("learn::update.adapter");
        JsonObject adapter_event;
        adapter_event["tag"] = Json("learn::update.adapter");
        adapter_event["adapter_norm"] = Json(stats.adapter_norm);
        adapter_event["adapter_name"] = Json(active->name());
        stats.learning_trace.emplace_back(Json(adapter_event));
    }

    auto max_it = std::max_element(probabilities.begin(), probabilities.end());
    std::vector<int> decoded;
    if (max_it != probabilities.end()) {
        decoded.push_back(static_cast<int>(std::distance(probabilities.begin(), max_it)));
    }
    std::string student_output = m_tokenizer.decode(decoded);
    m_curator.record_student_response(sample.prompt, student_output, sample);

    std::unordered_set<int> teacher_vocab;
    for (const auto& [token, _] : token_counts) {
        (void)_;
        teacher_vocab.insert(token);
    }

    stats.loss = loss;
    if (max_it != probabilities.end()) {
        const int prediction = static_cast<int>(std::distance(probabilities.begin(), max_it));
        stats.accuracy = teacher_vocab.count(prediction) ? 1.0 : 0.0;
    } else {
        stats.accuracy = 0.0;
    }
    stats.retrieval_hit_rate = m_retrieval.hit_rate();
    if (sample.provenance.is_object()) {
        const auto& prov = sample.provenance.as_object();
        if (auto src_it = prov.find("source"); src_it != prov.end() && src_it->second.is_string()) {
            stats.teacher_source = src_it->second.as_string();
        }
    }
    stats.learning_tags.emplace_back("learn::summary");
    JsonObject summary_event;
    summary_event["tag"] = Json("learn::summary");
    summary_event["loss"] = Json(stats.loss);
    summary_event["accuracy"] = Json(stats.accuracy);
    summary_event["retrieval_hit_rate"] = Json(stats.retrieval_hit_rate);
    if (!stats.teacher_source.empty()) {
        summary_event["teacher_source"] = Json(stats.teacher_source);
    }
    stats.learning_trace.emplace_back(Json(summary_event));
    log_stats(stats);
    m_student.base().save_weights(kWeightsPath.string());

    return stats;
}

TrainingStats ContinuousLearner::evaluate_canary() {
    TrainingStats stats;
    if (m_eval_data.empty()) {
        return stats;
    }
    const auto metrics = m_evaluator.evaluate(m_student, m_eval_data);
    stats.learning_tags.emplace_back("learn::evaluate.canary");
    JsonObject evaluate_event;
    evaluate_event["tag"] = Json("learn::evaluate.canary");
    evaluate_event["samples_evaluated"] = Json(static_cast<double>(m_eval_data.size()));
    stats.learning_trace.emplace_back(Json(evaluate_event));
    stats.step = m_step;
    stats.loss = metrics.loss;
    stats.accuracy = metrics.accuracy;
    stats.retrieval_hit_rate = m_retrieval.hit_rate();
    if (const Adapter* adapter = m_adapters.active_adapter()) {
        stats.adapter_norm = adapter->norm();
    }
    stats.teacher_source = "evaluation";
    stats.learning_tags.emplace_back("learn::summary");
    JsonObject summary_event;
    summary_event["tag"] = Json("learn::summary");
    summary_event["loss"] = Json(stats.loss);
    summary_event["accuracy"] = Json(stats.accuracy);
    summary_event["retrieval_hit_rate"] = Json(stats.retrieval_hit_rate);
    summary_event["teacher_source"] = Json(stats.teacher_source);
    stats.learning_trace.emplace_back(Json(summary_event));
    log_stats(stats);
    return stats;
}

void ContinuousLearner::fit(const std::string& path,
                            int epochs,
                            int batch,
                            std::function<void(int, double, double, double)> on_batch) {
    const int safe_epochs = std::max(1, epochs);
    const int safe_batch = std::max(1, batch);

    std::vector<CuratedSample> dataset = m_training_data;

    if (!path.empty()) {
        std::ifstream file(path);
        if (file) {
            std::string line;
            while (std::getline(file, line)) {
                if (auto sample = parse_sample_line(line)) {
                    const std::size_t before_vocab = m_tokenizer.vocab().size();
                    m_tokenizer.build_vocab({sample->prompt, sample->teacher_output});
                    if (m_tokenizer.vocab().size() > before_vocab) {
                        m_student.base().resize_vocab(m_tokenizer.vocab().size());
                        m_tokenizer.save_vocab(kVocabPath.string());
                    }
                    dataset.push_back(*sample);
                }
            }
        }
    }

    if (dataset.empty()) {
        return;
    }

    std::mt19937 rng = make_training_rng();
    const int steps_per_epoch = std::max(1, static_cast<int>((dataset.size() + safe_batch - 1) / safe_batch));
    const double base_lr = m_student.base().config().learning_rate;

    int global_step = 0;
    for (int epoch = 0; epoch < safe_epochs; ++epoch) {
        std::shuffle(dataset.begin(), dataset.end(), rng);
        for (std::size_t offset = 0; offset < dataset.size(); offset += static_cast<std::size_t>(safe_batch)) {
            const std::size_t end = std::min(dataset.size(), offset + static_cast<std::size_t>(safe_batch));
            if (end <= offset) {
                continue;
            }
            const auto batch_start_time = std::chrono::steady_clock::now();
            double loss_sum = 0.0;
            std::size_t token_count = 0;
            for (std::size_t i = offset; i < end; ++i) {
                token_count += m_tokenizer.encode(dataset[i].prompt).size();
                TrainingStats stats = train_step(dataset[i]);
                loss_sum += stats.loss;
            }
            const auto batch_end_time = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed = batch_end_time - batch_start_time;
            const double tokens_per_second = elapsed.count() > 0.0
                ? static_cast<double>(token_count) / elapsed.count()
                : 0.0;

            ++global_step;
            if (on_batch) {
                const double average_loss = loss_sum / static_cast<double>(end - offset);
                const double schedule = 0.5 + 0.5 * (1.0 - (static_cast<double>(global_step - 1) / static_cast<double>(safe_epochs * steps_per_epoch)));
                const double current_lr = base_lr * schedule;
                on_batch(global_step, average_loss, current_lr, tokens_per_second);
            }
        }
    }
}

void ContinuousLearner::promote_adapter(const std::string& name) {
    m_adapters.activate(name);
    if (const Adapter* adapter = m_adapters.active_adapter()) {
        m_student.base().attach_adapter(adapter);
    }
}

void ContinuousLearner::rollback_adapter() {
    m_adapters.deactivate();
    m_student.base().attach_adapter(nullptr);
}

void ContinuousLearner::log_stats(const TrainingStats& stats) {
    if (!m_log_file.is_open()) {
        return;
    }
    m_log_file << "Step " << stats.step
               << " | loss=" << std::fixed << std::setprecision(6) << stats.loss
               << " | accuracy=" << stats.accuracy
               << " | adapter_norm=" << stats.adapter_norm
               << " | retrieval_hit_rate=" << stats.retrieval_hit_rate
               << " | teacher_source=" << (stats.teacher_source.empty() ? std::string{"unknown"} : stats.teacher_source);
    if (!stats.learning_tags.empty()) {
        m_log_file << " | tags=[";
        for (std::size_t i = 0; i < stats.learning_tags.size(); ++i) {
            if (i != 0) {
                m_log_file << ' ';
            }
            m_log_file << stats.learning_tags[i];
        }
        m_log_file << ']';
    }
    if (!stats.learning_trace.empty()) {
        Json trace_json(stats.learning_trace);
        m_log_file << " | trace=" << trace_json.dump();
    }
    m_log_file << '\n';
    m_log_file.flush();
}

void ContinuousLearner::load_persistent_data() {
    namespace fs = std::filesystem;
    std::error_code ec;

    report_load_status("initializing", "Ensuring data directories exist");
    fs::create_directories(kTrainingDataPath.parent_path(), ec);

    report_load_status("seeds", "Verifying seed curriculum");
    ensure_seed_samples();

    m_retrieval.load_metadata(kRetrievalMetadataPath);

    if (fs::exists(kWeightsPath)) {
        report_load_status("weights", "Loading student weights");
        const bool loaded = m_student.base().load_weights(kWeightsPath.string());
        report_load_status("weights", loaded ? "Student weights loaded" : "Failed to load student weights");
    } else {
        report_load_status("weights", "No persisted student weights found");
    }

    if (!fs::exists(kTrainingDataPath) && fs::exists(kSeedDataPath)) {
        report_load_status("seeds", "Initialising training data from seed set");
        fs::copy_file(kSeedDataPath, kTrainingDataPath, fs::copy_options::overwrite_existing, ec);
    }

    auto count_samples = [](const fs::path& path) -> std::size_t {
        if (!fs::exists(path)) {
            return 0;
        }
        std::ifstream in(path);
        if (!in) {
            return 0;
        }
        std::size_t count = 0;
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty()) {
                ++count;
            }
        }
        return count;
    };

    const std::size_t total_samples = count_samples(kTrainingDataPath);
    if (total_samples > 0) {
        report_load_status("samples", "Loading persisted training samples", 0, total_samples);
    } else {
        report_load_status("samples", "No persisted samples found", 0, 0);
    }

    load_samples_from_file(kTrainingDataPath, total_samples);

    if (!m_training_data.empty()) {
        std::ostringstream detail;
        detail << "Loaded " << m_training_data.size() << " samples from disk";
        report_load_status("samples", detail.str(), m_training_data.size(),
                           total_samples == 0 ? m_training_data.size() : total_samples);
    }

    if (m_training_data.empty()) {
        const std::string seed_text = ensure_seed_text();
        const std::vector<SeedSpec> greeting_samples = bootstrap_curriculum_for_profile(determine_seed_profile());

        std::size_t seed_total = greeting_samples.size();
        if (!seed_text.empty()) {
            ++seed_total;
        }

        if (seed_total > 0) {
            report_load_status("seeds", "Bootstrapping seed curriculum", 0, seed_total);
        } else {
            report_load_status("seeds", "No seed curriculum available", 0, 0);
        }

        std::size_t seed_completed = 0;
        auto register_seed_sample = [&](const std::string& prompt,
                                        const std::string& teacher_output,
                                        const std::string& prompt_hash,
                                        const char* teacher_hash = nullptr) {
            if (prompt.empty() || teacher_output.empty()) {
                return;
            }

            CuratedSample sample;
            sample.prompt = prompt;
            sample.teacher_output = teacher_output;
            sample.constraints = Json(JsonObject{});
            JsonObject provenance;
            provenance["source"] = Json("seed");
            provenance["prompt_hash"] = Json(prompt_hash);
            if (teacher_hash && teacher_hash[0] != '\0') {
                provenance["teacher_hash"] = Json(teacher_hash);
            } else {
                provenance["teacher_hash"] = Json(std::to_string(std::hash<std::string>{}(sample.teacher_output)));
            }
            sample.provenance = Json(provenance);
            sample.semantic_tags = compute_semantic_tags(sample);

            const std::size_t before_vocab = m_tokenizer.vocab().size();
            m_tokenizer.build_vocab({sample.prompt, sample.teacher_output});
            if (m_tokenizer.vocab().size() > before_vocab) {
                m_student.base().resize_vocab(m_tokenizer.vocab().size());
            }

            m_training_data.push_back(sample);
            CuratedSample& stored = m_training_data.back();
            m_curator.register_curated(stored);
            if (m_eval_data.size() < 16) {
                m_eval_data.push_back(stored);
            }
            const std::size_t index = m_training_data.size() - 1;
            const std::string document_id = derive_document_id(stored, index);
            if (!document_id.empty()) {
                if (stored.provenance.is_object()) {
                    stored.provenance.as_object()["sample_hash"] = Json(document_id);
                }
                std::string retrieval_text = stored.prompt;
                if (!retrieval_text.empty() && !stored.teacher_output.empty()) {
                    retrieval_text.append("\n\n");
                }
                retrieval_text.append(stored.teacher_output);
                stored.semantic_tags = merge_semantic_tags(stored.semantic_tags, m_retrieval.tags_for(document_id));
                m_retrieval.ingest_document(document_id, retrieval_text, stored.semantic_tags);
                m_document_to_index[document_id] = index;
            } else {
                std::string retrieval_text = stored.prompt;
                if (!retrieval_text.empty() && !stored.teacher_output.empty()) {
                    retrieval_text.append("\n\n");
                }
                retrieval_text.append(stored.teacher_output);
                std::hash<std::string> hasher;
                std::ostringstream oss;
                oss << "seed:" << index << ':' << hasher(stored.prompt + stored.teacher_output);
                const std::string fallback_id = oss.str();
                stored.semantic_tags = merge_semantic_tags(stored.semantic_tags, m_retrieval.tags_for(fallback_id));
                m_retrieval.ingest_document(fallback_id, retrieval_text, stored.semantic_tags);
                m_document_to_index[fallback_id] = index;
            }

            m_retrieval.save_metadata(kRetrievalMetadataPath);

            m_tokenizer.save_vocab(kVocabPath.string());
            persist_sample(stored);
            train_step(stored);

            ++seed_completed;
            if (seed_total > 0) {
                std::ostringstream detail;
                detail << "Registered seed sample " << seed_completed << " of " << seed_total;
                if (!prompt_hash.empty()) {
                    detail << " (" << prompt_hash << ")";
                }
                report_load_status("seeds", detail.str(), seed_completed, seed_total);
            }
        };

        if (!seed_text.empty()) {
            register_seed_sample("Introduce AlmondAI to a new user.",
                                 seed_text,
                                 "seed::bootstrap");
        }

        for (const auto& sample : greeting_samples) {
            register_seed_sample(sample.prompt, sample.teacher_output, sample.prompt_hash, sample.teacher_hash);
        }

        if (seed_total > 0) {
            report_load_status("seeds", "Seed curriculum loaded", seed_completed, seed_total);
        }
    }

    report_load_status("ready", "Learner initialisation complete", m_training_data.size(), m_training_data.size());
}

void ContinuousLearner::load_samples_from_file(const std::filesystem::path& path,
                                               std::size_t total_samples_hint) {
    if (!std::filesystem::exists(path)) {
        report_load_status("samples", "Training data file not found", 0, total_samples_hint);
        return;
    }
    std::ifstream file(path);
    if (!file) {
        report_load_status("samples", "Failed to open training data file", 0, total_samples_hint);
        return;
    }
    std::string line;
    std::size_t loaded = 0;
    std::size_t last_reported = 0;
    const std::size_t step = total_samples_hint > 0
        ? std::max<std::size_t>(std::size_t{1}, total_samples_hint / 10)
        : std::size_t{25};
    auto notify_progress = [&](bool force) {
        if (!m_load_status_callback) {
            return;
        }
        if (!force) {
            if (loaded == 0) {
                return;
            }
            if (total_samples_hint > 0) {
                if (loaded < last_reported + step) {
                    return;
                }
            } else {
                if (loaded % step != 0) {
                    return;
                }
            }
        }
        last_reported = loaded;
        std::ostringstream detail;
        detail << "Loaded " << loaded;
        if (total_samples_hint > 0) {
            detail << " / " << total_samples_hint;
        }
        detail << " persisted samples";
        report_load_status("samples", detail.str(), loaded, total_samples_hint);
    };

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        if (auto sample = parse_sample_line(line)) {
            CuratedSample sample_value = *sample;
            sample_value.semantic_tags = compute_semantic_tags(sample_value);
            const std::size_t before_vocab = m_tokenizer.vocab().size();
            m_tokenizer.build_vocab({sample_value.prompt, sample_value.teacher_output});
            if (m_tokenizer.vocab().size() > before_vocab) {
                m_student.base().resize_vocab(m_tokenizer.vocab().size());
            }
            m_training_data.push_back(sample_value);
            CuratedSample& stored = m_training_data.back();
            m_curator.register_curated(stored);
            if (m_eval_data.size() < 16) {
                m_eval_data.push_back(stored);
            }
            const std::size_t index = m_training_data.size() - 1;
            const std::string document_id = derive_document_id(stored, index);
            std::string retrieval_text = stored.prompt;
            if (!retrieval_text.empty() && !stored.teacher_output.empty()) {
                retrieval_text.append("\n\n");
            }
            retrieval_text.append(stored.teacher_output);
            std::string retrieval_id = document_id;
            if (!document_id.empty()) {
                if (stored.provenance.is_object()) {
                    auto& prov = stored.provenance.as_object();
                    if (prov.find("sample_hash") == prov.end()) {
                        prov["sample_hash"] = Json(document_id);
                    }
                }
            } else {
                std::hash<std::string> hasher;
                std::ostringstream oss;
                oss << "sample:" << index << ':' << hasher(stored.prompt + stored.teacher_output);
                retrieval_id = oss.str();
            }
            stored.semantic_tags = merge_semantic_tags(stored.semantic_tags, m_retrieval.tags_for(retrieval_id));
            m_retrieval.ingest_document(retrieval_id, retrieval_text, stored.semantic_tags);
            m_document_to_index[retrieval_id] = index;
            ++loaded;
            notify_progress(false);
        }
    }

    if (loaded == 0) {
        report_load_status("samples", "No persisted samples were ingested", 0, total_samples_hint);
    } else {
        notify_progress(true);
    }

    if (loaded > 0) {
        m_retrieval.save_metadata(kRetrievalMetadataPath);
    }

    if (!m_training_data.empty()) {
        m_tokenizer.save_vocab(kVocabPath.string());
    }
}

const CuratedSample* ContinuousLearner::recall_sample(const std::string& document_id) const {
    auto it = m_document_to_index.find(document_id);
    if (it == m_document_to_index.end()) {
        return nullptr;
    }
    const std::size_t index = it->second;
    if (index >= m_training_data.size()) {
        return nullptr;
    }
    return &m_training_data[index];
}

std::vector<std::string> ContinuousLearner::prompts_for_tags(const std::vector<std::string>& required_tags) const {
    std::unordered_set<std::string> required;
    required.reserve(required_tags.size());
    for (const auto& tag : required_tags) {
        const std::string normalised = normalise_tag_value(tag);
        if (!normalised.empty()) {
            required.insert(normalised);
        }
    }

    std::unordered_set<std::string> seen_prompts;
    std::vector<std::string> prompts;
    prompts.reserve(m_training_data.size());

    for (const auto& sample : m_training_data) {
        if (!required.empty()) {
            std::unordered_set<std::string> sample_tags;
            sample_tags.reserve(sample.semantic_tags.size());
            for (const auto& tag : sample.semantic_tags) {
                const std::string normalised = normalise_tag_value(tag);
                if (!normalised.empty()) {
                    sample_tags.insert(normalised);
                }
            }
            bool matches = std::all_of(required.begin(), required.end(), [&](const std::string& tag) {
                return sample_tags.find(tag) != sample_tags.end();
            });
            if (!matches) {
                continue;
            }
        }
        if (seen_prompts.insert(sample.prompt).second) {
            prompts.push_back(sample.prompt);
        }
    }

    return prompts;
}

void ContinuousLearner::set_load_status_callback(LoadStatusCallback callback) {
    m_load_status_callback = std::move(callback);
}

void ContinuousLearner::persist_sample(const CuratedSample& sample) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(kTrainingDataPath.parent_path(), ec);
    std::ofstream file(kTrainingDataPath, std::ios::app);
    if (!file) {
        return;
    }
    JsonObject obj;
    obj["prompt"] = Json(sample.prompt);
    obj["teacher_output"] = Json(sample.teacher_output);
    obj["constraints"] = sample.constraints;
    obj["provenance"] = sample.provenance;
    if (!sample.semantic_tags.empty()) {
        JsonArray tags;
        tags.reserve(sample.semantic_tags.size());
        for (const auto& tag : sample.semantic_tags) {
            tags.emplace_back(Json(tag));
        }
        obj["semantic_tags"] = Json(tags);
    }
    file << Json(obj).dump() << '\n';
}

std::string ContinuousLearner::derive_document_id(const CuratedSample& sample, std::size_t index) const {
    if (sample.provenance.is_object()) {
        const auto& prov = sample.provenance.as_object();
        if (auto it = prov.find("sample_hash"); it != prov.end() && it->second.is_string()) {
            const std::string& value = it->second.as_string();
            if (!value.empty()) {
                return value;
            }
        }
        if (auto it = prov.find("prompt_hash"); it != prov.end() && it->second.is_string()) {
            const std::string& value = it->second.as_string();
            if (!value.empty()) {
                if (auto teacher_it = prov.find("teacher_hash"); teacher_it != prov.end() && teacher_it->second.is_string()) {
                    const std::string& teacher_hash = teacher_it->second.as_string();
                    if (!teacher_hash.empty()) {
                        return value + "::" + teacher_hash;
                    }
                }
                return value;
            }
        }
    }
    std::hash<std::string> hasher;
    std::ostringstream oss;
    oss << "sample:" << index << ':' << hasher(sample.prompt + sample.teacher_output);
    return oss.str();
}

void ContinuousLearner::report_load_status(std::string_view phase,
                                           std::string_view detail,
                                           std::size_t completed,
                                           std::size_t total) {
    if (!m_load_status_callback) {
        return;
    }
    LoadStatus status;
    status.phase = std::string(phase);
    status.detail = std::string(detail);
    status.completed = completed;
    status.total = total;
    m_load_status_callback(status);
}

} // namespace almondai

