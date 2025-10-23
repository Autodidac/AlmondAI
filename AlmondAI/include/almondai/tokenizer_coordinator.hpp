#pragma once

#include "model.hpp"
#include "tokenizer_bpe.hpp"
#include "tokenizer_word.hpp"

#include <filesystem>
#include <mutex>
#include <optional>
#include <string_view>

namespace almondai {

class TokenizerCoordinator {
public:
    struct PersistenceConfig {
        std::filesystem::path word_vocab;
        std::filesystem::path bpe_vocab;
        std::filesystem::path bpe_merges;
    };

    struct IngestResult {
        std::size_t word_tokens_added = 0;
        std::size_t bpe_tokens_added = 0;
        bool student_resized = false;
    };

    TokenizerCoordinator() = default;

    WordTokenizer& word() noexcept { return m_word_tokenizer; }
    const WordTokenizer& word() const noexcept { return m_word_tokenizer; }
    BpeTokenizer& bpe() noexcept { return m_bpe_tokenizer; }
    const BpeTokenizer& bpe() const noexcept { return m_bpe_tokenizer; }

    void set_persistence(PersistenceConfig config);
    const PersistenceConfig& persistence() const noexcept { return m_paths; }

    IngestResult ingest_training_pair(StudentModel& student,
                                      std::string_view prompt,
                                      std::string_view teacher_output);

    void sync_student_vocab(StudentModel& student);

    void persist(std::optional<std::size_t> version = std::nullopt) const;

private:
    WordTokenizer m_word_tokenizer;
    BpeTokenizer m_bpe_tokenizer;
    PersistenceConfig m_paths;
    mutable std::mutex m_mutex;
    mutable bool m_dirty = false;

    static void ensure_parent_directory(const std::filesystem::path& path);
    void resize_student_if_needed(StudentModel& student, IngestResult& result);
    void persist_locked(std::optional<std::size_t> version) const;
};

} // namespace almondai

