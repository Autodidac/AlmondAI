#include "../include/almondai/tokenizer_coordinator.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <system_error>

namespace almondai {

void TokenizerCoordinator::set_persistence(PersistenceConfig config) {
    std::scoped_lock lock(m_mutex);
    m_paths = std::move(config);
}

TokenizerCoordinator::IngestResult TokenizerCoordinator::ingest_training_pair(
    StudentModel& student,
    std::string_view prompt,
    std::string_view teacher_output) {
    IngestResult result;
    {
        std::scoped_lock lock(m_mutex);
        result.word_tokens_added = m_word_tokenizer.ingest_training_pair(prompt, teacher_output);
        result.bpe_tokens_added = m_bpe_tokenizer.ingest_training_pair(prompt, teacher_output);
        if (result.word_tokens_added > 0 || result.bpe_tokens_added > 0) {
            m_dirty = true;
        }
    }
    resize_student_if_needed(student, result);
    if (result.word_tokens_added > 0 || result.bpe_tokens_added > 0) {
        persist();
    }
    return result;
}

void TokenizerCoordinator::resize_student_if_needed(StudentModel& student, IngestResult& result) {
    std::size_t target_vocab = std::max(m_word_tokenizer.vocab_size(), m_bpe_tokenizer.vocab_size());
    const std::size_t current = student.base().config().vocab_size;
    if (target_vocab > current) {
        student.base().resize_vocab(target_vocab);
        result.student_resized = true;
    }
}

void TokenizerCoordinator::sync_student_vocab(StudentModel& student) {
    IngestResult result;
    resize_student_if_needed(student, result);
}

void TokenizerCoordinator::ensure_parent_directory(const std::filesystem::path& path) {
    if (path.empty() || !path.has_parent_path()) {
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
}

void TokenizerCoordinator::persist(std::optional<std::size_t> version) const {
    std::scoped_lock lock(m_mutex);
    if (!m_dirty && !version) {
        return;
    }
    persist_locked(version);
    m_dirty = false;
}

void TokenizerCoordinator::persist_locked(std::optional<std::size_t> version) const {
    if (!m_paths.word_vocab.empty()) {
        ensure_parent_directory(m_paths.word_vocab);
        m_word_tokenizer.save_vocab(m_paths.word_vocab.string());
        if (version) {
            std::ostringstream name;
            name << m_paths.word_vocab.stem().string() << ".step" << std::setw(6) << std::setfill('0') << *version
                 << m_paths.word_vocab.extension().string();
            std::filesystem::path version_path = m_paths.word_vocab.parent_path() / name.str();
            std::error_code ec;
            std::filesystem::copy_file(m_paths.word_vocab, version_path, std::filesystem::copy_options::overwrite_existing, ec);
        }
    }
    if (!m_paths.bpe_vocab.empty()) {
        ensure_parent_directory(m_paths.bpe_vocab);
        m_bpe_tokenizer.save_vocab(m_paths.bpe_vocab);
        if (version) {
            std::ostringstream name;
            name << m_paths.bpe_vocab.stem().string() << ".step" << std::setw(6) << std::setfill('0') << *version
                 << m_paths.bpe_vocab.extension().string();
            std::filesystem::path version_path = m_paths.bpe_vocab.parent_path() / name.str();
            std::error_code ec;
            std::filesystem::copy_file(m_paths.bpe_vocab, version_path, std::filesystem::copy_options::overwrite_existing, ec);
        }
    }
    if (!m_paths.bpe_merges.empty()) {
        ensure_parent_directory(m_paths.bpe_merges);
        m_bpe_tokenizer.save_merges(m_paths.bpe_merges);
        if (version) {
            std::ostringstream name;
            name << m_paths.bpe_merges.stem().string() << ".step" << std::setw(6) << std::setfill('0') << *version
                 << m_paths.bpe_merges.extension().string();
            std::filesystem::path version_path = m_paths.bpe_merges.parent_path() / name.str();
            std::error_code ec;
            std::filesystem::copy_file(m_paths.bpe_merges, version_path, std::filesystem::copy_options::overwrite_existing, ec);
        }
    }
}

} // namespace almondai

