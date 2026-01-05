from collections import defaultdict
from typing import List, Dict

import os
import regex as re
from typing import List, BinaryIO, Tuple

class RegexTokenizer:
    def __init__(self, filepath:str, PAT=None, pattern=None, special_tokens: List[str]=["<|endoftext|>"] ):
        self.filepath = filepath
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pattern = "|".join(map(re.escape, self.special_tokens))

        
    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))


    def pretokenize(self) -> List[str]:
        tokens = []
        with open(self.filepath, "rb") as f:
            num_processes = 4
            boundaries = self.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # 规范化换行符：Windows CRLF -> LF
                chunk = chunk.replace('\r\n', '\n').replace('\r', '\n')
                chunk_parts = re.split(self.pattern, chunk)
                tokens.extend([match.group() for chunk_item in chunk_parts for match in re.finditer(self.PAT, chunk_item)])
                # Run pre-tokenization on your chunk and store the counts for each pre-token
        return tokens
    
    def potokenize_text(self, text:str) -> List[str]:
        text_parts = re.split(self.pattern, text)
        tokens = [match.group() for chunk_item in text_parts for match in re.finditer(self.PAT, chunk_item)]
        return tokens
    
class BPE_Tokenizer_Trainer:
    def __init__(self, input_path:str, vocab_size:int, special_tokens:list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens else []

        self.word_freqs = {}
        self.merges:List[tuple[bytes, bytes]] = []
        self.vocab:Dict[int, bytes] = {}
        self.inverse_vocab: Dict[bytes, int] = {}


    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        min_vocab_size = 256 + len(self.special_tokens)
        self.vocab, self.inverse_vocab = self._initialize_vocab()

        assert self.vocab_size >= min_vocab_size, f"Vocab size must be at least {min_vocab_size}"

        self.vocab, self.inverse_vocab = self._initialize_vocab() # 初始化词表

        word_freqs = self._pretokenize_corpus() # 预分词并计算词频

        self.merges = self._compute_merges(word_freqs)  # 计算BPE合并规则

        self._update_vocab_with_merges() # 更新词表

        return self.vocab, self.merges
    
    def _initialize_vocab(self) -> Dict[int, bytes]:
        vocab = {}
        inverse_vocab = {}

        for i, token in enumerate(self.special_tokens):
            token_bytes = token.encode('utf-8')
            vocab[i] = token_bytes
            inverse_vocab[token_bytes] = i

        for i in range(256):
            token_id = len(self.special_tokens) + i
            byte_token = bytes([i])
            vocab[token_id] = byte_token
            inverse_vocab[byte_token] = token_id
    
        return vocab, inverse_vocab
    
    def _pretokenize_corpus(self) -> Dict[Tuple[bytes, ...], int]:
        word_freqs = defaultdict(int)
        regex_tokenizer = RegexTokenizer(
            filepath = self.input_path,
            special_tokens = self.special_tokens
        )

        tokens = regex_tokenizer.pretokenize()
        for token in tokens:
            # 将字符串 token 编码为 UTF-8 字节序列
            # 例如："你好" -> b'\xe4\xbd\xa0\xe5\xa5\xbd'
            # 语料中可能会包含多字节字符，如中文、日文、韩文等，不要拆分为单字节 token 再enco
            word_bytes = token.encode('utf-8')

            # 将字节序列拆分为单字节 token 的元组
            # 例如：b'cat' -> (b'c', b'a', b't')
            word_tuple = tuple(bytes([b]) for b in word_bytes)

            # 统计该 byte-level token 序列在语料中的出现频率
            word_freqs[word_tuple] += 1

        # 返回普通 dict，便于后续处理
        return dict(word_freqs)
    
    def _compute_merges(
            self,
            word_freqs: dict[tuple[bytes, ...], int]
    ) -> List[tuple[bytes, bytes]]:
        
        merges = []

        current_vocab_size = len(self.vocab)
        working_freqs = dict(word_freqs)

        target_merges = self.vocab_size - current_vocab_size

        for merge_num in range(target_merges):
            # 1. 统计所有 token 序列中相邻 byte pair 的出现频率
            pair_counts = self._count_pairs(working_freqs)

            if not pair_counts:
                break
            most_frequent_pair = max(
                pair_counts.items(),
                key=lambda x: (x[1], x[0])
            )[0]    
            # 3. 在所有 token 序列中执行该 pair 的合并操作
            #    (a, b) → ab
            working_freqs = self._merge_pair(
                working_freqs,
                most_frequent_pair
            )

            # 4. 将该 merge 规则记录下来（顺序很重要）
            merges.append(most_frequent_pair)

            # 返回完整的 BPE 合并规则列表
        return merges

    def _merge_pair(
            self,
            word_freqs: dict[tuple[bytes, ...], int],
            pair: tuple[bytes, bytes]
    ) -> dict[tuple[bytes, ...], int]:
        """
        在所有 token 序列中将指定的 byte pair 合并为单个 token。
        
        Args:
            word_freqs: 词频字典，键为 byte 元组，值为出现次数
            pair: 要合并的 byte pair，例如 (b'c', b'a')
        
        Returns:
            new_word_freqs: 合并后的词频字典
                            例如：{(b'ca', b't'): 5} （原来是 {(b'c', b'a', b't'): 5}）
        """
        new_word_freqs: dict[tuple[bytes, ...], int] = {}
        token_a, token_b = pair
        merged_token = token_a + token_b  # 合并后的新 token

        for word_tuple, freq in word_freqs.items():
            # 如果序列长度 < 2，无法合并，直接保留
            if len(word_tuple) < 2:
                new_word_freqs[word_tuple] = freq
                continue

            # 构建新的 token 序列，遇到 pair 就合并
            new_word: list[bytes] = []
            i = 0
            while i < len(word_tuple):
                # 检查当前位置是否匹配 pair
                if i < len(word_tuple) - 1 and word_tuple[i] == token_a and word_tuple[i + 1] == token_b:
                    new_word.append(merged_token)
                    i += 2  # 跳过两个 token
                else:
                    new_word.append(word_tuple[i])
                    i += 1

            new_word_tuple = tuple(new_word)
            # 相同序列的频率累加（不同原始序列可能合并成相同结果）
            new_word_freqs[new_word_tuple] = new_word_freqs.get(new_word_tuple, 0) + freq

        return new_word_freqs
    
    def _count_pairs(
            self,
            word_freqs: dict[tuple[bytes, ...], int]
    ) -> dict[tuple[bytes, bytes], int]:
        """
        统计所有 token 序列中相邻 byte pair 的出现频率。
        
        Args:
            word_freqs: 词频字典，键为 byte 元组，值为出现次数
                        例如：{(b'c', b'a', b't'): 5, (b'd', b'o', b'g'): 3}
        
        Returns:
            pair_counts: 相邻 pair 频率字典
                         例如：{(b'c', b'a'): 5, (b'a', b't'): 5, (b'd', b'o'): 3, (b'o', b'g'): 3}
        """
        pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)

        for word_tuple, freq in word_freqs.items():
            # 序列长度小于 2 时无法构成 pair
            if len(word_tuple) < 2:
                continue

            # 遍历相邻的 byte pair
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                # 累加频率（该 pair 在此序列出现 1 次，乘以序列频率）
                pair_counts[pair] += freq

        return dict(pair_counts)

    def _update_vocab_with_merges(self):
        # 基础词表大小：
        # = 特殊 token 数量 + 256 个单字节 token
        # 新生成的 BPE token 将从该索引之后依次编号
        base_vocab_size = 256 + len(self.special_tokens)

        # 依次遍历所有 merge 规则（顺序非常重要）
        for i, (token_a, token_b) in enumerate(self.merges):

            # 将 byte pair (a, b) 合并成一个新的 byte token
            # 例如：(b'h', b'e') -> b'he'
            merged_token = token_a + token_b

            # 为该合并 token 分配新的 token_id
            # token_id 连续递增，保证与 merge 顺序一致
            token_id = base_vocab_size + i

            # 更新正向词表：token_id -> byte 序列
            self.vocab[token_id] = merged_token

            # 更新反向词表：byte 序列 -> token_id
            self.inverse_vocab[merged_token] = token_id