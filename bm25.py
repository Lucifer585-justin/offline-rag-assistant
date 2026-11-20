"""
bm25.py — 关键词检索（BM25）

为什么 RAG 需要 BM25：
    向量检索擅长"语义相近"，但对**专有名词**很弱。
    问"组相联映射的 k 是多少"，向量可能召回一段讲"Cache 结构"的泛泛文字，
    而真正的答案里恰好有"组相联"三个字 —— BM25 一抓就中。

为什么用 RRF 而不是把两个分数加权平均：
    cosine 在 0~1，BM25 是无上界的实数，两者量纲不同，加权平均需要拍脑袋定权重，
    换语料后就失效。RRF（Reciprocal Rank Fusion）只用**排名**：
        score = Σ 1 / (k + rank_i)
    无需归一化、无需调参，是工业界事实标准（Elasticsearch / Vespa 都在用）。

中文分词：
    装了 jieba 就用 jieba；没装自动退回"字 bigram + 英文/数字整词"。
    bigram 对中文检索的效果接近分词（学界有共识），代价是索引稍大。
    这样本项目依然是零强制依赖：pip install jieba 只是可选增强。
"""

from __future__ import annotations

import math
import re
from collections import Counter

try:                                    # 可选增强
    import jieba
    jieba.setLogLevel(60)               # 关掉 "Building prefix dict" 的输出
    _HAS_JIEBA = True
except Exception:                       # noqa: BLE001
    _HAS_JIEBA = False

_WORD = re.compile(r"[a-zA-Z]+|\d+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """
    统一的切词入口。中英混排的教材文本要分别处理：
      - 英文/数字：按词切，转小写
      - 中文：jieba 分词；没有 jieba 就用相邻二字 bigram
    """
    if not text:
        return []
    text = text.lower()

    if _HAS_JIEBA:
        tokens = [t for t in jieba.lcut(text) if t.strip()]
    else:
        tokens = _WORD.findall(text)
        cjk = _CJK.findall(text)
        # 只取"相邻二字"，**不保留单字**：
        # 单字（的/是/在/怎/么）在任何中文语料里都高频，IDF 低但小语料下仍能凑出分数，
        # 实测会让"今天天气怎么样"这种无关问题拿到 BM25 > 1，导致防幻觉闸门失效。
        # bigram 的语义密度高得多："今天""天气" 在计组语料里就是不存在的词。
        tokens += [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]

    return [t for t in tokens if t.strip() and not t.isspace()]


class BM25:
    """
    标准 Okapi BM25，参数取工程默认值：
        k1 = 1.5   词频饱和速度（越大越"宽容"高频词）
        b  = 0.75  文档长度归一化强度
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avg_len: float = 0.0
        self.df: Counter = Counter()          # 文档频率
        self.n_docs: int = 0
        self._idf_cache: dict[str, float] = {}

    def fit(self, docs: list[str]) -> None:
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.n_docs = len(docs)
        self.avg_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.df = Counter()
        for toks in self.doc_tokens:
            for term in set(toks):
                self.df[term] += 1
        self._idf_cache = {}

    def _idf(self, term: str) -> float:
        """
        BM25 的 IDF：log((N - df + 0.5) / (df + 0.5) + 1)
        +0.5 平滑避免除零，+1 保证结果非负（否则常见词会得负分，反而把文档排到最后）
        """
        if term in self._idf_cache:
            return self._idf_cache[term]
        df = self.df.get(term, 0)
        val = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
        self._idf_cache[term] = val
        return val

    def scores(self, query: str) -> list[float]:
        """返回与 fit() 时文档列表等长的分数数组。"""
        out = [0.0] * self.n_docs
        if not self.n_docs:
            return out
        q_terms = tokenize(query)
        if not q_terms:
            return out

        for i, toks in enumerate(self.doc_tokens):
            if not toks:
                continue
            tf = Counter(toks)
            dl = self.doc_len[i] or 1
            s = 0.0
            for term in q_terms:
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
                s += self._idf(term) * (f * (self.k1 + 1)) / denom
            out[i] = s
        return out


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """
    Reciprocal Rank Fusion：把多路检索的**排名**融合成一个分数。
        score(d) = Σ_i 1 / (k + rank_i(d))
    k 取 60 是原论文的推荐值，作用是压低"第一名"的绝对优势，让多路结果更均衡。
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return fused
