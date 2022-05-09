# Changelog

CLIP-as-service follows semantic versioning. However, before the project reach 1.0.0, any breaking change will only bump the minor version.  An automated release note is [generated on every release](https://github.com/jina-ai/clip-as-service/releases). The release note includes features, bugs, refactorings etc. 

This chapter only tracks the most important breaking changes and explain the rationale behind them.

## 0.4.0: rename `rerank` concept to `rank`

"Reranking" is a new feature introduced since 0.3.3. This feature allows user to rank and score `document.matches` in a cross-modal way. From 0.4.0, this feature as well as all related functions will refer it simply as "rank".

## 0.2.0: improve the service scalability with replicas

This change is mainly intended to improve the inference performance with replicas.

Here is the short benchmark summary of the improvement (`replicas=4`):

| batch_size  | before | after   |
|-------------|--------|---------|
| 1           | 23.74  | 18.89   |
| 8           | 58.88  | 30.38   |
| 16          | 14.96  | 91.86   |
| 32          | 14.78  | 101.75  |
