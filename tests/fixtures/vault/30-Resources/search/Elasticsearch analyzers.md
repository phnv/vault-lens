---
id: 202610031010
type: atomic
status: active
tags: ai/rag search
---
# Elasticsearch analyzers for PT/EN notes

## Analyzer choice

Combine a Portuguese analyzer with ASCII folding so reuniões de projeto matches "reuniao".

```python
# [[not-a-link]] #not-a-tag
x = "[fake](other.md)"
```

## Folding accents

See [local](../dayo/dayo-notes-server.md) and ![[diagram.png]]. Inline `code #nope` ignored. #search/tuning
