# repoindex-analyzer-c

First-party C-family analyzer plugin for `repoindex`.

Repository-local editable install:

```bash
source .venv/bin/activate
pip install -e ../repoindex
pip install -e ../repoindex/packages/repoindex-analyzer-c
```

After installation, verify discovery with:

```bash
repoindex plugins
repoindex coverage
```

Package-local verification:

```bash
pytest -q packages/repoindex-analyzer-c/tests
```
