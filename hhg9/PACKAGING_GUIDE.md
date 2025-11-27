# Packaging guidance for `hhg9`

The package now follows a `src/` layout so wheels include the Python modules while keeping tests and examples out of the install.
This guide summarizes what changed and how to build and develop against the package locally.

## Layout

```
hhg9/
├── pyproject.toml
├── README.md            # PyPI long description
├── LICENSE              # Apache-2.0 license referenced in metadata
├── src/
│   └── hhg9/            # installable package code
└── tests/               # local tests, not shipped in wheels
```

Key points:
- `pyproject.toml` declares `package-dir = {"" = "src"}` and searches for packages under `src/`, matching the current tree.
- The package README and license live beside `pyproject.toml`, so metadata generation succeeds during builds.

## Building

From the `hhg9/` directory:
1. Install build tooling if needed: `python -m pip install --upgrade build`.
2. Create artifacts: `python -m build`.
3. Inspect `dist/` to confirm the wheel includes `hhg9` modules and metadata files.

## Local development

- Editable install: run `pip install -e .[dev]` from `hhg9/` to develop against the `src/` tree.
- Tests: keep test modules under `hhg9/tests/` so they run locally without being bundled into the wheel.
- Examples/notebooks: place them outside `src/` (e.g., `examples/`) to avoid shipping them unintentionally.
