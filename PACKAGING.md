# Packaging status and next steps

This repository is set up to publish the `hhg9` package from the repository root with `setuptools`. The key configuration lives in `pyproject.toml`, which now declares the package discovery rules and long description source. Before releasing, run through the checks below:

1. **Validate package discovery.**
   - Configuration is defined under `[tool.setuptools.packages.find]` with `where = ["."]` and `include = ["hhg9*"]`, which discovers the root-level `hhg9` package and its submodules.
   - Run `python -m build` (requires `build`) and inspect the generated wheel/SDist to confirm `hhg9` files are present.

2. **Long description source.**
   - `pyproject.toml` points `project.readme` to the plain `README` file with `content-type = "text/markdown"` so PyPI renders it correctly even without a `.md` extension.
   - Keep `README` self-contained (no external images/assets) to avoid broken links on PyPI.

3. **Pre-release sanity checks.**
   - Verify dependency bounds are accurate for all supported Python versions (`>=3.10`).
   - Tag the release in version control and ensure `version = "0.1.0a0"` is updated when publishing.
   - Consider adding `project.urls` (e.g., documentation, repository, issue tracker) for better PyPI metadata.

Running these steps should surface any remaining packaging issues before publishing to PyPI.
