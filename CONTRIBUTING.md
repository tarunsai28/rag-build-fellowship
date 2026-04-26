# Contributing

Pull requests welcome. This is primarily a teaching repository, so
contributions that improve clarity for beginners are valued more highly than
contributions that add new capabilities.

## Workflow

1. Fork the repo, branch from `main`.
2. Make your change. Add tests if you add behaviour.
3. Run the quality gates locally before pushing:

   ```bash
   uv run ruff check .
   uv run ruff format .
   uv run pytest
   ```

4. Open a PR. Keep the change small and focused — one fix per PR is far
   easier to review than a sweeping refactor.

## Things we'd happily merge

- Better sample documents (more interesting, still permissively licensed).
- Improved error messages — anything that turns a stack trace into a
  one-line "here's what went wrong and how to fix it".
- Additional troubleshooting entries for issues you ran into.
- Small, well-tested provider additions (e.g. Mistral, Cohere, Voyage).
- Documentation fixes.

## Things we probably won't merge

- LangChain / LlamaIndex integrations. The whole point of the codebase is
  that it doesn't use them. (They are great libraries — wrong fit for
  *this* repo.)
- Major architectural changes without prior discussion in an issue.
- New required dependencies. Optional extras are fine; new defaults raise
  the bar to setup, which works against students.
- Code that duplicates or replaces logic listed as a Workshop deliverable.

## Style

- Docstrings on every public function.
- Type hints on every signature.
- `ruff` for linting and formatting (configured in `pyproject.toml`).
- Comments only for the *why*, not the *what*. The code already says what.
