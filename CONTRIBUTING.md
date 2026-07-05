# Contributing to Bifrost

## Setup

```bash
git clone https://github.com/quantumind/bifrost.git
cd bifrost
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Standards

- Follow PEP 8. Use `black` (line length 100).
- Document all public functions: purpose, parameters, return type.
- Non-trivial algorithms note Big-O complexity.
- Handle errors explicitly; silent failures are not acceptable.
- Target > 85% test coverage for new code.

## Workflow

1. Fork the repository and create a feature branch.
2. Write a failing test for the behaviour you are adding or fixing.
3. Implement the change.
4. Confirm all tests pass: `pytest tests/ -v`.
5. Submit a pull request with a clear description of the change and why.

## Commit messages

```
Short description (≤ 50 characters)

Longer explanation if needed (wrap at 72 characters).

Fixes #123
```

## Reporting issues

Include: Python version, OS, steps to reproduce, expected vs. actual behaviour, full error traceback.

## Questions

Open an issue or email engineering@quantumind.io.
