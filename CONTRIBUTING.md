# Contributing to Spectral Encoder Phase 1

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Write/update tests
5. Submit a pull request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/quantumind/spectral-encoder-phase1.git
cd fbc-phase1

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest spectral_encoder/tests/
```

## Code Style

- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small

## Testing

- Write tests for new features
- Ensure all existing tests pass
- Aim for >85% code coverage
- Test edge cases and error conditions

## Commit Messages

Use clear, descriptive commit messages:
```
Short description (50 chars or less)

Longer explanation if needed (70 chars or less per line)

Fixes #123
```

## Pull Requests

- Reference related issues
- Describe changes clearly
- Include test cases
- Update documentation as needed

## Documentation

- Update README.md if changing behavior
- Add docstrings to code
- Create examples for new features
- Update API documentation

## Reporting Issues

Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

## Questions?

Open an issue or check the documentation in the project.

---

**Thank you for contributing!**
