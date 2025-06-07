# Contributing to Tarmac

We love your input! We want to make contributing to Tarmac as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Licensing

By contributing to Tarmac, you agree that your contributions will be licensed under its GNU Affero General Public License v3.0 (AGPL-3.0).

This means that:
- Your contributions must be made available under the AGPL-3.0 license
- Any modifications to the code, including when used to provide a network service, must be made available under AGPL-3.0
- You grant Adam Rida and any future legal entities created to maintain and develop Tarmac (including but not limited to companies, organizations, or other corporate structures) the right to use your contributions under the AGPL-3.0 license

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/your-username/tarmac.git
   cd tarmac
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .
   ```

3. Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **Ruff** for linting
- **pre-commit** hooks to automate checks

Your code should pass all these checks before being submitted. The pre-commit hooks will run automatically when you commit changes.

## Testing

We use pytest for testing. To run the tests:

```bash
pytest
```

When adding new features, please include appropriate tests in the `tests/` directory.

## Pull Request Process

1. Fork the repo and create your branch from `master`:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. Make your changes and ensure they follow our coding standards
3. Write or update tests as needed
4. Update documentation if you're changing functionality
5. Run the test suite and ensure it passes
6. Push your changes and create a pull request

## Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. Your commit messages should be structured as follows:

```
<type>: <description>

[optional body]
[optional footer]
```

Types include:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat: add support for regression models

Added capability to compare regression models using epsilon-based difference detection.
```

## Report Issues

We use GitHub issues to track public bugs and feature requests. Report a bug by [opening a new issue](https://github.com/adrida/tarmac/issues/new).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Feature Requests

We love feature requests! Please open an issue to suggest new features, and tag it with `enhancement`. Include:

- The use case you have in mind
- How this feature would help you
- Any ideas about implementation

## Any contributions you make will be under the AGPL-3.0 Software License

When you submit code changes, your submissions are understood to be under the same [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.en.html) that covers the project. By contributing, you agree that your contributions will be licensed under its AGPL-3.0 License. 