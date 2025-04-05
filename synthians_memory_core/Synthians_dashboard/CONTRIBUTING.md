# Contributing to Synthians Cognitive Dashboard

Thank you for considering contributing to the Synthians Cognitive Dashboard! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Please read it to understand expected behavior.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue tracker to see if the problem has already been reported. If it hasn't, create a new issue with a clear description, including:

- A clear and descriptive title
- Steps to reproduce the behavior
- Expected behavior
- Screenshots (if applicable)
- Environment details (browser, OS, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- Use a clear and descriptive title
- Provide a detailed description of the proposed enhancement
- Explain why this enhancement would be useful
- Include mockups or examples if possible

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

## Coding Standards

### TypeScript

- Use TypeScript for all new code
- Ensure proper typing for all variables, parameters, and return values
- Follow the existing project structure for new files

### React Components

- Use functional components with hooks
- Keep components small and focused on a single responsibility
- Use the shadcn component library for consistent UI

### Styling

- Use TailwindCSS for styling
- Follow the existing theme configuration
- Use the provided color variables for consistency

### Testing

- Write tests for new features using the testing framework in place
- Ensure all tests pass before submitting a PR
- Aim for good test coverage for new code

## Commit Guidelines

- Use clear, concise commit messages
- Reference issue numbers in commit messages when applicable
- Keep commits focused on a single logical change

## Documentation

- Update documentation when changing functionality
- Document new features, including:
  - Usage examples
  - API documentation
  - Configuration options

## Release Process

1. Version bump follows semantic versioning (MAJOR.MINOR.PATCH)
2. Releases are created from the main branch
3. Release notes document all significant changes

Thank you for contributing to the Synthians Cognitive Dashboard!