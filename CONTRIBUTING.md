# Contributing to ASR-Pro

First off, thank you for considering contributing to ASR-Pro! It's people like you that make ASR-Pro the leading open-source speech intelligence platform for contact centers.

## 1. Where do I go from here?

If you've noticed a bug or have a feature request, make sure to check our [Issues](https://github.com/ardamoustafa/ASR-Pro/issues) page to see if someone else in the community has already created a ticket. If not, go ahead and make one!

## 2. Setting up your environment

We encourage using the standard fork-and-branch workflow:

1. **Fork** the repository on GitHub.
2. **Clone** the project to your own machine:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ASR-Pro.git
   cd ASR-Pro
   ```
3. **Add Upstream Remote**:
   ```bash
   git remote add upstream https://github.com/ardamoustafa/ASR-Pro.git
   ```

### Local Development Setup

1. **Python Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Environment Variables**:
   ```bash
   cp .env.example .env
   # Edit .env — set ASR_JWT_SECRET_KEY and ASR_ADMIN_PASSWORD at minimum
   ```
3. **Frontend Setup**:
   ```bash
   npm install
   ```
4. **Database Initialization** (creates schema and seeds default data):
   ```bash
   python -m asr_pro.db.seed
   # Or equivalently:
   python -c "from asr_pro.db.seed import init_db_with_seed; init_db_with_seed()"
   ```
5. **Run Locally**:
   ```bash
   make dev
   ```
   - Backend API: http://localhost:8000/api/docs
   - Frontend Dashboard: http://localhost:5173

## 3. Developer Workflow & Conventions

### Branch Naming Convention
Please use the following prefixes for your branch names:
- `feature/` for new features (e.g., `feature/live-translation`)
- `fix/` or `hotfix/` for bug fixes (e.g., `fix/websocket-timeout`)
- `docs/` for documentation updates
- `chore/` for tooling or dependency updates

### Commit Message Format (Conventional Commits)
We strictly follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.
Format: `<type>(<scope>): <subject>`

**Examples**:
- `feat(asr): add support for whisper v3 turbo`
- `fix(ui): resolve overflow issue on mobile dashboard`
- `docs(readme): update installation instructions`
- `test(compliance): add negation filtering tests`

Valid types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

### Code Style & Testing
* **Linting:** We use `ruff` for linting and formatting. Run `ruff check .` and `ruff format .` before pushing.
* **Import Order:** We use `isort`. Run `isort asr_pro/ tests/` before pushing.
* **Security:** Run `bandit -r asr_pro/` to check for security issues.
* **Testing:** ASR-Pro requires high test coverage for all Core Engines (`asr_pro/core/`). Before submitting your PR, ensure that you have written tests for your new logic and that `pytest tests/` passes successfully.
* **Architecture:** Do not bypass the FastAPI endpoints or the SQLAlchemy models. All logic should flow through the designated controllers.

## 4. Submitting a Pull Request

1. Fetch the latest `main` branch from upstream and rebase if necessary.
2. Provide a detailed description of the changes you've made in the PR template.
3. If your PR resolves an open issue, link the issue in the description (e.g., `Fixes #123`).
4. Include screenshots if your changes affect the React User Interface.

Thank you for contributing! 🎉

<!-- 
  ==============================================================================
  Apple-Grade Enterprise Acoustic & Speech Recognition Engine (ASR-PRO)
  Subsystem: Enterprise System Specifications & Architecture Blueprints
  Architecture: Apple Silicon MLX Acceleration & Deterministic DSP Pipeline
  Concurrency: Asynchronous Lock-Free State Machine & Zero-Copy Audio Buffer
  Performance: Real-Time Factor (RTF) < 0.08 on Apple M-Series Neural Engine
  ============================================================================== 
-->
