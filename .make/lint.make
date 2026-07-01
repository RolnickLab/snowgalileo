## -- Linting targets ----------------------------------------------------------------------------------------------- ##

.PHONY: precommit
precommit: ## Run Pre-commit on all files manually (Only lint target that works without dev dependencies)
	@$(ENV_COMMAND_TOOL) pre-commit run --all-files

.PHONY: check-lint
check-lint: ## Check code linting (ruff, docformatter and pylint)
	@$(ENV_COMMAND_TOOL) nox -s check

.PHONY: fix-lint
fix-lint: ## Fix code linting (ruff, flynt, docformatter)
	@$(ENV_COMMAND_TOOL) nox -s fix

.PHONY: autotyping
autotyping: ## Add basic types using autotyping
	@$(ENV_COMMAND_TOOL) nox -s autotyping

.PHONY: mypy
mypy: ## Check code with mypy
	@$(ENV_COMMAND_TOOL) nox -s mypy

.PHONY: pylint
pylint: ## Check code with pylint
	@$(ENV_COMMAND_TOOL) nox -s pylint

.PHONY: markdown-lint
markdown-lint: ## Fix markdown linting using mdformat
	@$(ENV_COMMAND_TOOL) nox -s mdformat

.PHONY: ruff
ruff: ## Run the ruff linter
	@$(ENV_COMMAND_TOOL) nox -s ruff-lint

.PHONY: ruff-fix
ruff-fix: ## Run the ruff linter and fix automatically fixable errors
	@$(ENV_COMMAND_TOOL) nox -s ruff-fix

.PHONY: ruff-format
ruff-format: ## Run the ruff code formatter
	@$(ENV_COMMAND_TOOL) nox -s ruff-format
