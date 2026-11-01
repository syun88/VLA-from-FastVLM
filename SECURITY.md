# Security Policy

## Supported versions

| Version | Status |
| --- | --- |
| `v0.2.0-alpha` | Supported for security reports |
| `v0.1.0-alpha` | Legacy PyTorch workflow; security fixes are limited to critical issues |

This project is currently alpha software. The code is intended for research
workflows and should not be treated as a certified robotics safety system.

Model loading uses upstream MLX-VLM/Transformers code and optional remote model repositories.
Treat community checkpoints and remote code as untrusted, pin revisions, and run them with the
least filesystem/network access available in your environment. Never place Hugging Face or W&B
tokens in source files, result files, notebook outputs, or workflow YAML.

## Reporting a vulnerability

Please do not open a public issue for vulnerabilities. Use GitHub private
vulnerability reporting if it is enabled for this repository. If that option is
not available, contact the maintainer through GitHub with a minimal description
and a safe way to share details privately.

Helpful reports include:

- Affected commit, tag, or release.
- Steps to reproduce.
- Whether the issue affects local training, model loading, dataset handling, or
  robot execution.
- Any recommended mitigation.

## Robotics safety scope

Real robot rollout is not yet validated in this alpha release. Users are
responsible for independent safety controls, workspace limits, emergency stop
procedures, and human supervision before running policies on hardware.
