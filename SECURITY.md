# Security Policy

## Supported versions

| Version | Status |
| --- | --- |
| `v0.1.0-alpha` | Supported for security reports |

This project is currently alpha software. The code is intended for research
workflows and should not be treated as a certified robotics safety system.

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
