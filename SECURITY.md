# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly. **Do not open a public GitHub issue for security vulnerabilities.**

### How to Report

Please use **GitHub's private vulnerability reporting** as the primary channel:

1. **GitHub Private Vulnerability Reporting** (preferred): open the repository's **Security** tab → **Report a vulnerability**, or use [this link](../../security/advisories/new). This creates a private advisory visible only to you and the maintainers.
2. **Email** (alternative): if you cannot use GitHub, email **opensource@gruposantander.com**.

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response SLA

| Stage | SLA |
|:---|:---|
| Acknowledgment of report | < 48 hours |
| Initial assessment and severity classification | < 7 days |
| Fix for Critical/High severity | < 30 days |
| Fix for Medium/Low severity | < 90 days |

### What Happens Next

1. We will acknowledge your report within 48 hours.
2. We will investigate and determine the severity.
3. We will develop and test a fix.
4. We will release the fix and publish a security advisory.
5. We will credit you publicly in the advisory and CHANGELOG (unless you prefer to remain anonymous).

## Scope

This security policy applies **only** to code in this repository. It does not cover:

- Santander's internal infrastructure or systems
- Other Santander products or services
- Third-party dependencies (report those to the respective maintainers)

## Supported Versions

| Version | Supported |
|:---|:---|
| Latest release | Yes |
| Previous minor release | Security fixes only |
| Older versions | No |

## Security Best Practices for Contributors

- Never commit secrets, API keys, tokens, or credentials
- Never commit internal URLs, IP addresses, or corporate email addresses
- Never commit personally identifiable information (PII) or customer data
- Use environment variables for any configuration that could be sensitive
- Keep dependencies up to date (Dependabot is enabled on this repository)

## Disclosure Policy

We follow a coordinated disclosure process. We ask that you:

- Give us reasonable time to fix the vulnerability before public disclosure
- Do not exploit the vulnerability beyond what is necessary to demonstrate it
- Do not access or modify data that does not belong to you
