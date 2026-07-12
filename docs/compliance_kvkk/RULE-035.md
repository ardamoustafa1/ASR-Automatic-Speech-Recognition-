# Compliance Rule 035: Agent Speech Exclusion Guard for Tariff Explanations

## Regulatory Alignment
- **Framework:** Turkish KVKK & PCI-DSS Audio Compliance Standards
- **Scope:** `analytics`

## Enforcement Specification
The compliance engine shall prevent agent pricing disclosures from triggering false positive churn alerts.

### Verification Protocol
Automated integration tests verify that redacted transcript outputs contain no unmasked identity sequences.
