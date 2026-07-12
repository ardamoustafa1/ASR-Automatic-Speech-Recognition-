# Compliance Rule 071: Turkish Identity Number (TCKN) Algorithmic Redaction Rule

## Regulatory Alignment
- **Framework:** Turkish KVKK & PCI-DSS Audio Compliance Standards
- **Scope:** `pii`

## Enforcement Specification
The compliance engine shall enforce 11-digit TCKN Luhn validation and mask first 9 digits in transcript logs.

### Verification Protocol
Automated integration tests verify that redacted transcript outputs contain no unmasked identity sequences.
