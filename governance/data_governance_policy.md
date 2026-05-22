# Data Governance Policy
## Multilingual African Health Assistant

**Version:** 1.0 | **Effective:** 2025 | **Review:** Annual

---

## 1. Scope

This policy governs all data collected, processed, stored, and shared in connection with the Multilingual African Health Assistant project, including training data, user interactions, feedback logs, and model outputs.

---

## 2. Applicable Regulations

| Jurisdiction | Regulation |
|---|---|
| Nigeria | Nigeria Data Protection Act (NDPA) 2023 |
| South Africa | Protection of Personal Information Act (POPIA) |
| Kenya | Data Protection Act 2019 |
| Uganda | Data Protection and Privacy Act 2019 |
| Ethiopia | Computer Crime Proclamation + draft Personal Data Protection Proclamation |
| EU/Global | GDPR (for any EU data subjects) |

---

## 3. Data Categories & Retention

| Category | Storage | Retention | Encryption |
|---|---|---|---|
| User queries (anonymised) | Secure server | 30 days | AES-256 at rest |
| Feedback ratings | Secure server | 12 months | AES-256 at rest |
| Flagged answer logs | Governance store | Indefinite (anonymised) | AES-256 at rest |
| Training data | Object store (S3-compatible) | Indefinite | AES-256 at rest |
| Model checkpoints | Object store | 24 months | AES-256 at rest |
| Meeting minutes | Governance store | Indefinite | AES-256 at rest |

**Raw user input is never retained beyond 30 days unless explicit, informed consent is obtained.**

---

## 4. PII Handling

- No personally identifiable information (names, phone numbers, locations, health record IDs) may be stored in logs, training data, or model outputs.
- All IP addresses are hashed (SHA-256) before logging; original IPs are never stored.
- Any PII discovered in training data is quarantined and flagged in `cleaning_log.csv`.
- Users have the right to request deletion of their interaction data within the 30-day retention window.

---

## 5. Data Subject Rights

Users of the service have the right to:
- **Access** — know what data is held about them
- **Correction** — correct inaccurate data
- **Erasure** — request deletion within retention period
- **Portability** — receive their data in a machine-readable format
- **Objection** — opt out of any data processing

Requests must be submitted to the Data Privacy Officer and responded to within **30 days**.

---

## 6. Third-Party Data Sharing

Training data derived from external sources (WHO, MedQuAD, AmHQA) is used in accordance with their respective licences. No user interaction data is shared with third parties except:
- As required by law (with appropriate legal process)
- With the Oversight Committee for safety review (anonymised only)
- With Zindi for competition submission (predictions only, no PII)

---

## 7. Security Controls

- **Encryption at rest:** AES-256 for all stored data
- **Encryption in transit:** TLS 1.3 for all API communications
- **Access control:** Role-based; only Committee members access flagged logs
- **Audit logging:** All data access is logged
- **Incident response:** Security breaches notified to affected users and relevant DPA within 72 hours

---

## 8. Open Data Commitments

In line with the project's Open Science principles:
- Cleaned and augmented training datasets are published (PII-free rows only)
- Model weights are published on HuggingFace Hub (Apache 2.0)
- Evaluation results and model cards are published on GitHub
- No proprietary or export-controlled data is used in training

---

*This policy is reviewed annually by the Oversight Committee and the Data Privacy Officer.*
