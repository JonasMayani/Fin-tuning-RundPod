# Oversight Committee Charter
## Multilingual African Health Assistant

**Version:** 1.0
**Adopted:** 2025
**Review Cycle:** Annual

---

## 1. Purpose

The Oversight Committee (the "Committee") exists to ensure that the Multilingual African Health Assistant is deployed and operated safely, ethically, and in the best interests of the communities it serves. The Committee provides independent governance over model updates, clinical safety, cultural appropriateness, and data governance.

---

## 2. Mandate

The Committee is empowered to:
- Review and approve all major model updates before deployment
- Review flagged and low-rated answers monthly
- Commission red-team and adversarial safety evaluations
- Conduct quarterly cultural appropriateness audits
- Update the Data Governance Policy (see `data_governance_policy.md`)
- Escalate serious safety concerns to Anthropic, WHO, or national health authorities

The Committee operates independently of the engineering team. Its decisions on safety and ethics are binding.

---

## 3. Composition

The Committee shall comprise **six standing members:**

| Role | Responsibilities | Regional Coverage |
|---|---|---|
| Clinical Health Professional #1 | Medical accuracy review, safety flagging | West Africa (Ghana, Nigeria) |
| Clinical Health Professional #2 | Medical accuracy review, safety flagging | East Africa (Kenya, Uganda, Tanzania) / Ethiopia |
| AI Ethics / Responsible AI Specialist | Bias audits, fairness evaluation, AI policy | Any |
| Community Advocate — Akan | Cultural review for Twi/Akan content | Ghana |
| Community Advocate — Amharic/Luganda/Swahili | Cultural review for East African content | Ethiopia / Uganda / East Africa |
| Data Privacy / Legal Expert | NDPA, POPIA, GDPR compliance; data subject rights | Pan-African |

### Additional Advisors (non-voting)
- Representative from Zindi / ITU (during competition period)
- Patient safety officer (on request)
- NGO partner liaisons (on request)

### Quorum
Decisions require at least **4 of 6 members**, including at least **1 clinical professional** and **1 community advocate**.

---

## 4. Responsibilities

### 4.1 Monthly Review
- Review all answers flagged in the past month (rating ≤ 2/5 or safety keyword triggered)
- Categorise flags: clinical inaccuracy / cultural insensitivity / language quality / harmful advice
- Approve corrections; add corrections to next retraining cycle with weight=3.0
- Report: flagged count, category breakdown, resolution status

### 4.2 Quarterly Cultural Appropriateness Audit
- Randomly sample **50 Q&A pairs per language** (250 total)
- Evaluate each on: factual accuracy, cultural fit, safety, fluency (5-point Likert scale)
- Compute Krippendorff's α for inter-rater reliability (target α ≥ 0.60)
- Publish audit summary in `reports/cultural_audit_YYYY_QN.md`

### 4.3 Model Update Sign-Off
Before any major model update is deployed to production, the Committee must:
1. Review evaluation results on validation set (ROUGE, LLM Judge, BERTScore)
2. Review results of safety evaluation against adversarial prompt set (100 prompts)
3. Confirm no regression on flagged answer categories
4. Record sign-off decision in meeting minutes (see Section 7)

**Minor updates** (e.g. config changes, no model weight changes) do not require full sign-off but must be notified to the Committee within 48 hours.

### 4.4 Annual Policy Review
- Review and update Data Governance Policy
- Review community feedback and usage analytics for systemic issues
- Update adversarial test suite with new threat vectors
- Review composition of Committee; recruit replacements for vacated roles

---

## 5. Meeting Schedule

| Cadence | Type | Format |
|---|---|---|
| Monthly | Flagged answer review | Async (shared document) + 60-min call if needed |
| Quarterly | Cultural audit + model review | 2-hour video call |
| Annually | Full policy review | Half-day workshop (in-person preferred) |
| Ad hoc | Safety incident response | Within 48 hours of alert |

Meeting minutes are stored in `governance/meeting_minutes/` and made available to all Committee members.

---

## 6. Escalation Protocol

### Level 1 — Internal Review (Committee)
- Trigger: Single flagged answer or minor inaccuracy
- Action: Correction added to retraining queue; logged

### Level 2 — Deployment Pause
- Trigger: ≥ 5 serious safety flags in 7 days, or ROUGE-L drop > 0.05
- Action: Suspend model updates; emergency Committee meeting within 48h

### Level 3 — External Notification
- Trigger: Evidence of patient harm, serious adverse event, or systematic bias
- Action: Notify national health authority; consider temporary service suspension; engage WHO

### Level 4 — Service Shutdown
- Trigger: Imminent risk of serious patient harm
- Action: Immediate suspension by Clinical Professional members (majority vote); notify all stakeholders

---

## 7. Documentation & Transparency

All Committee activities are documented:
- `governance/flagged_answers_log.csv` — all flagged answer records
- `governance/meeting_minutes/YYYY-MM-DD.md` — all meeting notes
- `governance/model_update_signoffs.csv` — sign-off records
- `reports/cultural_audit_*.md` — quarterly audit reports

In accordance with the project's Open Science commitment, summary findings (without PII) are published annually.

---

## 8. Conflict of Interest

Committee members must disclose any financial or professional relationship with organisations using or competing with this system. Members with conflicts on a specific decision must recuse themselves from that vote.

---

## 9. Amendment Process

This charter may be amended by a **supermajority (5/6)** of Committee members. Proposed amendments must be circulated at least 14 days before the vote. All amendments are versioned and dated.

---

*Adopted by the founding Oversight Committee — 2025*
