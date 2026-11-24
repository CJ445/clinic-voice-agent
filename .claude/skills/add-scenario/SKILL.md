---
name: add-scenario
description: How to write or generate caller-simulator persona scenarios for this repo. Use when adding, editing, or generating scenarios under sim/personas.
---
# Writing simulator scenarios

Each scenario is YAML in `sim/personas/<category>/<id>.yaml`:

```yaml
id: resched-hinglish-impatient-014
category: reschedule            # book | reschedule | cancel | info | out_of_scope
language: hinglish              # hi | en | hinglish
behaviours: [impatient, interrupts]
interrupt_plan: [{after_agent_ms: 800, turn: 2}]
date_ambiguity: relative        # none | relative | vague | conflicting
opening: "Haan hello, mera Tuesday wala appointment shift karna hai"
goal: "Move the existing Tuesday 10:00 Dr. Rao appointment to next Friday evening"
hidden_facts: {patient_name: "Anjali Mehta", phone: "98xxxxxx12"}
expected:
  db_end_state:
    - {patient: "Anjali Mehta", doctor: "Dr. Rao", slot: "2026-12-04T18:00+05:30", status: booked}
    - {patient: "Anjali Mehta", doctor: "Dr. Rao", slot: "2026-12-01T10:00+05:30", status: cancelled}
  tool_calls: [check_slots, cancel, book]   # order-insensitive set unless `ordered: true`
  must_not: [book_without_confirmation, offer_unavailable_slot]
```

## Rules
- Dates are relative to the frozen sim clock in `sim/clock.py`, never "today".
- Every scenario must be solvable with the seed DB. Run `python -m sim.validate` after editing.
- Out-of-scope scenarios (medical advice, emergencies) expect a refusal + redirect, zero mutating tool calls.
- Emergency utterances ("chest pain") must expect the emergency-redirect script. Never relax this.
- When generating in bulk with an LLM, Cyril spot-checks ≥ 15%; record checked IDs in `sim/personas/REVIEWED.txt`.
- Balance: ~30% Hinglish, ~30% Hindi, ~40% English; ≥ 20% with interruptions; ≥ 25% ambiguous dates.
