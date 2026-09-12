# Trackshift VMAX — event workspace

This branch is a fresh implementation workspace. It currently contains documentation only: no imported model, trained weights, application, saved predictions, demo launcher or import workflow.

## Prior work disclosure

A pre-existing VMAX prototype was copied into this repository at commit 3e3cd935a13b704dc42a27b4ef7f50a4dd61ff1d. Its model training and evaluation were completed on 11 September 2026, before the event. The import did not make that work new.

That snapshot remains at [reference/pre-event-vmax-not-submission](https://github.com/arnavmandre/trackshift_vmax/tree/reference/pre-event-vmax-not-submission) for transparent disclosure and historical reference. The original source is [arnavmandre/vmax](https://github.com/arnavmandre/vmax). Git history has not been rewritten.

The prior checkpoint, code, UI, tests and reported 73.3% precision / 64.7% recall are not claimed as work built during this hackathon. No new event implementation or new event accuracy result is claimed by this documentation.

## Rule basis and limits

The supplied Trackshift AMA FAQ, section 6, permits pre-event research and demo dataset preparation, permits the idea-submission prototype as a reference, and permits either a fresh prototype or enhancements to an existing prototype. Final judging considers work actually built during the event; an identical unchanged prototype is not an eligible new contribution. Section 7 permits open-source and AI tools provided the team understands and can modify the code.

It has not been established that all of the September 11 work was the prototype submitted for Round 1. Its eligibility for reuse needs organiser clarification. Follow any stricter instructions delivered at the event.

Separating branches does not undo pre-event work or provide organiser approval. Do not present this repository's history as exclusively event-built.

## Build from here

Develop new implementation during the official event window from the problem requirements and permitted general references. Record changes and evidence in EVENT_WORK.md. Keep the earlier implementation, weights, results and UI out of this branch unless the organisers explicitly permit their reuse and the disclosure is updated.

Reformatting, renaming, translating, or asking an AI to paraphrase the old code does not establish an independent new implementation.

The official event window in the FAQ is 12 September 12:30 PM to 13 September 12:00 PM local event time; confirm the actual start, deadline and any amendments with organisers.

## Organiser clarification to ask in the event group

We developed a VMAX prototype and trained weights before the event, and initially imported them into this repository. We have now separated them as disclosed prior work and removed them from the active implementation branch without rewriting history. May we use the earlier prototype only as conceptual reference, and are its prepared synthetic datasets allowed? If incremental reuse is permitted, which components may be retained and how should we document the work judged during the event?

No organiser confirmation has been recorded. This question has not been sent automatically.
