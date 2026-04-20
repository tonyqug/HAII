# Final Design Notes

## Product focus

The final system centers on two grounded learning workflows:

- `Study Planner`: produces a sequential, citation-backed checklist tailored to the user's visible planning inputs and uploaded course materials.
- `Ask`: answers questions only when the lecture evidence is strong enough, otherwise it asks for clarification or clearly labels any fallback as external background.

The separate practice-test feature was removed so the system can concentrate effort, interface space, and reliability on the two flows that matter most.

## Class concepts reflected in the design

### Human-in-the-loop and agent oversight

- The study planner now pauses when a user names a topic that does not match strong lecture evidence, instead of confidently generating a misleading plan.
- The ask flow now prefers clarification when grounding is weak, rather than pretending the answer is lecture-grounded.
- This treats the model as a bounded assistant that should defer when the user needs to steer it, which aligns with the course discussion of agent oversight and meaningful human control.

### Transparency and interpretability

- Study plans explicitly show which inputs were used, which optional inputs were missing, and which source materials/slides grounded the plan.
- Chat responses keep support-status labels and visible citations so users can inspect the exact lecture evidence.
- Transparency is aimed at the user-facing question "why did the system do this?" rather than trying to expose internal model weights or pseudo-explanations.

### Trust and AI literacy

- The system now separates grounded lecture claims from external background more clearly.
- Weakly grounded cases are surfaced as uncertainty or clarification requests instead of being hidden behind fluent output.
- This helps students build better mental models of what the system can and cannot know from their uploads, which is a core AI-literacy goal.

### Auditing AI systems

- The shell now preserves clearer artifact history for study plans and conversations while pruning stale remote references that caused broken local state.
- Grounding evidence, uncertainty notes, and source links create a lightweight audit trail for whether an answer or plan was actually supported.
- This is a practical auditability layer: it helps users and developers inspect failures after the fact.

### Responsible AI in practice

- The implementation removes an underperforming feature instead of leaving it in the product surface.
- The remaining workflows were made more conservative, more debuggable, and easier to verify locally.
- This reflects the course theme that responsible AI is often about product scoping, operational safeguards, and maintenance choices, not just better prompts.

### Fairness and fairness in the field

- Formal fairness metrics are less central in this single-user study setting, but perceived fairness still matters.
- A system that appears personalized while actually ignoring the user's inputs is unfair in the field because it misrepresents whose needs shaped the output.
- Tailoring transparency and clarification behavior reduce that mismatch by making the basis of personalization visible.

## Key implementation choices

- Removed the practice-generation UI and public API routes.
- Study plans now avoid filler padding and reduce repeated checklist language.
- Topic mismatches trigger clarification instead of low-quality "tailored" plans.
- Chat evidence selection is stricter and deduplicated by grounded slide evidence.
- Chat rendering preserves full multi-line responses instead of visually collapsing long outputs.
- Workspace hydration now prunes stale remote study-plan and conversation references, which fixes a class of `404` and "artifact not found" failures.
