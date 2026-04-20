# Final Design Notes

## Product focus

The final app focuses on two grounded learning workflows:

- **Practice test generation** from the user’s uploaded lecture materials
- **Ask** for grounded lecture Q&A with citations and source inspection

We intentionally removed the study-plan generator and the old practice-template path because both introduced brittle behavior and unclear expectations. The final product is narrower, but much more robust and more honest about what the system can and cannot reliably do.

## 1. Human-in-the-loop and agent oversight

This is the clearest class concept expressed by the final design.

- The practice generator does **not** silently run on ambiguous topics. If the requested topic is weakly grounded, the system pauses and asks the user to confirm a narrower topic.
- The clarification UI offers grounded topic suggestions derived from the uploaded lecture evidence, but the user remains the decision-maker.
- Before generation, the app asks the user to confirm the exact request: topic, format, question count, difficulty, coverage, grounding mode, answer key, and rubric settings.
- After a draft is created, the user can **lock** good questions and selectively **regenerate** weak or redundant ones instead of replacing the whole artifact.

Why this matters:

- This treats the model as a bounded assistant, not an autonomous authority.
- It creates meaningful human control at the moments where model errors are most likely: ambiguous scope, weak grounding, and iterative revision.
- It follows the AI agents theme from class by keeping humans in charge of task framing, approval, and correction.

## 2. Transparency and interpretability

The app is designed so the user can see **why** the system produced an answer or question set.

- Every grounded question and chat answer keeps visible citations that open the exact lecture slide or page in the source viewer.
- The practice artifact includes a **human-loop summary** describing which inputs were actually used to produce the current draft.
- Coverage notes are shown directly in the UI rather than hidden in logs or backend metadata.
- Weak or uncovered areas are marked explicitly instead of being blended into a confident-looking output.
- The app surfaces service-health state and disables actions when the learning or content subsystem is unavailable, so failures are understandable rather than mysterious.

Why this matters:

- In class, we discussed that transparency for users is different from developer-facing explainability. This app prioritizes **user-relevant transparency**: what evidence was used, what settings were used, and where uncertainty still exists.
- This helps users inspect, challenge, and correct the system without needing ML expertise.

## 3. Trust and calibrated uncertainty

The design tries to earn trust by being appropriately cautious.

- In **Ask**, weakly matched lecture evidence now causes the system to decline or request clarification more often, rather than forcing an answer.
- If fallback knowledge is allowed, it is labeled as an external supplement instead of being mixed invisibly into the lecture-grounded answer.
- Chat rendering preserves full multi-line responses instead of clipping them into short, overly compressed fragments.
- Practice coverage notes stay visible so the user can see when a test is well grounded and when it is thinner than desired.

Why this matters:

- Trust in AI systems does not come from confidence alone. It comes from the system being predictably honest about uncertainty, scope, and evidence boundaries.
- This aligns with class discussion around trustworthy AI behavior: refusing gracefully is often better UX than confidently hallucinating.

## 4. Auditing AI systems

The app includes lightweight but useful auditability.

- Artifact history preserves prior practice sets and conversations so users can compare versions over time.
- Selective revision plus question locking makes it easier to see what changed and what stayed fixed.
- Source citations create a concrete trace from output back to input evidence.
- Weak-grounding clarifications reveal failure modes during use rather than hiding them.

Why this matters:

- Auditing is not only a back-office activity. In practice, end users also need ways to inspect whether a system behaved responsibly in context.
- This app supports both developer debugging and user accountability by making failure states visible and reviewable.

## 5. Fairness in the field

Formal fairness metrics are not the primary tool here because this is a single-user educational assistant, not a classifier allocating opportunities across groups. But fairness in the field still matters.

- The app avoids pretending that personalization is deeper than it is. It only uses user-visible, editable inputs plus uploaded course materials.
- It does not infer hidden student traits from chat history or opaque profiling.
- It makes the basis of adaptation visible in the interface, so users can tell whether the system is reacting to what they actually asked for.
- Human approval before generation reduces the chance that a vague request gets turned into a misleading or low-value test.

Why this matters:

- In real use, people often judge fairness based on whether a system feels legible, contestable, and respectful.
- This design addresses that perception gap by making adaptation explicit and revisable instead of silently inferred.

## 6. AI literacy

The app is also a small AI-literacy tool.

- It teaches users that AI outputs should be inspected against evidence, not accepted as magic.
- The source viewer makes “grounded answer” concrete by linking answers back to exact lecture material.
- The distinction between grounded content and external supplement helps users learn that AI systems can mix sources unless the interface makes those boundaries explicit.
- Human-in-the-loop controls show that prompting is not enough by itself; careful scoping and review are part of using AI responsibly.

Why this matters:

- AI literacy is a design challenge, not just a knowledge problem.
- The interface helps users develop better habits around verification, scope setting, and skeptical review.

## 7. Responsible AI in practice

Several implementation choices reflect responsible deployment rather than only idealized UX.

- We removed features that looked impressive but were not robust enough for repeated use.
- The app is optimized to work locally with a local database, reducing unnecessary system complexity.
- Material roles are simplified to grounded lecture sources that the user understands and controls.
- The UI avoids pretending that the model has persistent hidden understanding beyond the uploaded materials and explicit settings.

Why this matters:

- Responsible AI practice often means constraining product scope until reliability and integratability are good enough.
- Removing shaky features was a deliberate design choice in favor of correctness, maintainability, and user trust.

## 8. Best-practice UX decisions in the final app

The final UX is strongest in these ways:

- The core workflows are narrow and high-value instead of broad and unreliable.
- Users can inspect evidence directly from both practice questions and chat answers.
- Ambiguity is resolved through confirmation and clarification instead of silent guessing.
- Revision is granular: users can keep strong questions and only regenerate the weak parts.
- Uncertainty and missing coverage are visible, not buried.
- Failure states are communicated clearly through service status, weak-grounding prompts, and actionable warnings.

## 9. Limits and honest framing

The app is still limited by the quality of the uploaded lecture extraction and by lexical grounding behavior. It is not a universal tutor and should not be framed that way. The best experience comes when users upload clear lecture materials, specify a focused topic when needed, and use citations plus revision controls to shape the final artifact.

That limitation is part of the design argument, not a flaw in the writeup: the final app demonstrates that a responsible AI learning tool should narrow scope, preserve user oversight, expose evidence, and refuse or clarify when grounding is weak.
