"""Refinement prompts for Architect and Engineer."""

ARCHITECT_REFINEMENT_PROMPT = """You previously proposed an architecture for the following question.
You have now received feedback from The Engineer. Based on this critique, improve your proposal.

## Original Question
{user_question}

## System Context
{system_context}

## Your Original Proposal
Title: {my_title}
Summary: {my_summary}
Approach: {my_approach}

Components:
{my_components}

Trade-offs:
{my_trade_offs}

Risks:
{my_risks}

---

## Critique from The Engineer
Agreement Level: {critique_agreement:.0%}
Strengths: {critique_strengths}
Weaknesses: {critique_weaknesses}
Concerns: {critique_concerns}
Suggestions: {critique_suggestions}

---

## Your Task
Refine your proposal to address the critique while maintaining your architectural vision.
- Address valid concerns raised by The Engineer
- Incorporate practical suggestions where they improve the design
- Defend or adjust trade-offs based on the feedback
- Maintain your confidence in areas where you disagree with the critique

Provide an IMPROVED proposal that synthesizes the best of both perspectives.
"""

ENGINEER_REFINEMENT_PROMPT = """You previously proposed an implementation plan for the following question.
You have now received feedback from The Architect. Based on this critique, improve your proposal.

## Original Question
{user_question}

## System Context
{system_context}

## Your Original Proposal
Title: {my_title}
Summary: {my_summary}
Approach: {my_approach}

Components:
{my_components}

Trade-offs:
{my_trade_offs}

Risks:
{my_risks}

---

## Critique from The Architect
Agreement Level: {critique_agreement:.0%}
Strengths: {critique_strengths}
Weaknesses: {critique_weaknesses}
Concerns: {critique_concerns}
Suggestions: {critique_suggestions}

---

## Your Task
Refine your proposal to address the critique while maintaining practical implementation focus.
- Address valid architectural concerns raised by The Architect
- Incorporate design suggestions where they improve implementation
- Defend or adjust trade-offs based on the feedback
- Maintain your confidence in areas where you disagree with the critique

Provide an IMPROVED proposal that synthesizes the best of both perspectives.
"""
