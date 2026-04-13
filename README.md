Multi-Agent Onboarding Coordinator
A two-agent system that generates personalized 30-day onboarding plans from a new hire brief. Agent 1 profiles the new hire from unstructured input. Agent 2 consumes that profile plus accumulated org context from prior onboardings to generate week-by-week milestones, check-ins, and risk flags.
A coordinator function sequences both agents and manages the handoff via shared JSON state. Gets smarter over time as each new onboarding adds to the org context log.
Setup:
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here
python onboarding_agent.py
