"""
Multi-Agent Onboarding Coordinator
------------------------------------
Two specialized agents share state to onboard new team members:

  Agent 1 - Intake Profiler   : reads a new hire brief and extracts a
                                 structured profile (role, skills, gaps,
                                 working style signals)

  Agent 2 - Plan Generator    : consumes the profile + shared org context
                                 to produce a personalized 30-day plan
                                 with milestones, check-ins, and resources

Shared state is a JSON object passed between agents so each one builds
on the last. A coordinator function sequences the agents and manages
the handoff. Output is saved to a persistent onboarding log.

This demonstrates: multi-agent orchestration, shared persistent state,
role-specialized prompting, and compounding context across agent calls.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import anthropic

MODEL = "claude-opus-4-5"
LOG_FILE = Path("onboarding_log.json")
CLIENT = anthropic.Anthropic()


# ── Shared state helpers ──────────────────────────────────────────────────────

def load_log() -> list:
    if LOG_FILE.exists():
        return json.loads(LOG_FILE.read_text())
    return []


def save_log(log: list):
    LOG_FILE.write_text(json.dumps(log, indent=2))


def build_org_context(log: list) -> str:
    """Pull patterns from past onboardings to inform the current one."""
    if not log:
        return "No prior onboarding history available."
    roles = [e.get("profile", {}).get("role", "unknown") for e in log[-5:]]
    common_gaps = {}
    for e in log:
        for gap in e.get("profile", {}).get("skill_gaps", []):
            common_gaps[gap] = common_gaps.get(gap, 0) + 1
    top_gaps = sorted(common_gaps.items(), key=lambda x: -x[1])[:3]
    summary = f"Recent hires: {', '.join(roles)}."
    if top_gaps:
        summary += f" Recurring skill gaps across the team: {', '.join(g for g, _ in top_gaps)}."
    return summary


# ── Agent 1: Intake Profiler ──────────────────────────────────────────────────

def run_intake_agent(brief: str) -> dict:
    """
    Reads a free-form new hire brief and extracts a structured profile.
    Returns a dict with role, background, skill_gaps, working_style, priorities.
    """
    system = """You are an intake profiling agent. Read a new hire brief and extract a structured profile.
Return ONLY valid JSON with no preamble or markdown fences:
{
  "name": "person's name or 'unknown'",
  "role": "their role or title",
  "background": "2 sentence summary of their background",
  "strengths": ["list of 2-4 clear strengths"],
  "skill_gaps": ["list of 2-4 areas they will need support in"],
  "working_style": "1 sentence on how they prefer to work based on any signals in the brief",
  "first_week_priority": "the single most important thing for them in week 1"
}"""

    resp = CLIENT.messages.create(
        model=MODEL,
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": f"New hire brief:\n\n{brief}"}]
    )
    raw = resp.content[0].text.strip().strip("```json").strip("```").strip()
    return json.loads(raw)


# ── Agent 2: Plan Generator ───────────────────────────────────────────────────

def run_plan_agent(profile: dict, org_context: str) -> dict:
    """
    Takes the structured profile from Agent 1 plus org context and generates
    a personalized 30-day onboarding plan.
    """
    system = """You are an onboarding plan generator. Using a new hire profile and org context,
produce a personalized 30-day onboarding plan.
Return ONLY valid JSON with no preamble or markdown fences:
{
  "overview": "2 sentence summary of the onboarding approach for this person",
  "week_1": {
    "theme": "short theme title",
    "goals": ["2-3 specific goals"],
    "tasks": ["3-4 concrete tasks"],
    "check_in": "what to cover in the week 1 check-in"
  },
  "week_2_3": {
    "theme": "short theme title",
    "goals": ["2-3 specific goals"],
    "tasks": ["3-4 concrete tasks"],
    "check_in": "what to cover in the week 2-3 check-in"
  },
  "week_4": {
    "theme": "short theme title",
    "goals": ["2-3 specific goals"],
    "tasks": ["3-4 concrete tasks"],
    "check_in": "what to cover at the 30-day review"
  },
  "resources": ["3-5 specific resources or tools to point them to"],
  "risk_flags": ["1-2 things to watch for given their gaps"]
}"""

    user_prompt = f"""New hire profile:
{json.dumps(profile, indent=2)}

Org context from prior onboardings:
{org_context}

Generate their personalized 30-day plan."""

    resp = CLIENT.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user_prompt}]
    )
    raw = resp.content[0].text.strip().strip("```json").strip("```").strip()
    return json.loads(raw)


# ── Coordinator ───────────────────────────────────────────────────────────────

def run_coordinator(brief: str) -> dict:
    """
    Sequences Agent 1 and Agent 2, passing shared state between them.
    Returns the combined output and persists to log.
    """
    log = load_log()
    org_context = build_org_context(log)

    print("  [Agent 1] Profiling new hire...", flush=True)
    profile = run_intake_agent(brief)

    print("  [Agent 2] Generating 30-day plan...", flush=True)
    plan = run_plan_agent(profile, org_context)

    result = {
        "timestamp": datetime.now().isoformat(timespec="minutes"),
        "profile": profile,
        "plan": plan
    }
    log.append(result)
    save_log(log)
    return result


# ── Display ───────────────────────────────────────────────────────────────────

def display(result: dict):
    p = result["profile"]
    pl = result["plan"]

    print("\n" + "=" * 60)
    print(f"  ONBOARDING PLAN: {p.get('name', 'New Hire').upper()}")
    print(f"  Role: {p.get('role', 'Unknown')}")
    print("=" * 60)

    print(f"\n  Profile Overview")
    print(f"  Background : {p.get('background', '')}")
    print(f"  Strengths  : {', '.join(p.get('strengths', []))}")
    print(f"  Gaps       : {', '.join(p.get('skill_gaps', []))}")
    print(f"  Style      : {p.get('working_style', '')}")
    print(f"  Week 1 focus: {p.get('first_week_priority', '')}")

    print(f"\n  Plan Overview")
    print(f"  {pl.get('overview', '')}")

    for week_key, label in [("week_1", "Week 1"), ("week_2_3", "Weeks 2-3"), ("week_4", "Week 4 / 30-Day Review")]:
        w = pl.get(week_key, {})
        print(f"\n  {label}: {w.get('theme', '')}")
        for goal in w.get("goals", []):
            print(f"    - {goal}")
        print(f"  Check-in: {w.get('check_in', '')}")

    if pl.get("risk_flags"):
        print(f"\n  Watch for:")
        for r in pl["risk_flags"]:
            print(f"  ! {r}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "=" * 60)
    print("  MULTI-AGENT ONBOARDING COORDINATOR")
    print("  Paste a new hire brief, then type END on its own line")
    print("  Type 'quit' to exit")
    print("=" * 60 + "\n")

    while True:
        print("New hire brief:")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip().lower() == "quit":
                print("Done.\n")
                return
            if line.strip() == "END":
                break
            lines.append(line)

        brief = "\n".join(lines).strip()
        if not brief:
            print("No input. Try again.\n")
            continue

        print("\n  Running agents...")
        try:
            result = run_coordinator(brief)
            display(result)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Error processing output: {e}\n")


if __name__ == "__main__":
    run()
