# Copyright Sierra

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from litellm import completion

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)

MAX_CRITIC_RETRIES = 2
READ_ONLY_PREFIXES = ("get_", "find_", "list_", "search_", "calculate", "think")


# ---------------------------------------------------------------------------
# State models (per-run, never stored on self to stay thread-safe)
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    id: str
    description: str
    status: str  # pending | in_progress | done
    is_verified: bool = False

@dataclass
class Plan:
    goal: str
    steps: List[PlanStep]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [
                {
                    "id": s.id,
                    "description": s.description,
                    "status": s.status,
                    "is_verified": s.is_verified,
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Plan":
        return cls(
            goal=d.get("goal", ""),
            steps=[
                PlanStep(
                    id=s["id"],
                    description=s["description"],
                    status=s.get("status", "pending"),
                    is_verified=s.get("is_verified", False),
                )
                for s in d.get("steps", [])
            ],
        )


@dataclass
class ConversationState:
    executor_messages: List[Dict[str, Any]] = field(default_factory=list)
    approved_plan: Optional[Plan] = None
    pending_plan_update: Optional[Dict[str, Any]] = None
    active_step_id: Optional[str] = None
    instruction_vault: str = ""
    total_cost: float = 0.0
    internal_trace: List[Dict[str, Any]] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0


# ---------------------------------------------------------------------------
# Role prompts
# ---------------------------------------------------------------------------

PLANNER_INSTRUCTION = """<memory>
{instruction_vault}
</memory>

You are a planning and intent-standardization agent for a customer service system.

# Domain Policy
{wiki}

# Available Tools
{tools}

# Current Approved Plan
{current_plan}

# Pending Plan Update (awaiting user confirmation)
{pending_update}

# Conversation So Far
{conversation}

# Instructions
1. Standardize the user's intent into a formal canonical intent label.
2. Infer the required structured fields for that intent (e.g., order_id, item_id).
3. Design or update the plan so that, before any tools are used, there is a
   "Requirement Document" step that explicitly lists and collects all required fields.
4. Update step statuses (pending -> in_progress -> done) as appropriate.

You MUST respond with ONLY valid JSON (no markdown, no extra text) in this exact format:
{{
  "decision": "continue_existing_plan" | "propose_plan_change" | "request_clarification",
  "reason": "Brief explanation of your decision",
  "intent_schema": {{
    "detected_intent": "INTENT_EXCHANGE" | "INTENT_REFUND" | "INTENT_INFORMATION" | null,
    "raw_utterance": "The latest user utterance in natural language",
    "canonical_utterance": "Cleaned, canonical wording of the user's request",
    "required_fields": ["order_id", "item_id"],
    "mappings": [
      {{"raw": "get a new one", "intent": "INTENT_EXCHANGE"}},
      {{"raw": "refund me", "intent": "INTENT_REFUND"}}
    ]
  }},
  "plan": {{
    "goal": "The overall goal based on the user request",
    "steps": [
      {{"id": "s1", "description": "Requirement Document for INTENT_EXCHANGE: list and collect required fields: order_id, item_id.", "status": "pending"}}
    ]
  }},
  "active_step_id": "s1",
  "confirmation_question": null
}}

Rules:
- The "intent_schema" field is REQUIRED and must always be present.
- If you cannot confidently determine the intent, set "detected_intent" to null and
  leave "required_fields" as an empty list.
- If there is no existing plan, create one and use "continue_existing_plan".
- If the user's latest message materially changes the goal, constraints, or approach, \
use "propose_plan_change" and set "confirmation_question" to ask for confirmation.
- The first step in any new plan for a concrete intent should normally be a \
"Requirement Document" step that gathers all required fields.
- Step status updates (pending -> in_progress -> done) do NOT require user confirmation.
- If a pending plan update exists and the user confirmed it, apply the change and return \
"continue_existing_plan" with the updated plan.
- If a pending plan update exists and the user rejected it, return "continue_existing_plan" \
with the original plan unchanged.
- If you need more information to proceed, use "request_clarification" and set \
"confirmation_question" to your question.
- The "plan" field must ALWAYS be present.
"""

EXECUTOR_SYSTEM_TEMPLATE = """<memory>
{instruction_vault}
</memory>

{wiki}

# Current Plan
{plan_summary}

# Active Step
{active_step}

Execute the active step of the plan using the available tools. Follow the domain policy strictly.
If you have enough information to respond to the user, respond directly.
If you need to gather information or perform an action, use the appropriate tool."""

CRITIC_INSTRUCTION = """You are the Policy-Sentinel Reviewer for a customer service system.

Your ONLY job is to find reasons NOT to allow a proposed action. You should be conservative:
if you are uncertain or see any potential violation, you must NOT approve the action.

# Domain Policy
{wiki}

# Current Plan
{plan_summary}

# Active Step
{active_step}

# Proposed Action
Tool: {action_name}
Arguments: {action_args}

# Recent Conversation & Database State
The recent conversation may include tool outputs that reflect the current database state.
Treat these tool outputs as the ground-truth current state of the world.
{recent_context}

# Reviewer Instructions (Policy-Sentinel)
Act as a strict Reviewer whose role is to block unsafe, non-compliant, or premature actions:
1. Compare the proposed action against the Domain Policy. Look for any policy violations,
   missing prerequisites, or unsafe arguments.
2. Compare the proposed action against the current database state as revealed in recent
   tool outputs. Check that identifiers, statuses, and entities referenced by the action
   are valid and consistent with that state.
3. Verify that the action is appropriate for the current plan step and does not skip any
   required “Requirement Document” or data-gathering steps.
4. For user-facing responses, ensure they are accurate, complete, and do not fabricate
   data that has not been observed.
5. Always err on the side of rejection: if there is any doubt, set approved to false.

You MUST respond with ONLY valid JSON (no markdown, no extra text):
{{
  "approved": false,
  "reason": "Detailed explanation of why the action should be blocked or, if truly safe, why it can be allowed",
  "feedback_for_executor": "Concrete Interpreter Feedback that can trigger an Aha Moment and guide the Executor to self-correct.",
  "risk_level": "low" | "medium" | "high"
}}
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MultiAgentV1(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        planner_model: Optional[str] = None,
        planner_provider: Optional[str] = None,
        critic_model: Optional[str] = None,
        critic_provider: Optional[str] = None,
        max_critic_retries: int = MAX_CRITIC_RETRIES,
    ) -> None:
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.planner_model = planner_model or model
        self.planner_provider = planner_provider or provider
        self.critic_model = critic_model or model
        self.critic_provider = critic_provider or provider
        self.max_critic_retries = max_critic_retries
        self._tools_str = json.dumps(tools_info, indent=2)

    # ---- JSON helpers ----

    @staticmethod
    def _parse_json_response(content: str) -> Optional[Dict[str, Any]]:
        """Best-effort JSON extraction: raw → fenced block → first brace pair."""
        if not content:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        start = content.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(content)):
                if content[i] == "{":
                    depth += 1
                elif content[i] == "}":
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(content[start : i + 1])
                    except json.JSONDecodeError:
                        break
        return None

    # ---- Formatting helpers ----

    @staticmethod
    def _format_plan(plan: Optional[Plan]) -> str:
        if plan is None:
            return "No plan yet."
        markers = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]"}
        lines = [f"Goal: {plan.goal}"]
        for s in plan.steps:
            lines.append(f"  {markers.get(s.status, '[ ]')} {s.id}: {s.description}")
        return "\n".join(lines)

    @staticmethod
    def _get_active_step_description(
        plan: Optional[Plan], step_id: Optional[str]
    ) -> str:
        if plan is None or step_id is None:
            return "No active step."
        for s in plan.steps:
            if s.id == step_id:
                return f"{s.id}: {s.description} (status: {s.status})"
        return f"Step {step_id} not found in plan."

    @staticmethod
    def _format_conversation_for_planner(
        messages: List[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            if role == "system":
                continue
            elif role == "tool":
                name = msg.get("name", "tool")
                lines.append(f"[Tool:{name}] {(msg.get('content') or '')[:200]}")
            elif role == "assistant":
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        lines.append(
                            f"[Agent called {fn.get('name', '?')}({fn.get('arguments', '')})]"
                        )
                elif msg.get("content"):
                    lines.append(f"[Agent] {msg['content'][:300]}")
            elif role == "user":
                lines.append(f"[User] {(msg.get('content') or '')[:300]}")
        return "\n".join(lines) if lines else "No conversation yet."

    @staticmethod
    def _get_latest_user_message(state: ConversationState) -> Optional[str]:
        for msg in reversed(state.executor_messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        return None

    @staticmethod
    def _check_discrepancies(state: ConversationState) -> Optional[str]:
        """
        FACT Agent-style check: compare the latest user message against the
        instruction_vault to detect unexplained contradictions (e.g., changing
        the number of items without explanation).

        Returns a Mandatory Clarification question string if a discrepancy is
        detected; otherwise returns None.
        """
        vault = state.instruction_vault or ""
        latest_user = MultiAgentV1._get_latest_user_message(state) or ""
        if not vault or not latest_user:
            return None

        # Simple heuristic: detect mismatched numeric quantities without
        # obvious "change/explanation" language in the latest user message.
        vault_nums = re.findall(r"\d+", vault)
        user_nums = re.findall(r"\d+", latest_user)
        if not vault_nums or not user_nums:
            return None

        if set(vault_nums) == set(user_nums):
            return None

        explanation_markers = [
            "change",
            "changed",
            "update",
            "updated",
            "instead",
            "different",
            "correction",
            "correct",
            "actually",
            "now",
            "revision",
            "revised",
            "modify",
            "modified",
            "adjust",
            "adjusted",
        ]
        lower_msg = latest_user.lower()
        if any(marker in lower_msg for marker in explanation_markers):
            return None

        return (
            "Mandatory Clarification: your latest message appears to contradict your "
            "original instructions (for example, the number of items or quantities "
            "has changed without explanation). Before I call any tools or change the "
            "system state, could you clarify which version is correct and why it "
            "changed?"
        )

    @staticmethod
    def _all_milestones_verified(plan: Optional[Plan]) -> bool:
        """
        Cognitive planning-style check: ensure that all milestones (plan steps) have
        been explicitly marked as verified before allowing the conversation to truly
        terminate.

        If there is no plan or no steps, this returns True.
        """
        if plan is None or not plan.steps:
            return True
        return all(step.is_verified for step in plan.steps)

    @staticmethod
    def _inject_requirement_doc_step(
        plan_data: Dict[str, Any], intent_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ensure that the plan contains a 'Requirement Document' step when a specific intent
        has been identified. This step should explicitly list the required fields inferred
        from the intent_schema before any tools are used.
        """
        detected_intent = intent_schema.get("detected_intent")
        if not detected_intent:
            return plan_data

        required_fields: List[str] = intent_schema.get("required_fields") or []
        steps: List[Dict[str, Any]] = plan_data.get("steps") or []

        # Check if a requirement document step already exists
        for step in steps:
            desc = (step.get("description") or "").lower()
            if "requirement document" in desc:
                return plan_data

        # Generate a non-colliding step id
        existing_ids = {str(s.get("id", "")) for s in steps}
        base_id = "req_1"
        if base_id not in existing_ids:
            new_id = base_id
        else:
            i = 2
            while f"req_{i}" in existing_ids:
                i += 1
            new_id = f"req_{i}"

        if required_fields:
            fields_str = ", ".join(required_fields)
            desc = (
                f"Requirement Document for {detected_intent}: list and collect "
                f"required fields: {fields_str}."
            )
        else:
            desc = (
                f"Requirement Document for {detected_intent}: identify and collect all "
                f"required structured fields for this intent (e.g., order_id, item_id)."
            )

        new_step = {"id": new_id, "description": desc, "status": "pending"}
        plan_data = dict(plan_data)
        plan_data["steps"] = [new_step] + steps
        return plan_data

    # ---- LLM role callers ----

    def _call_planner(self, state: ConversationState) -> Dict[str, Any]:
        pending_str = (
            json.dumps(state.pending_plan_update, indent=2)
            if state.pending_plan_update
            else "None"
        )
        prompt = PLANNER_INSTRUCTION.format(
            instruction_vault=state.instruction_vault,
            wiki=self.wiki,
            tools=self._tools_str,
            current_plan=self._format_plan(state.approved_plan),
            pending_update=pending_str,
            conversation=self._format_conversation_for_planner(
                state.executor_messages
            ),
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "Analyze the conversation and return your planning decision as JSON.",
            },
        ]
        res = completion(
            model=self.planner_model,
            custom_llm_provider=self.planner_provider,
            messages=messages,
            temperature=self.temperature,
        )
        cost = res._hidden_params.get("response_cost", 0) or 0
        state.total_cost += cost

        content = res.choices[0].message.content or ""
        parsed = self._parse_json_response(content)
        state.internal_trace.append(
            {"role": "planner", "raw_output": content, "parsed": parsed, "cost": cost}
        )

        if parsed is None:
            return {
                "decision": "continue_existing_plan",
                "reason": "Failed to parse planner output, continuing.",
                "intent_schema": {
                    "detected_intent": None,
                    "raw_utterance": "",
                    "canonical_utterance": "",
                    "required_fields": [],
                    "mappings": [],
                },
                "plan": (
                    state.approved_plan.to_dict()
                    if state.approved_plan
                    else {"goal": "Help the user", "steps": []}
                ),
                "active_step_id": state.active_step_id,
                "confirmation_question": None,
            }

        # Post-process the plan using the intent_schema to ensure a Requirement Document step.
        intent_schema = parsed.get("intent_schema") or {}
        plan_data = parsed.get("plan") or {}
        if isinstance(plan_data, dict):
            plan_data = self._inject_requirement_doc_step(plan_data, intent_schema)
            parsed["plan"] = plan_data

        return parsed

    def _call_executor(
        self,
        state: ConversationState,
        extra_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Action, float]:
        system_prompt = EXECUTOR_SYSTEM_TEMPLATE.format(
            instruction_vault=state.instruction_vault,
            wiki=self.wiki,
            plan_summary=self._format_plan(state.approved_plan),
            active_step=self._get_active_step_description(
                state.approved_plan, state.active_step_id
            ),
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(state.executor_messages[1:])
        if extra_messages:
            messages.extend(extra_messages)

        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            tools=self.tools_info,
            temperature=self.temperature,
        )
        cost = res._hidden_params.get("response_cost", 0) or 0
        msg = res.choices[0].message.model_dump()
        action = self._message_to_action(msg)

        state.internal_trace.append(
            {
                "role": "executor",
                "action": {"name": action.name, "kwargs": action.kwargs},
                "cost": cost,
            }
        )
        return msg, action, cost

    def _call_critic(
        self, state: ConversationState, action: Action
    ) -> Dict[str, Any]:
        recent = state.executor_messages[-6:]
        prompt = CRITIC_INSTRUCTION.format(
            wiki=self.wiki,
            plan_summary=self._format_plan(state.approved_plan),
            active_step=self._get_active_step_description(
                state.approved_plan, state.active_step_id
            ),
            action_name=action.name,
            action_args=json.dumps(action.kwargs, indent=2),
            recent_context=self._format_conversation_for_planner(recent),
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "Evaluate the proposed action and return your assessment as JSON.",
            },
        ]
        res = completion(
            model=self.critic_model,
            custom_llm_provider=self.critic_provider,
            messages=messages,
            temperature=self.temperature,
        )
        cost = res._hidden_params.get("response_cost", 0) or 0
        state.total_cost += cost

        content = res.choices[0].message.content or ""
        parsed = self._parse_json_response(content)
        state.internal_trace.append(
            {
                "role": "critic",
                "action_reviewed": {"name": action.name, "kwargs": action.kwargs},
                "raw_output": content,
                "parsed": parsed,
                "cost": cost,
            }
        )
        if parsed is None:
            return {
                "approved": False,
                "reason": "Policy-Sentinel output unparseable; conservatively blocking the action.",
                "feedback_for_executor": (
                    "Interpreter Feedback: I could not reliably verify your proposed action "
                    "against the domain policy and current database state. Please re-check "
                    "your assumptions, required fields, and any referenced entities before "
                    "trying a different approach."
                ),
                "risk_level": "medium",
            }
        return parsed

    # ---- Heuristic gate ----

    @staticmethod
    def _is_read_only_tool(action_name: str) -> bool:
        name_lower = action_name.lower()
        return any(name_lower.startswith(p) for p in READ_ONLY_PREFIXES)

    @staticmethod
    def _requires_critique(action: Action) -> bool:
        if action.name == RESPOND_ACTION_NAME:
            return True
        return not MultiAgentV1._is_read_only_tool(action.name)

    @staticmethod
    def _message_to_action(message: Dict[str, Any]) -> Action:
        if (
            message.get("tool_calls")
            and len(message["tool_calls"]) > 0
            and message["tool_calls"][0].get("function") is not None
        ):
            tc = message["tool_calls"][0]
            return Action(
                name=tc["function"]["name"],
                kwargs=json.loads(tc["function"]["arguments"]),
            )
        return Action(
            name=RESPOND_ACTION_NAME,
            kwargs={RESPOND_ACTION_FIELD_NAME: message.get("content", "")},
        )

    # ---- Outcome-Driven Validation helpers ----

    def _select_audit_tool(self, write_tool_name: str) -> Optional[str]:
        """
        Pick a read-only tool to use for Environment Audit after a write action.
        For now, we simply choose the first tool whose name matches the
        READ_ONLY_PREFIXES heuristic.
        """
        for t in self.tools_info:
            fn = t.get("function", {})
            name = fn.get("name")
            if isinstance(name, str) and self._is_read_only_tool(name):
                return name
        return None

    # ---- Orchestrated solve loop ----

    def solve(
        self,
        env: Env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        reset_resp = env.reset(task_index=task_index)
        state = ConversationState()
        state.instruction_vault = reset_resp.observation
        state.info = (
            reset_resp.info.model_dump()
            if hasattr(reset_resp.info, "model_dump")
            else {}
        )
        state.executor_messages = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": reset_resp.observation},
        ]

        last_source = "user"

        for _ in range(max_num_steps):
            # ---------- PLANNER PHASE (runs on every user message) ----------
            if last_source == "user":
                planner_result = self._call_planner(state)
                decision = planner_result.get("decision", "continue_existing_plan")
                plan_data = planner_result.get("plan")

                if decision == "propose_plan_change":
                    state.pending_plan_update = {
                        "proposed_plan": plan_data,
                        "reason": planner_result.get("reason", ""),
                    }
                    question = planner_result.get(
                        "confirmation_question",
                        "Could you confirm you'd like me to proceed with this updated plan?",
                    )
                    env_resp = env.step(
                        Action(
                            name=RESPOND_ACTION_NAME,
                            kwargs={RESPOND_ACTION_FIELD_NAME: question},
                        )
                    )
                    state.reward = env_resp.reward
                    state.info = {**state.info, **env_resp.info.model_dump()}
                    state.executor_messages.extend(
                        [
                            {"role": "assistant", "content": question},
                            {"role": "user", "content": env_resp.observation},
                        ]
                    )
                    last_source = "user"
                    if env_resp.done:
                        break
                    continue

                if decision == "request_clarification":
                    question = planner_result.get(
                        "confirmation_question",
                        "Could you provide more details about your request?",
                    )
                    env_resp = env.step(
                        Action(
                            name=RESPOND_ACTION_NAME,
                            kwargs={RESPOND_ACTION_FIELD_NAME: question},
                        )
                    )
                    state.reward = env_resp.reward
                    state.info = {**state.info, **env_resp.info.model_dump()}
                    state.executor_messages.extend(
                        [
                            {"role": "assistant", "content": question},
                            {"role": "user", "content": env_resp.observation},
                        ]
                    )
                    last_source = "user"
                    if env_resp.done:
                        break
                    continue

                # continue_existing_plan
                if plan_data and isinstance(plan_data, dict):
                    state.approved_plan = Plan.from_dict(plan_data)
                state.pending_plan_update = None
                state.active_step_id = planner_result.get(
                    "active_step_id", state.active_step_id
                )

            # ---------- FACT Agent discrepancy gate ----------
            # Before allowing any tool calls, check for contradictions between the latest
            # user message and the instruction_vault. If found, force a Mandatory
            # Clarification question and skip the executor/tool phase for this turn.
            mandatory_clarification = self._check_discrepancies(state)
            if mandatory_clarification:
                env_resp = env.step(
                    Action(
                        name=RESPOND_ACTION_NAME,
                        kwargs={RESPOND_ACTION_FIELD_NAME: mandatory_clarification},
                    )
                )
                state.reward = env_resp.reward
                state.info = {**state.info, **env_resp.info.model_dump()}
                state.executor_messages.extend(
                    [
                        {"role": "assistant", "content": mandatory_clarification},
                        {"role": "user", "content": env_resp.observation},
                    ]
                )
                last_source = "user"
                if env_resp.done:
                    break
                continue

            # ---------- EXECUTOR + CRITIC PHASE ----------
            action: Optional[Action] = None
            executor_msg: Optional[Dict[str, Any]] = None
            retry_context: List[Dict[str, Any]] = []

            last_critic_result: Optional[Dict[str, Any]] = None
            for attempt in range(self.max_critic_retries + 1):
                executor_msg, action, cost = self._call_executor(
                    state, extra_messages=retry_context or None
                )
                state.total_cost += cost

                if self._requires_critique(action):
                    critic_result = self._call_critic(state, action)
                    last_critic_result = critic_result
                    if critic_result.get("approved", False):
                        break
                    feedback = critic_result.get(
                        "feedback_for_executor",
                        "Interpreter Feedback: Your proposed action appears to conflict with the domain policy or current database state.",
                    )
                    retry_context.append(
                        {
                            "role": "user",
                            "content": (
                                "[Interpreter Feedback] Policy-Sentinel Reviewer has blocked "
                                "your previous action. Use this feedback to trigger an Aha Moment "
                                "and self-correct your plan:\n"
                                f"{feedback}"
                            ),
                        }
                    )
                else:
                    break
            else:
                action = Action(
                    name=RESPOND_ACTION_NAME,
                    kwargs={
                        RESPOND_ACTION_FIELD_NAME: (
                            "I want to make sure I handle your request correctly. "
                            "Could you please clarify or confirm what you'd like me to do?"
                        )
                    },
                )
                executor_msg = {
                    "role": "assistant",
                    "content": action.kwargs[RESPOND_ACTION_FIELD_NAME],
                }

            # ---------- EXECUTE ACTION ----------
            assert action is not None and executor_msg is not None
            env_resp = env.step(action)
            state.reward = env_resp.reward
            state.info = {**state.info, **env_resp.info.model_dump()}

            if action.name != RESPOND_ACTION_NAME:
                if executor_msg.get("tool_calls"):
                    executor_msg["tool_calls"] = executor_msg["tool_calls"][:1]
                    state.executor_messages.extend(
                        [
                            executor_msg,
                            {
                                "role": "tool",
                                "tool_call_id": executor_msg["tool_calls"][0]["id"],
                                "name": executor_msg["tool_calls"][0]["function"][
                                    "name"
                                ],
                                "content": env_resp.observation,
                            },
                        ]
                    )
                else:
                    state.executor_messages.append(
                        {
                            "role": "user",
                            "content": f"API output: {env_resp.observation}",
                        }
                    )

                # Outcome-Driven Validation (ReTool): if this was a write tool
                # (i.e., not read-only), immediately perform an Environment Audit
                # via a read tool to verify the effect on the database.
                if not self._is_read_only_tool(action.name):
                    audit_tool_name = self._select_audit_tool(action.name)
                    if audit_tool_name is not None:
                        audit_action = Action(
                            name=audit_tool_name,
                            kwargs=action.kwargs,
                        )
                        audit_resp = env.step(audit_action)
                        # Do not override main reward/info; this is an auxiliary check.
                        state.executor_messages.extend(
                            [
                                {
                                    "role": "tool",
                                    "tool_call_id": f"audit_{audit_tool_name}",
                                    "name": audit_tool_name,
                                    "content": audit_resp.observation,
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        "[System Feedback] Environment Audit after your "
                                        "last write action returned:\n"
                                        f"{audit_resp.observation}\n"
                                        "Compare this audited state with what you claimed "
                                        "to have done. If they differ, treat this as an "
                                        "Aha Moment and self-correct your plan or actions."
                                    ),
                                },
                            ]
                        )

                last_source = "tool"
            else:
                content = action.kwargs.get(RESPOND_ACTION_FIELD_NAME, "")
                state.executor_messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": env_resp.observation},
                    ]
                )
                last_source = "user"

                # Mark active step done after responding to user and record verification
                if state.approved_plan and state.active_step_id:
                    for s in state.approved_plan.steps:
                        if s.id == state.active_step_id and s.status == "in_progress":
                            s.status = "done"
                            # Milestone-based KPI tracking: mark as verified only if the
                            # Policy-Sentinel approved the final response for this step.
                            if last_critic_result is not None:
                                s.is_verified = bool(
                                    last_critic_result.get("approved", False)
                                )
                            break

            if env_resp.done:
                # Cognitive planning / Plangen-style guard: do not allow the agent to
                # truly terminate the conversation until all milestones are verified.
                if self._all_milestones_verified(state.approved_plan):
                    break
                # Force a re-plan on the next iteration by treating this as coming from
                # the user again; the planner will see that not all milestones are
                # verified and can adjust the plan accordingly.
                last_source = "user"

        return SolveResult(
            reward=state.reward,
            messages=state.executor_messages,
            info={**state.info, "multi_agent_trace": state.internal_trace},
            total_cost=state.total_cost,
        )
