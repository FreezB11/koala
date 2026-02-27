"""
orchestrator.py — Multi-agent orchestration with the Contract/Stub pattern

The "Make-Belief" system:
  One agent says: "I have F(a, b) -> x. Here's the interface."
  Another agent uses F without knowing the implementation.
  The orchestrator routes the call to the owning agent when needed.

This allows building large systems incrementally:
  - Planner agent defines contracts
  - Specialist agents implement them
  - Other agents call them as black boxes
  - Unit tests verify them without knowing internals

Example project: build a compiler
  - planner registers: tokenize(source) -> tokens, parse(tokens) -> ast, codegen(ast) -> asm
  - lexer_agent implements tokenize
  - parser_agent uses tokenize (stub), implements parse
  - codegen_agent uses parse (stub), implements codegen
  - tester_agent writes unit tests for each
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from router import ModelRouter, Message
from memory import SharedMemory, WorkingMemory
from agent import Agent, AgentConfig


# ─────────────────────────────────────────────
# Task graph
# ─────────────────────────────────────────────

@dataclass
class Task:
    id: str
    description: str
    assigned_to: str           # agent_id
    depends_on: list[str] = field(default_factory=list)  # Task IDs this depends on
    status: str = "pending"    # pending | running | done | failed
    result: str = ""
    error: str = ""


class TaskGraph:
    """Dependency-aware task scheduler."""

    def __init__(self):
        self.tasks: dict[str, Task] = {}

    def add(self, task: Task):
        self.tasks[task.id] = task

    def ready_tasks(self) -> list[Task]:
        """Return tasks where all dependencies are done."""
        ready = []
        for task in self.tasks.values():
            if task.status != "pending":
                continue
            deps_done = all(
                self.tasks.get(dep, Task("", "", "")).status == "done"
                for dep in task.depends_on
            )
            if deps_done:
                ready.append(task)
        return ready

    def complete(self, task_id: str, result: str):
        if task_id in self.tasks:
            self.tasks[task_id].status = "done"
            self.tasks[task_id].result = result

    def fail(self, task_id: str, error: str):
        if task_id in self.tasks:
            self.tasks[task_id].status = "failed"
            self.tasks[task_id].error = error

    def all_done(self) -> bool:
        return all(t.status in ("done", "failed") for t in self.tasks.values())

    def summary(self) -> str:
        lines = ["Task Graph:"]
        for task in self.tasks.values():
            icon = {"pending": "○", "running": "◌", "done": "●", "failed": "✗"}.get(task.status, "?")
            lines.append(f"  {icon} [{task.assigned_to}] {task.id}: {task.description[:60]}")
            if task.depends_on:
                lines.append(f"      depends on: {', '.join(task.depends_on)}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Unit test record
# ─────────────────────────────────────────────

@dataclass
class UnitTest:
    id: str
    contract_name: str
    description: str
    input_example: str
    expected_output: str
    status: str = "unrun"     # unrun | passed | failed
    actual_output: str = ""


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────

class Orchestrator:
    """
    Manages a team of agents working on a project.
    
    Responsibilities:
      1. Create and configure agents
      2. Register function contracts (stubs) in shared memory
      3. Build and execute task graphs
      4. Route contract calls to the owning agent
      5. Generate and track unit tests
      6. Collect and store build artifacts
    """

    def __init__(self, router: ModelRouter, memory_dir: str = "./memory"):
        self.router = router
        self.shared = SharedMemory(memory_dir)
        self.agents: dict[str, Agent] = {}
        self.task_graph = TaskGraph()
        self.unit_tests: dict[str, UnitTest] = {}
        self.memory_dir = memory_dir

    # ── Agent management ──────────────────────────────────────────────

    def add_agent(self, config: AgentConfig) -> Agent:
        agent = Agent(config, self.router, self.shared)
        self.agents[config.agent_id] = agent
        print(f"  + Agent registered: {config.agent_id} ({config.role}) using {config.model_alias}")
        return agent

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self.agents.get(agent_id)

    # ── Contract system ────────────────────────────────────────────────

    def register_contract(
        self,
        name: str,
        signature: str,
        description: str,
        owner_agent_id: str,
        implemented: bool = False,
    ):
        """
        Register a function contract in shared memory.
        All agents will be aware of this contract after refresh_contracts().
        """
        self.shared.register_contract(
            name=name,
            signature=signature,
            description=description,
            owner_agent=owner_agent_id,
            implemented=implemented,
        )
        print(f"  ⚙ Contract registered: {name}({signature}) → owner: {owner_agent_id}")

        # Refresh all agents' system prompts
        for agent in self.agents.values():
            agent.refresh_contracts()

    def mark_contract_implemented(self, name: str):
        contract = self.shared.get(f"contract:{name}")
        if contract:
            contract["implemented"] = True
            self.shared.set(f"contract:{name}", contract)
            print(f"  ✓ Contract implemented: {name}")

    def call_contract(
        self,
        name: str,
        caller_agent_id: str,
        task_description: str,
        args: dict,
        verbose: bool = True,
    ) -> str:
        """
        Route a contract call to the owning agent.
        The caller doesn't know the implementation — just the interface.
        """
        contract = self.shared.get(f"contract:{name}")
        if not contract:
            raise KeyError(f"Contract '{name}' not registered")

        owner_id = contract["owner"]
        owner = self.agents.get(owner_id)
        if not owner:
            raise ValueError(f"Owner agent '{owner_id}' not found")

        # Build a task message for the owner agent
        task = (
            f"You are being asked to execute the contract: {name}({contract['signature']})\n"
            f"Caller: {caller_agent_id}\n"
            f"Task: {task_description}\n"
            f"Arguments: {json.dumps(args, indent=2)}\n\n"
            f"Execute this contract and return the result."
        )

        if verbose:
            print(f"\n  📞 Contract call: {caller_agent_id} → {name}() → {owner_id}")

        self.shared.call_contract(name, caller_agent_id, args)
        return owner.run(task, verbose=verbose)

    # ── Task graph execution ───────────────────────────────────────────

    def add_task(self, task: Task):
        self.task_graph.add(task)

    def run_task(self, task: Task, verbose: bool = True) -> str:
        agent = self.agents.get(task.assigned_to)
        if not agent:
            err = f"No agent found for: {task.assigned_to}"
            self.task_graph.fail(task.id, err)
            return err

        task.status = "running"
        try:
            result = agent.run(task.description, verbose=verbose)
            self.task_graph.complete(task.id, result)
            # Store as artifact
            self.shared.store_artifact(
                f"task_{task.id}_result",
                result,
                task.assigned_to,
                artifact_type="task_result",
            )
            return result
        except Exception as e:
            err = str(e)
            self.task_graph.fail(task.id, err)
            return err

    def run_all(self, verbose: bool = True, max_rounds: int = 20) -> dict[str, str]:
        """
        Execute the full task graph, respecting dependencies.
        Returns results per task ID.
        """
        print(f"\n{'═'*60}")
        print("  Running task graph...")
        print(self.task_graph.summary())
        print(f"{'═'*60}\n")

        results = {}
        round_num = 0

        while not self.task_graph.all_done() and round_num < max_rounds:
            round_num += 1
            ready = self.task_graph.ready_tasks()
            if not ready:
                print(f"  ⚠  No ready tasks (possible circular dependency?)")
                break

            for task in ready:
                print(f"\n  ▶ Running task: [{task.id}] {task.description[:60]}")
                result = self.run_task(task, verbose=verbose)
                results[task.id] = result

        print(f"\n{'═'*60}")
        print("  Task graph complete.")
        print(self.task_graph.summary())
        print(f"{'═'*60}")
        return results

    # ── Unit test generation ───────────────────────────────────────────

    def generate_unit_tests(
        self,
        contract_name: str,
        tester_agent_id: str,
        num_tests: int = 3,
    ) -> list[UnitTest]:
        """
        Ask the tester agent to generate unit tests for a contract.
        This is "build safety" — tests are generated against the interface,
        not the implementation.
        """
        contract = self.shared.get(f"contract:{contract_name}")
        if not contract:
            raise KeyError(f"Contract '{contract_name}' not found")

        tester = self.agents.get(tester_agent_id)
        if not tester:
            raise ValueError(f"Tester agent '{tester_agent_id}' not found")

        task = f"""
Generate {num_tests} unit tests for this function contract:

Contract: {contract_name}
Signature: {contract['signature']}
Description: {contract['description']}
Owner: {contract['owner']}

For each test, output a JSON object with:
  - id: unique test ID
  - description: what this test checks
  - input_example: example input (as string)
  - expected_output: expected result (as string)

Output as a JSON array. Tests should cover: happy path, edge cases, error cases.
"""
        raw = tester.run(task, verbose=False)

        # Parse the JSON array from the response
        tests = []
        try:
            # Try to extract JSON array
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                test_data = json.loads(match.group())
                for i, t in enumerate(test_data):
                    test = UnitTest(
                        id=t.get("id", f"{contract_name}_test_{i}"),
                        contract_name=contract_name,
                        description=t.get("description", ""),
                        input_example=t.get("input_example", ""),
                        expected_output=t.get("expected_output", ""),
                    )
                    self.unit_tests[test.id] = test
                    tests.append(test)
        except Exception as e:
            print(f"  ⚠  Could not parse unit tests: {e}")
            print(f"  Raw output: {raw[:200]}")

        print(f"  ✓ Generated {len(tests)} unit tests for {contract_name}")
        return tests

    def show_tests(self):
        print(f"\n{'─'*60}")
        print("  Unit Tests:")
        for test in self.unit_tests.values():
            icon = {"unrun": "○", "passed": "✓", "failed": "✗"}.get(test.status, "?")
            print(f"  {icon} [{test.contract_name}] {test.id}: {test.description}")
        print(f"{'─'*60}")

    # ── Project status ─────────────────────────────────────────────────

    def status(self):
        print(f"\n{'═'*60}")
        print("  Orchestrator Status")
        print(f"{'═'*60}")
        print(f"  Agents:    {len(self.agents)}")
        print(f"  Contracts: {len(self.shared.get_contracts())}")
        print(f"  Artifacts: {len(self.shared.list_artifacts())}")
        print(f"  Tests:     {len(self.unit_tests)}")
        print(f"  Tasks:     {len(self.task_graph.tasks)}")
        print()
        print("  Contracts:")
        for name, c in self.shared.get_contracts().items():
            status = "✓" if c.get("implemented") else "⚙"
            print(f"    {status} {name}  [{c['owner']}]  {c['signature']}")
        print()
        self.task_graph.summary()
        print(f"{'═'*60}")


import re  # needed for generate_unit_tests
