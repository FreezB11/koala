"""
demo_cpp.py — Multi-agent C++ project demo

Builds a C++ linked list library from scratch using 4 agents:
  - architect  : designs the API, writes header file
  - implementer: writes the .cpp implementation
  - tester     : writes unit tests
  - builder    : compiles and runs everything, reports results

Uses the contract/stub system:
  - architect registers: LinkedList::push(), pop(), print(), size()
  - implementer uses the header (stub) to implement
  - tester uses the header (stub) to write tests
  - builder compiles and verifies everything works

Run: python demo_cpp.py
"""

import os
import sys
from orchestrator import Orchestrator, Task
from agent import AgentConfig
from router import ModelRouter


def run(project_dir: str = "./cpp_project"):
    router = ModelRouter()
    available = router.available_models()

    if not available:
        print("No API keys found — check your .env")
        sys.exit(1)

    # Pick best models for each role
    # Codestral is ideal for implementer/tester, groq for speed
    def pick(preferences):
        return next((m for m in preferences if m in available), available[0])

    architect_model   = pick(["groq-llama-70b", "mistral-small", "hermes-3-405b"])
    implementer_model = pick(["codestral", "groq-llama-70b", "mistral-small"])
    tester_model      = pick(["codestral", "groq-llama-70b", "mistral-small"])
    builder_model     = pick(["groq-llama-70b", "groq-mixtral", "mistral-small"])

    print(f"\n  Models selected:")
    print(f"    architect   → {architect_model}")
    print(f"    implementer → {implementer_model}")
    print(f"    tester      → {tester_model}")
    print(f"    builder     → {builder_model}")

    orc = Orchestrator(router, memory_dir="./memory")

    # ── Register agents ──────────────────────────────────────────────

    orc.add_agent(AgentConfig(
        agent_id="architect",
        role="C++ software architect",
        model_alias=architect_model,
        system_prompt=f"""You design clean C++ APIs.
Your job: create the project directory and write clean header files.
Project directory: {project_dir}
Always use write_file to save your work. Always use run_shell to create directories.
Write modern C++17. Use include guards. No external dependencies.""",
        tools_enabled=["write_file", "read_file", "run_shell", "list_directory"],
    ))

    orc.add_agent(AgentConfig(
        agent_id="implementer",
        role="C++ implementer",
        model_alias=implementer_model,
        system_prompt=f"""You implement C++ code based on header files.
Project directory: {project_dir}
Always read the header file first before implementing.
Use write_file to save .cpp files. Write clean, correct C++17.""",
        tools_enabled=["write_file", "read_file", "run_shell", "list_directory"],
    ))

    orc.add_agent(AgentConfig(
        agent_id="tester",
        role="C++ test engineer",
        model_alias=tester_model,
        system_prompt=f"""You write thorough C++ unit tests.
Project directory: {project_dir}
Read the header file to understand the API before writing tests.
Write a standalone test file using assert(). Cover happy path, edge cases, stress cases.""",
        tools_enabled=["write_file", "read_file", "run_shell", "list_directory"],
    ))

    orc.add_agent(AgentConfig(
        agent_id="builder",
        role="C++ build engineer",
        model_alias=builder_model,
        system_prompt=f"""You compile and run C++ projects.
Project directory: {project_dir}
Use run_shell with g++ to compile. Fix any compilation errors by reading the source files.
Report all test pass/fail results clearly.""",
        tools_enabled=["write_file", "read_file", "run_shell", "list_directory", "search_codebase"],
    ))

    # ── Register contracts (make-belief stubs) ───────────────────────

    orc.register_contract(
        name="LinkedList::push",
        signature="push(int value) -> void",
        description="Appends an integer to the end of the linked list",
        owner_agent_id="implementer",
        implemented=False,
    )
    orc.register_contract(
        name="LinkedList::pop",
        signature="pop() -> int",
        description="Removes and returns the last element. Throws if empty.",
        owner_agent_id="implementer",
        implemented=False,
    )
    orc.register_contract(
        name="LinkedList::size",
        signature="size() -> int",
        description="Returns the number of elements in the list",
        owner_agent_id="implementer",
        implemented=False,
    )

    # ── Task graph ───────────────────────────────────────────────────

    orc.add_task(Task(
        id="setup",
        description=f"""
Create the project directory structure for a C++ linked list library.

1. Run: mkdir -p {project_dir}
2. Write {project_dir}/linked_list.h — a header file defining:
   - A Node struct with: int data, Node* next
   - A LinkedList class with public methods:
       void push(int value)    // append to end
       int  pop()              // remove+return last, throw std::runtime_error if empty
       int  size() const       // return count
       void print() const      // print all elements space-separated
       bool empty() const      // return true if size == 0
       void clear()            // remove all elements
       ~LinkedList()           // destructor (free memory)
   - Include guards: LINKED_LIST_H
   - Include: <stdexcept>, <iostream>

3. Write {project_dir}/CMakeLists.txt:
   cmake_minimum_required(VERSION 3.10)
   project(LinkedList)
   set(CMAKE_CXX_STANDARD 17)
   add_library(linked_list linked_list.cpp)
   add_executable(test_runner test_runner.cpp linked_list.cpp)

After writing files, list the directory to confirm.
""",
        assigned_to="architect",
    ))

    orc.add_task(Task(
        id="implement",
        description=f"""
Implement the LinkedList class.

1. Read {project_dir}/linked_list.h to understand the interface
2. Write {project_dir}/linked_list.cpp implementing ALL methods:
   - push(int): create new Node, append to end (handle empty list)
   - pop(): find last node, remove it, return its data, throw if empty
   - size(): count and return number of nodes
   - print(): iterate and print each node's data space-separated, then newline
   - empty(): return size() == 0
   - clear(): delete all nodes, set head to nullptr
   - ~LinkedList(): call clear()

Use a singly linked list with a head pointer.
Do NOT use std::list or any STL containers — implement from scratch.
After writing, read the file back to verify it looks correct.
""",
        assigned_to="implementer",
        depends_on=["setup"],
    ))

    orc.add_task(Task(
        id="test",
        description=f"""
Write comprehensive unit tests for the LinkedList class.

1. Read {project_dir}/linked_list.h to understand the API
2. Write {project_dir}/test_runner.cpp with a main() function that:

   Tests to include:
   - push 3 elements, verify size() == 3
   - pop() returns correct last element
   - pop() on empty list throws std::runtime_error
   - empty() returns true on new list, false after push
   - clear() makes size() == 0
   - push 1000 elements (stress test), verify size() == 1000
   - push then pop repeatedly — verify correct LIFO order
   - print() runs without crash

   Use assert() for all checks. Print "TEST PASSED: <name>" for each test.
   Print "ALL TESTS PASSED" at the end.
   Include: linked_list.h, <cassert>, <iostream>, <stdexcept>

After writing, read the file back to verify.
""",
        assigned_to="tester",
        depends_on=["setup"],
    ))

    orc.add_task(Task(
        id="build_and_run",
        description=f"""
Compile and run the C++ linked list project.

1. List {project_dir} to see all files
2. Compile with:
   g++ -std=c++17 -Wall -o {project_dir}/test_runner {project_dir}/test_runner.cpp {project_dir}/linked_list.cpp

3. If compilation fails:
   - Read the error carefully
   - Read the relevant source file
   - Fix the issue with write_file
   - Recompile (up to 3 attempts)

4. Run the tests:
   {project_dir}/test_runner

5. Report:
   - Whether compilation succeeded
   - Which tests passed/failed
   - Any output from the program

If all tests pass, print a summary of what was built.
""",
        assigned_to="builder",
        depends_on=["implement", "test"],
    ))

    # ── Run ──────────────────────────────────────────────────────────
    print(f"\n  Building C++ linked list library in {project_dir}/")
    print(f"  Contract stubs registered: {len(orc.shared.get_contracts())}")

    results = orc.run_all(verbose=True)

    # Show final files
    print("\n  Generated files:")
    from tools import list_directory
    r = list_directory(project_dir)
    print(r.output)

    print("\n  Build result:")
    print(results.get("build_and_run", "No build result"))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="./cpp_project", help="Output directory")
    args = p.parse_args()
    run(project_dir=args.dir)