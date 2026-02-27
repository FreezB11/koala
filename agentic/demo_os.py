"""
demo_os.py — Multi-agent Simple OS Builder

Builds a bootable x86 OS from scratch using 5 agents:

  architect     — designs layout, creates dirs, writes linker script
  asm_writer    — writes the x86 bootloader (boot.asm) and entry point (entry.asm)
  kernel_writer — writes the C kernel (kernel.c, vga.c, vga.h)
  build_eng     — writes Makefile, compiles everything with NASM + GCC
  tester        — runs in QEMU, verifies boot, reports output

What gets built:
  os/
    boot/boot.asm       ← 512-byte MBR bootloader (x86 real mode)
    kernel/entry.asm    ← switches to protected mode, calls kernel_main()
    kernel/kernel.c     ← kernel_main(), prints to VGA text buffer
    kernel/vga.h        ← VGA text mode helpers
    kernel/vga.c        ← VGA implementation
    linker.ld           ← linker script (kernel loads at 1MB)
    Makefile            ← builds boot.bin + kernel.bin → os.img

Requirements (install if missing):
  sudo apt install nasm gcc-multilib qemu-system-x86

Run: python demo_os.py
Run with custom dir: python demo_os.py --dir ./my_os
"""

import os
import sys
from orchestrator import Orchestrator, Task
from agent import AgentConfig
from router import ModelRouter
from tools import list_directory, run_shell


def run(project_dir: str = "./os_project"):

    router = ModelRouter()
    available = router.available_models()

    if not available:
        print("No API keys found — check your .env")
        sys.exit(1)

    def pick(prefs):
        return next((m for m in prefs if m in available), available[0])

    # Codestral is best for low-level C/ASM, groq for speed
    asm_model    = pick(["codestral", "groq-llama-70b", "mistral-small"])
    kernel_model = pick(["codestral", "groq-llama-70b", "mistral-small"])
    arch_model   = pick(["groq-llama-70b", "mistral-small", "codestral"])
    build_model  = pick(["codestral", "groq-llama-70b", "mistral-small"])
    test_model   = pick(["groq-llama-70b", "groq-mixtral", "mistral-small"])

    print(f"\n  OS Build — model assignments:")
    print(f"    architect     → {arch_model}")
    print(f"    asm_writer    → {asm_model}")
    print(f"    kernel_writer → {kernel_model}")
    print(f"    build_eng     → {build_model}")
    print(f"    tester        → {test_model}")

    orc = Orchestrator(router, memory_dir="./memory")

    # ── Agents ───────────────────────────────────────────────────────

    orc.add_agent(AgentConfig(
        agent_id="architect",
        role="OS architect",
        model_alias=arch_model,
        system_prompt=f"""You design OS project layouts and write linker scripts.
Project root: {project_dir}
Use run_shell to create directories. Use write_file for all files.
You know x86 memory layout: BIOS loads bootloader at 0x7C00, kernel at 1MB (0x100000).
Write precise, minimal files. No fluff.""",
        tools_enabled=["write_file", "read_file", "run_shell", "list_directory"],
    ))

    orc.add_agent(AgentConfig(
        agent_id="asm_writer",
        role="x86 assembly programmer",
        model_alias=asm_model,
        system_prompt=f"""You write x86 assembly for bare-metal OS development.
Project root: {project_dir}
Use NASM syntax. You are targeting real 32-bit x86 (i686).
You know:
- BIOS loads MBR (512 bytes) at 0x7C00 in real mode
- Bootloader must switch to protected mode and jump to kernel
- GDT setup, A20 line, protected mode entry sequence
- VGA text buffer is at 0xB8000
Write clean, commented NASM assembly. Use write_file to save .asm files.""",
        tools_enabled=["write_file", "read_file", "run_shell", "list_directory"],
    ))

    orc.add_agent(AgentConfig(
        agent_id="kernel_writer",
        role="OS kernel developer",
        model_alias=kernel_model,
        system_prompt=f"""You write bare-metal C kernels for x86.
Project root: {project_dir}
Rules for freestanding C kernels:
- NO standard library (no printf, malloc, etc.)
- NO stack protector (-fno-stack-protector)
- Compile as 32-bit (-m32)
- VGA text buffer is volatile uint16_t* at 0xB8000
- 80x25 character grid, each cell = (color << 8) | char
- kernel_main() is called from assembly entry point
Write clean, commented C. Use write_file to save files.""",
        tools_enabled=["write_file", "read_file", "run_shell", "list_directory"],
    ))

    orc.add_agent(AgentConfig(
        agent_id="build_eng",
        role="build engineer",
        model_alias=build_model,
        system_prompt=f"""You write Makefiles and compile OS projects.
Project root: {project_dir}
Tools available: nasm, gcc (multilib), ld, qemu-system-i386
Build steps for this OS:
  1. nasm -f elf32 entry.asm → entry.o
  2. gcc -m32 -ffreestanding -fno-stack-protector -c kernel.c → kernel.o
  3. gcc -m32 -ffreestanding -fno-stack-protector -c vga.c → vga.o
  4. ld -m elf_i386 -T linker.ld entry.o kernel.o vga.o → kernel.bin
  5. nasm -f bin boot.asm → boot.bin
  6. cat boot.bin kernel.bin > os.img (pad boot.bin to 512 bytes first)
Use run_shell to compile. Fix errors by reading source files. Try up to 4 times.""",
        tools_enabled=["write_file", "read_file", "run_shell", "list_directory", "search_codebase"],
    ))

    orc.add_agent(AgentConfig(
        agent_id="tester",
        role="OS tester",
        model_alias=test_model,
        system_prompt=f"""You test OS images using QEMU.
Project root: {project_dir}
Run: qemu-system-i386 -drive format=raw,file=os.img -nographic -no-reboot -serial stdio
     (add -display none if no display available)
A successful boot shows text on screen via VGA.
If QEMU is not available, verify the binary files exist and report their sizes.
Use run_shell. Report exactly what happened.""",
        tools_enabled=["run_shell", "read_file", "list_directory"],
    ))

    # ── Contracts ─────────────────────────────────────────────────────

    orc.register_contract(
        name="kernel_main",
        signature="void kernel_main(void)",
        description="C entry point called from assembly. Initializes VGA and prints to screen.",
        owner_agent_id="kernel_writer",
        implemented=False,
    )
    orc.register_contract(
        name="vga_print",
        signature="void vga_print(const char* str, uint8_t color)",
        description="Prints a string to VGA text buffer at current cursor position.",
        owner_agent_id="kernel_writer",
        implemented=False,
    )
    orc.register_contract(
        name="protected_mode_entry",
        signature="_start() -> calls kernel_main",
        description="ASM entry: sets up GDT, switches to protected mode, sets up stack, calls kernel_main.",
        owner_agent_id="asm_writer",
        implemented=False,
    )

    # ── Task graph ────────────────────────────────────────────────────

    orc.add_task(Task(
        id="scaffold",
        description=f"""
Set up the OS project structure.

1. Create directories:
   {project_dir}/boot
   {project_dir}/kernel
   {project_dir}/build

2. Write {project_dir}/linker.ld — linker script:
```
ENTRY(_start)
SECTIONS {{
    . = 0x100000;
    .text   : {{ *(.text)   }}
    .rodata : {{ *(.rodata) }}
    .data   : {{ *(.data)   }}
    .bss    : {{ *(.bss)    }}
}}
```

3. Write {project_dir}/Makefile:
```makefile
CC      = gcc
LD      = ld
NASM    = nasm
DIR     = {project_dir}

CFLAGS  = -m32 -ffreestanding -fno-stack-protector -fno-pic -nostdlib -Wall -O2
LDFLAGS = -m elf_i386 -T $(DIR)/linker.ld --oformat binary

.PHONY: all clean run

all: $(DIR)/build/os.img

$(DIR)/build/entry.o: $(DIR)/kernel/entry.asm
\t$(NASM) -f elf32 $< -o $@

$(DIR)/build/kernel.o: $(DIR)/kernel/kernel.c
\t$(CC) $(CFLAGS) -c $< -o $@

$(DIR)/build/vga.o: $(DIR)/kernel/vga.c
\t$(CC) $(CFLAGS) -c $< -o $@

$(DIR)/build/kernel.bin: $(DIR)/build/entry.o $(DIR)/build/kernel.o $(DIR)/build/vga.o
\t$(LD) $(LDFLAGS) -o $@ $^

$(DIR)/build/boot.bin: $(DIR)/boot/boot.asm
\t$(NASM) -f bin $< -o $@

$(DIR)/build/os.img: $(DIR)/build/boot.bin $(DIR)/build/kernel.bin
\tcat $^ > $@

clean:
\trm -f $(DIR)/build/*.o $(DIR)/build/*.bin $(DIR)/build/*.img

run: all
\tqemu-system-i386 -drive format=raw,file=$(DIR)/build/os.img -nographic -no-reboot
```

4. List the directory to confirm structure.
""",
        assigned_to="architect",
    ))

    orc.add_task(Task(
        id="bootloader",
        description=f"""
Write the x86 bootloader in NASM assembly.

Write {project_dir}/boot/boot.asm — a 512-byte MBR bootloader that:

1. Sets up real mode segment registers (ds, es, ss = 0)
2. Sets stack pointer (sp = 0x7C00)
3. Prints "Booting NexusOS..." using BIOS INT 0x10 (teletype mode AH=0x0E)
4. Loads the kernel from disk:
   - Use BIOS INT 0x13, AH=0x02 (read sectors)
   - Read 32 sectors from drive 0x80, head 0, cylinder 0, sector 2
   - Load into 0x1000:0x0000 (ES:BX)
5. Enables A20 line (fast A20 via port 0x92)
6. Loads a GDT with:
   - Null descriptor
   - Code descriptor: base=0, limit=4GB, 32-bit, ring 0
   - Data descriptor: base=0, limit=4GB, 32-bit, ring 0
7. Switches to protected mode (set PE bit in CR0)
8. Far jumps to 0x10000 (where kernel is loaded) with code selector 0x08
9. Ends with TIMES 510-($-$$) DB 0 and DW 0xAA55

Example A20 enable:
    in al, 0x92
    or al, 2
    out 0x92, al

Example GDT structure:
    gdt_null: dq 0
    gdt_code: dw 0xFFFF, 0, 0x9A00, 0x00CF
    gdt_data: dw 0xFFFF, 0, 0x9200, 0x00CF
    gdt_end:
    gdt_desc: dw gdt_end - gdt_null - 1
              dd gdt_null

Comment every section clearly.
""",
        assigned_to="asm_writer",
        depends_on=["scaffold"],
    ))

    orc.add_task(Task(
        id="entry_asm",
        description=f"""
Write the kernel entry point in NASM assembly.

Write {project_dir}/kernel/entry.asm — called after bootloader switches to protected mode:

1. [bits 32] — we are already in protected mode
2. [global _start] — export for linker
3. _start:
   - Set up segment registers: mov ax, 0x10 (data descriptor), load ds/es/fs/gs/ss
   - Set up stack: mov esp, 0x90000
   - Call kernel_main (declared extern)
   - Hang loop: cli / hlt / jmp hang

Full example structure:
```nasm
[bits 32]
[global _start]
[extern kernel_main]

_start:
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov esp, 0x90000

    call kernel_main

hang:
    cli
    hlt
    jmp hang
```

Write exactly this — it is the bridge between assembly and C.
""",
        assigned_to="asm_writer",
        depends_on=["scaffold"],
    ))

    orc.add_task(Task(
        id="vga_driver",
        description=f"""
Write the VGA text mode driver in C.

Write {project_dir}/kernel/vga.h:
```c
#pragma once
#include <stdint.h>

#define VGA_WIDTH  80
#define VGA_HEIGHT 25
#define VGA_BUFFER ((volatile uint16_t*)0xB8000)

typedef enum {{
    VGA_COLOR_BLACK         = 0,
    VGA_COLOR_BLUE          = 1,
    VGA_COLOR_GREEN         = 2,
    VGA_COLOR_CYAN          = 3,
    VGA_COLOR_RED           = 4,
    VGA_COLOR_MAGENTA       = 5,
    VGA_COLOR_BROWN         = 6,
    VGA_COLOR_LIGHT_GREY    = 7,
    VGA_COLOR_DARK_GREY     = 8,
    VGA_COLOR_LIGHT_BLUE    = 9,
    VGA_COLOR_LIGHT_GREEN   = 10,
    VGA_COLOR_LIGHT_CYAN    = 11,
    VGA_COLOR_LIGHT_RED     = 12,
    VGA_COLOR_LIGHT_MAGENTA = 13,
    VGA_COLOR_LIGHT_BROWN   = 14,
    VGA_COLOR_WHITE         = 15,
}} vga_color_t;

void vga_init(void);
void vga_clear(void);
void vga_putchar(char c, uint8_t color);
void vga_print(const char* str, uint8_t color);
void vga_println(const char* str, uint8_t color);
uint8_t vga_make_color(vga_color_t fg, vga_color_t bg);
```

Write {project_dir}/kernel/vga.c implementing all functions:
- vga_init(): clear screen, reset cursor to (0,0)
- vga_clear(): fill buffer with spaces (light grey on black)
- vga_putchar(c, color): write char at cursor, advance cursor, scroll on newline
- vga_print(str, color): call vga_putchar for each char
- vga_println(str, color): vga_print + newline
- vga_make_color(fg, bg): return (bg << 4) | fg
- Handle \\n by moving to next row, col=0
- Handle scrolling when row >= VGA_HEIGHT
No stdlib. No includes except vga.h and stdint.h/stddef.h.
""",
        assigned_to="kernel_writer",
        depends_on=["scaffold"],
    ))

    orc.add_task(Task(
        id="kernel",
        description=f"""
Write the main C kernel.

Read {project_dir}/kernel/vga.h first.

Write {project_dir}/kernel/kernel.c:

```c
#include "vga.h"

// Colors: vga_make_color(fg, bg)
// Example: vga_make_color(VGA_COLOR_LIGHT_GREEN, VGA_COLOR_BLACK)

void kernel_main(void) {{
    vga_init();
    vga_clear();

    uint8_t green  = vga_make_color(VGA_COLOR_LIGHT_GREEN,  VGA_COLOR_BLACK);
    uint8_t cyan   = vga_make_color(VGA_COLOR_LIGHT_CYAN,   VGA_COLOR_BLACK);
    uint8_t white  = vga_make_color(VGA_COLOR_WHITE,        VGA_COLOR_BLACK);
    uint8_t yellow = vga_make_color(VGA_COLOR_LIGHT_BROWN,  VGA_COLOR_BLACK);

    vga_println("============================================", white);
    vga_println("        NexusOS v0.1 - x86 Kernel          ", cyan);
    vga_println("============================================", white);
    vga_println("", white);
    vga_println("  [OK] Bootloader loaded", green);
    vga_println("  [OK] Protected mode active", green);
    vga_println("  [OK] VGA driver initialized", green);
    vga_println("  [OK] kernel_main() reached", green);
    vga_println("", white);
    vga_println("  System ready.", yellow);
    vga_println("", white);
    vga_println("  Built by NexusAgent multi-agent system", white);

    // Halt
    while (1) {{
        __asm__ volatile ("hlt");
    }}
}}
```

Write this exactly — it is the first code that runs in the OS.
""",
        assigned_to="kernel_writer",
        depends_on=["vga_driver"],
    ))

    orc.add_task(Task(
        id="compile",
        description=f"""
Compile the entire OS project.

1. First list {project_dir} and {project_dir}/build to see all files.

2. Check tools are available:
   run: nasm --version
   run: gcc --version
   run: ld --version

3. Compile step by step:

   Step 1 - Kernel entry (ASM):
   nasm -f elf32 {project_dir}/kernel/entry.asm -o {project_dir}/build/entry.o

   Step 2 - VGA driver:
   gcc -m32 -ffreestanding -fno-stack-protector -fno-pic -nostdlib -Wall -O2 -c {project_dir}/kernel/vga.c -o {project_dir}/build/vga.o

   Step 3 - Kernel:
   gcc -m32 -ffreestanding -fno-stack-protector -fno-pic -nostdlib -Wall -O2 -I{project_dir}/kernel -c {project_dir}/kernel/kernel.c -o {project_dir}/build/kernel.o

   Step 4 - Link kernel binary:
   ld -m elf_i386 -T {project_dir}/linker.ld --oformat binary -o {project_dir}/build/kernel.bin {project_dir}/build/entry.o {project_dir}/build/kernel.o {project_dir}/build/vga.o

   Step 5 - Bootloader:
   nasm -f bin {project_dir}/boot/boot.asm -o {project_dir}/build/boot.bin

   Step 6 - Create disk image:
   cat {project_dir}/build/boot.bin {project_dir}/build/kernel.bin > {project_dir}/build/os.img

4. After each step, check for errors. If a step fails:
   - Read the failing source file carefully
   - Fix syntax errors with write_file
   - Retry that step (up to 3 retries per step)

5. Verify final image exists:
   ls -la {project_dir}/build/os.img

Report success or failure for each step.
""",
        assigned_to="build_eng",
        depends_on=["bootloader", "entry_asm", "kernel", "vga_driver"],
    ))

    orc.add_task(Task(
        id="run_os",
        description=f"""
Test the OS image with QEMU.

1. Check if os.img was built:
   ls -la {project_dir}/build/os.img

2. Check if QEMU is available:
   which qemu-system-i386 || which qemu-system-x86_64

3. If QEMU available, run for 3 seconds and capture output:
   timeout 3 qemu-system-i386 -drive format=raw,file={project_dir}/build/os.img -nographic -no-reboot -serial stdio 2>&1 || true

4. If QEMU not available, report the file sizes:
   ls -la {project_dir}/build/

5. Report:
   - Was os.img created? (size in bytes)
   - Is boot.bin exactly 512 bytes?
   - Did QEMU start?
   - What appeared on screen?

Install QEMU hint if missing:
   sudo apt install qemu-system-x86
""",
        assigned_to="tester",
        depends_on=["compile"],
    ))

    # ── Run ────────────────────────────────────────────────────────────
    print(f"\n  Building NexusOS in {project_dir}/")
    print(f"  Contracts: {len(orc.shared.get_contracts())} registered")
    print(f"  Tasks: {len(orc.task_graph.tasks)}")
    print()

    results = orc.run_all(verbose=True)

    print("\n  Final file listing:")
    r = list_directory(f"{project_dir}/build")
    print(r.output)

    print("\n  Run result:")
    print(results.get("run_os", "No run result"))

    print(f"""
  ═══════════════════════════════════════════════
  To run the OS yourself (if QEMU is installed):
    qemu-system-i386 -drive format=raw,file={project_dir}/build/os.img -nographic -no-reboot

  To install QEMU:
    sudo apt install qemu-system-x86 nasm gcc-multilib
  ═══════════════════════════════════════════════
""")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Multi-agent OS builder")
    p.add_argument("--dir", default="./os_project", help="Output directory for OS files")
    args = p.parse_args()
    run(project_dir=args.dir)