# Easy PolyFEM Wrapper

This directory contains a **C++ wrapper for PolyFEM** that provides a simplified command-line interface for running simulations.

The wrapper demonstrates how PolyFEM can be invoked through a small C++ program that:

1. Accepts a mesh file and output directory
2. Generates a PolyFEM JSON configuration automatically
3. Optionally runs the PolyFEM solver (`PolyFEM_bin`)

The goal is to provide a lightweight example of exposing PolyFEM functionality through a simpler interface **without modifying the PolyFEM core codebase**.

---

# Background

PolyFEM normally expects a **JSON configuration file** describing the simulation setup.

This configuration includes information such as:

- mesh file
- material / physics model
- boundary conditions
- solver settings
- output configuration

A typical PolyFEM execution looks like:

```
PolyFEM_bin --json input.json
```

where `input.json` must be written manually by the user.

---

# What This Wrapper Changes

Instead of requiring users to manually write JSON configuration files, this wrapper **generates the JSON automatically** based on command-line arguments.

### Standard PolyFEM workflow

```
User writes JSON configuration
        ↓
PolyFEM_bin reads JSON
        ↓
Simulation runs
```

### Wrapper workflow

```
easy_polyfem
      ↓
Wrapper generates JSON configuration
      ↓
PolyFEM_bin runs
      ↓
Simulation output is produced
```

This simplifies the process of running PolyFEM simulations.

---

# Supported Features

The wrapper currently supports:

### Problem Types

The following PolyFEM problems can be selected:

```
laplacian
helmholtz
linear_elasticity
```

Example:

```
--problem laplacian
```

---

### Configurable Boundary Conditions

Dirichlet boundary conditions can be specified directly from the command line using the format:

```
--dirichlet <boundary_id>:<value>
```

Multiple boundaries can be provided:

```
--dirichlet 1:0
--dirichlet 4:1
```

---

### JSON-Only Mode

Generate the PolyFEM JSON input **without running the solver**:

```
--json-only
```

This is useful for debugging or inspecting the generated configuration.

---

### Custom Output File Names

You can customize generated output file names:

```
--json-name <file>
--stats-name <file>
--vtu-name <file>
```

Example:

```
--json-name simulation.json
--stats-name statistics.json
--vtu-name result.vtu
```

---

# Directory Structure

```
easy_wrapper/
 ├── CMakeLists.txt
 ├── README.md
 ├── include/
 │   └── easy_polyfem/
 │        ├── cli.hpp
 │        ├── config.hpp
 │        ├── json_writer.hpp
 │        ├── runner.hpp
 │        └── validator.hpp
 └── src/
      ├── main.cpp
      ├── cli.cpp
      ├── json_writer.cpp
      ├── runner.cpp
      └── validator.cpp
```

| File | Description |
|-----|-------------|
| main.cpp | Program entry point |
| cli.cpp | Command-line argument parsing |
| json_writer.cpp | Generates PolyFEM JSON configuration |
| runner.cpp | Executes `PolyFEM_bin` |
| validator.cpp | Input validation |
| config.hpp | Configuration data structures |

---

# Building the Wrapper

From the repository root:

```bash
mkdir -p easy_wrapper/build
cd easy_wrapper/build
cmake ..
make
```

This produces the executable:

```
easy_polyfem
```

---

# Usage

Show help:

```
./easy_wrapper/build/easy_polyfem --help
```

Example help output:

```
easy_polyfem --mesh <mesh_file>
             [--output <dir>]
             [--problem <laplacian|helmholtz|linear_elasticity>]
             [--rhs <value>]
             [--dirichlet <id:value>]
             [--polyfem-bin <path>]
             [--json-name <file>]
             [--stats-name <file>]
             [--vtu-name <file>]
             [--json-only]
```

---

# Example Usage

## Generate JSON Only

```
./easy_wrapper/build/easy_polyfem \
--mesh mesh.obj \
--output out \
--problem laplacian \
--rhs 10 \
--dirichlet 1:0 \
--dirichlet 4:1 \
--json-only
```

This generates:

```
generated_input.json
```

inside the output directory.

---

## Run PolyFEM Through the Wrapper

First build the PolyFEM solver.

From the repository root:

```
mkdir build
cd build
cmake ..
make -j4
```

This produces:

```
PolyFEM_bin
```

Then run the wrapper:

```
./easy_wrapper/build/easy_polyfem \
--mesh mesh.obj \
--output out \
--problem laplacian \
--rhs 10 \
--dirichlet 1:0 \
--dirichlet 4:1 \
--polyfem-bin ./build/PolyFEM_bin
```

The wrapper will:

1. Generate a PolyFEM JSON configuration
2. Call the PolyFEM solver
3. Write simulation outputs

---

# Output Files

PolyFEM may generate files such as:

```
generated_input.json
stats.json
result.vtu
```

These files are **generated outputs** and should **not be committed to the repository**.

They are written to the output directory specified by the user.

---

# Visualizing Results

Simulation results (`.vtu`) can be visualized using **ParaView**.

Steps:

1. Open ParaView
2. File → Open
3. Select the `.vtu` file
4. Click **Apply**

---

# Repository Hygiene

To avoid polluting the repository with generated files, the following should **not be committed**:

```
easy_wrapper/build/
build/
*.vtu
generated_input.json
stats.json
```

Generated outputs should remain local.

---

# Purpose

This wrapper demonstrates how PolyFEM functionality can be exposed through a simplified C++ interface while keeping the PolyFEM core architecture unchanged.

It provides a foundation for building higher-level tools or integrations around PolyFEM.