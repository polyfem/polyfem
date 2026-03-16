# Easy PolyFEM Wrapper

This directory contains a small **prototype C++ wrapper** for PolyFEM that provides a simplified command-line interface for running simulations.

The wrapper demonstrates how PolyFEM can be invoked through a minimal C++ program that:

1. Accepts a mesh file path and output directory
2. Generates a PolyFEM JSON configuration file automatically
3. Optionally invokes the PolyFEM solver (`PolyFEM_bin`)

The goal is to provide a lightweight example of exposing PolyFEM functionality through a simpler interface without modifying the internal PolyFEM codebase.

---

# Background

PolyFEM normally expects a **JSON configuration file** that describes the simulation setup.  
This configuration includes information such as:

- mesh file
- material model
- boundary conditions
- solver settings
- output options

A typical PolyFEM run looks like this:

```
PolyFEM_bin --json input.json
```

where `input.json` must be written by the user.

---

# What This Wrapper Changes

Instead of requiring users to manually write a JSON configuration file, the wrapper **generates the JSON automatically** based on simple command-line arguments.

Typical PolyFEM workflow:

```
User writes JSON configuration
        ↓
PolyFEM_bin reads JSON
        ↓
Simulation runs
```

Wrapper workflow:

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

# Directory Contents

```
easy_wrapper/
 ├── easy_polyfem.cpp
 ├── CMakeLists.txt
 └── README.md
```

| File | Description |
|-----|-------------|
| easy_polyfem.cpp | Main wrapper implementation |
| CMakeLists.txt | Build configuration for the wrapper |
| README.md | Documentation |

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

## Show Help

```
./easy_wrapper/build/easy_polyfem --help
```

Example output:

```
Usage:
easy_polyfem --mesh <mesh_file> [--problem laplacian] [--output <dir>]
             [--rhs <value>] [--polyfem-bin <path>] [--json-only]
```

---

# Generate JSON Only

This mode generates a PolyFEM JSON configuration file but **does not run the solver**.

```
./easy_wrapper/build/easy_polyfem \
--mesh path/to/mesh.obj \
--output output_directory \
--json-only
```

This creates a file such as:

```
generated_input.json
```

in the specified output directory.

---

# Running PolyFEM Through the Wrapper

First build the PolyFEM solver.

From the repository root:

```
mkdir build
cd build
cmake ..
make -j4
```

This produces the executable:

```
PolyFEM_bin
```

Then run the wrapper:

```
./easy_wrapper/build/easy_polyfem \
--mesh path/to/mesh.obj \
--output output_directory \
--polyfem-bin ./build/PolyFEM_bin
```

The wrapper will:

1. Generate a PolyFEM JSON configuration
2. Call the PolyFEM solver
3. Write simulation outputs

---

# Output Files

The wrapper and PolyFEM may generate files such as:

```
generated_input.json
stats.json
result.vtu
```

These are **generated files** and should **not be committed to the repository**.

They are written to the user-specified output directory.

---

# Visualizing Results

Simulation output files (e.g. `.vtu`) can be visualized using **ParaView**.

Steps:

1. Open ParaView
2. File → Open
3. Select the `.vtu` file
4. Click **Apply**

---

# Current Limitations

This wrapper is currently a **prototype**.

Limitations include:

- Supports only a simple **Laplacian** setup
- Boundary IDs are currently **hardcoded**
- Mesh boundary detection is not implemented
- Error handling is minimal

Future improvements could include:

- configurable boundary conditions
- support for additional PolyFEM problem types
- improved command-line interface
- automatic mesh boundary detection

---

# Repository Hygiene

To avoid polluting the repository with generated data, the following should **not be committed**:

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

It can serve as a starting point for building higher-level tools or integrations around PolyFEM.