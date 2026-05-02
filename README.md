# Nexus Infinite

High-performance chess engine with NNUE evaluation and automated training pipeline.

**Repository Architecture:**
- **Public repo (nexusinfinitepublic)**: Training workflows, releases, documentation
- **Private repo (nexusinfinitecore)**: Source code, data, weights (storage only)

## Downloads

| Version | Windows | Linux | macOS | NNUE Weights |
|---------|---------|-------|-------|--------------|
| Latest | [nexus.exe](releases/latest) | [nexus](releases/latest) | [nexus](releases/latest) | [best.nnue](releases/latest) |

## Elo Progress

See [ELO_DASHBOARD.md](ELO_DASHBOARD.md) for detailed performance history.

## Features

- NNUE neural network evaluation
- Automated training via GitHub Actions
- Multi-threaded search
- UCI protocol support

## Usage

```bash
# Windows
nexus.exe

# Linux/macOS
./nexus
```

## Training

This repository contains **release binaries only**. 

Source code and training infrastructure are private. Training is triggered via automated pipeline.

## License

Binary releases only. See LICENSE for details.
