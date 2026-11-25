# LaunchForge AI — WSL2 Environment Setup

Target environment: **WSL2 Ubuntu**, **16 GB host RAM**, **RTX 5050 (8 GB VRAM, Blackwell `sm_120`)**, **CUDA 12.8+**.

---

## 1. Configure `.wslconfig` (Windows Host)

Cap WSL2 RAM at **12 GB** so Windows stays responsive and training/preprocessing never OOM-crashes the host.

Create or edit `C:\Users\<YOUR_WINDOWS_USER>\.wslconfig`:

```ini
[wsl2]
memory=12GB
swap=4GB
```

Apply the change from **PowerShell**:

```powershell
wsl --shutdown
```

Reopen your WSL2 Ubuntu terminal and verify:

```bash
free -h
```

---

## 2. Create a Python 3.11+ Virtual Environment (WSL2 Ubuntu)

```bash
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv python3-pip build-essential cmake git
cd /mnt/d/PROJECTS/CodeDoctor
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

---

## 3. Install PyTorch with CUDA 12.8+ (`cu128`)

Blackwell (`sm_120`) requires CUDA 12.8+. Install PyTorch **before** Unsloth or other training packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## 4. Install Project Dependencies

Install base + dev dependencies via `Makefile`:

```bash
make setup
```

Install training and RAG dependencies (when ready for GPU/RAG phases):

```bash
pip install -e ".[train,rag,dev]"
```

If `unsloth` has a version conflict with your `cu128` PyTorch wheel, install it with `--no-deps` first and verify with `scripts/check_env.py`:

```bash
pip install --no-deps unsloth
```

Copy `.env.example` to `.env` if you plan to use optional tokens:

```bash
cp .env.example .env
```

---

## 5. Build `llama.cpp` with CUDA (`sm_120`)

Build `llama-server` outside (or inside, git-ignored) the repository with CUDA enabled for Blackwell:

```bash
git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release -j"$(nproc)"
```

Verify the binary:

```bash
~/llama.cpp/build/bin/llama-server --help
```

---

## 6. Verify Setup

From `/mnt/d/PROJECTS/CodeDoctor` with `.venv` activated:

```bash
make setup && python scripts/check_env.py && make test
```
