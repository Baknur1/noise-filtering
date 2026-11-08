$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

python --version 2>&1 | Tee-Object "logs\python_version.txt"
python -m pip freeze | Set-Content "logs\requirements-lock.txt"
ollama --version 2>&1 | Tee-Object "logs\ollama_version.txt"
ollama list 2>&1 | Tee-Object "logs\ollama_models.txt"
nvidia-smi 2>&1 | Tee-Object "logs\nvidia_smi.txt"
systeminfo 2>&1 | Set-Content "logs\systeminfo.txt"
git rev-parse HEAD 2>&1 | Tee-Object "logs\git_commit.txt"
git status --short 2>&1 | Set-Content "logs\git_status.txt"

Write-Host "Environment evidence saved under logs\. Review the files before committing them."
