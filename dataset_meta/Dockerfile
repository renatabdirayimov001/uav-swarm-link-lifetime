# Reproducible CPU environment for the simulator, analytical baselines, GBT and analyses
FROM python:3.13.16-slim
WORKDIR /work
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY . .
# Verify the canonical dataset fingerprints, then run e.g.:  NO_NEURAL=1 python run_v4.py main
CMD ["python", "verify_dataset.py"]
