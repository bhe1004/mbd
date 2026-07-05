FROM python:3.11-slim
RUN pip install --no-cache-dir numpy scipy matplotlib cvxpy
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir mujoco
WORKDIR /work
CMD ["python", "tier0_verify.py"]
