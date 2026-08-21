"""FR3 experiments for the Koopman-accelerated Model-Based Diffusion paper.

Layout:
    config.py     typed config tree, JSON files under ``configs``, CLI overrides
    plant.py      FR3 plant, observation b = [q, p_tcp], excitation
    koopman.py    linear / bilinear lifted models, MLP control, lambda scaling
    training.py   multi-step identification and checkpoint cache
    planner.py    annealed sampling optimizer and its schedules
    backends.py   rollout backends (oracle, learned model, split)
    trial.py      one closed-loop trial and its record
    run_b.py      experiment B: rollout class and planning latency
    run_c.py      experiment C: annealing under controlled surrogate error
    stats.py      aggregation for both
"""
