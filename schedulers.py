"""
schedulers.py
Classical scheduling policies: FIFO, SJF, EDF.
Each function has the same signature as the action_fn expected by ChargingSimulator.
"""


def fifo(queue, free_ports, t):
    """
    First-In First-Out.
    Assigns the first 'free_ports' EVs in queue order (arrival order).
    """
    return list(queue[:free_ports])


def sjf(queue, free_ports, t):
    """
    Shortest-Job-First.
    Prioritises EVs with the smallest remaining energy demand.
    """
    sorted_q = sorted(queue, key=lambda ev: ev["energy_rem"])
    return sorted_q[:free_ports]


def edf(queue, free_ports, t):
    """
    Earliest-Deadline-First.
    Prioritises EVs with the earliest departure deadline.
    """
    sorted_q = sorted(queue, key=lambda ev: ev["deadline_step"])
    return sorted_q[:free_ports]


def llf(queue, free_ports, t):
    """
    Least-Laxity-First.
    Laxity = (deadline_step - t) - min_steps_needed
    Prioritises EVs with the smallest laxity (most urgent).
    """
    def laxity(ev):
        slack = (ev["deadline_step"] - t)
        min_steps = ev.get("min_steps_needed", 1)
        return slack - min_steps

    sorted_q = sorted(queue, key=laxity)
    return sorted_q[:free_ports]


def priority(queue, free_ports, t):
    """
    Priority Scheduler.
    Prioritises EVs based on 'claimed' status (1 is higher priority than 0).
    Ties are broken by EDF (earliest deadline first).
    """
    # Sort key: (-claimed, deadline_step) -> this sorts descending for claimed, ascending for deadline
    sorted_q = sorted(queue, key=lambda ev: (-ev.get("claimed", 0), ev["deadline_step"]))
    return sorted_q[:free_ports]


# ── Scheduler registry for easy lookup ───────────────────────────────────────
SCHEDULERS = {
    "FIFO": fifo,
    "SJF":  sjf,
    "EDF":  edf,
    "LLF":  llf,
    "PRIORITY": priority,
}

def get_scheduler(name):
    """Return scheduler function by name (case-insensitive)."""
    fn = SCHEDULERS.get(name.upper())
    if fn is None:
        raise ValueError(f"Unknown scheduler '{name}'. "
                         f"Choose from {list(SCHEDULERS.keys())}")
    return fn
