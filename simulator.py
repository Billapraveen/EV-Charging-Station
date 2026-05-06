"""
simulator.py
Discrete-time EV charging simulator.
Each step = 5 minutes.  The simulator is used by all schedulers and the RL agent.
"""

import numpy as np
import pandas as pd
from collections import deque

STEP_MINUTES = 5
STEP_HOURS   = STEP_MINUTES / 60.0   # 5/60 hours per step


class ChargingSimulator:
    """
    Multi-port EV charging simulator with a hard site power cap.

    Parameters
    ----------
    sessions   : pd.DataFrame  (output of data_loader.load_sessions)
    n_ports    : int           number of EVSEs (default 54 = Caltech)
    power_cap  : float         site power cap in kW (default 150)
    tou_rates  : dict          TOU tariff {'on_peak':..., 'mid_peak':..., 'off_peak':...}
    """

    # Southern California Edison TOU rates (INR, 1 USD = Rs.84)
    DEFAULT_TOU = {
        "on_peak":  21.00,   # Rs./kWh  weekdays 14:00-20:00 summer
        "mid_peak":  7.56,
        "off_peak":  5.04,
        "demand":  1300.0,   # Rs./kW/month peak demand charge
    }

    def __init__(self, sessions, n_ports=54, power_cap=150.0,
                 tou_rates=None, v2g_enabled=False):
        self.sessions   = sessions.copy()
        self.n_ports    = n_ports
        self.power_cap  = power_cap
        self.tou        = tou_rates or self.DEFAULT_TOU
        self.v2g_enabled = v2g_enabled

        # Build arrival lookup: step -> list of session rows
        self._arrivals = {}
        for _, row in self.sessions.iterrows():
            step = int(row["arrival_step"])
            self._arrivals.setdefault(step, []).append(row.to_dict())

        self.reset()

    # ── Public interface ─────────────────────────────────────────────────────
    def reset(self):
        """Reset simulator to time step 0."""
        self.t          = 0
        self.queue      = deque()           # waiting EVs
        self.active     = {}                # port_id -> EV dict
        self.completed  = []
        self.dropped    = []

        # Per-step history for metrics
        self.history_load     = []
        self.history_queue    = []
        self.history_active   = []

        self._max_load   = 0.0
        self._total_cost = 0.0

        return self._observe()

    def step(self, action_fn):
        """
        Advance simulator by one 5-minute step using a scheduling action.

        Parameters
        ----------
        action_fn : callable
            Takes (queue_list, free_ports, t) -> list of EV dicts to assign.
            'queue_list' is a list of EV dicts currently waiting.
            'free_ports' is how many ports are free.

        Returns
        -------
        obs    : dict   - current state observation
        reward : float  - step reward
        done   : bool   - True when no more sessions remain
        info   : dict   - extra diagnostics
        """
        # 1. Arrivals
        for ev in self._arrivals.get(self.t, []):
            ev["energy_rem"]  = float(ev["energy_kwh"])
            ev["wait_start"]  = self.t
            ev["charging"]    = False
            self.queue.append(ev)

        # 2. Departures
        drops_this_step = 0
        departed = [pid for pid, ev in self.active.items()
                    if ev["deadline_step"] <= self.t]
        for pid in departed:
            ev = self.active.pop(pid)
            if ev["energy_rem"] <= 1e-4:
                self.completed.append(ev)
            else:
                drops_this_step += 1
                self.dropped.append(ev)

        # Also expire queued EVs past deadline
        new_q = deque()
        for ev in self.queue:
            if ev["deadline_step"] <= self.t:
                drops_this_step += 1
                self.dropped.append(ev)
            else:
                new_q.append(ev)
        self.queue = new_q

        # 3. Port assignment via scheduler
        free_ports = self.n_ports - len(self.active)
        if free_ports > 0 and self.queue:
            chosen = action_fn(list(self.queue), free_ports, self.t)
            assigned_ids = {id(ev) for ev in chosen}
            self.queue = deque(
                ev for ev in self.queue if id(ev) not in assigned_ids)
            for i, ev in enumerate(chosen):
                ev["charging"]    = True
                ev["wait_time"]   = (self.t - ev["wait_start"]) * STEP_MINUTES
                self.active[len(self.active) + i] = ev

        # 4. Energy delivery with power cap
        load = self._deliver_energy()

        # 5. Record
        self.history_load.append(load)
        self.history_queue.append(len(self.queue))
        self.history_active.append(len(self.active))
        if load > self._max_load:
            self._max_load = load
        self._total_cost += load * self._tou_rate(self.t) * STEP_HOURS

        # 6. Reward
        reward = self._compute_reward(drops_this_step, load)

        self.t += 1
        done = (self.t >= self._max_step() and
                len(self.queue) == 0 and len(self.active) == 0)

        return self._observe(), reward, done, {
            "drops": drops_this_step,
            "load":  load,
            "queue": len(self.queue),
        }

    # ── Energy delivery ──────────────────────────────────────────────────────
    def _deliver_energy(self):
        if not self.active:
            return 0.0

        # Requested power for each active EV
        req = {}
        for pid, ev in self.active.items():
            p = min(ev["max_rate_kw"],
                    ev["energy_rem"] / STEP_HOURS)
            req[pid] = max(0.0, p)

        total_req = sum(req.values())

        # V2G logic: if enabled and total_req > cap OR currently on-peak
        v2g_discharged = 0.0
        if self.v2g_enabled:
            is_peak = self._tou_rate(self.t) == self.tou["on_peak"]
            if total_req > self.power_cap or is_peak:
                for pid, ev in list(self.active.items()):
                    slack_steps = ev["deadline_step"] - self.t
                    steps_needed = ev["energy_rem"] / (ev["max_rate_kw"] * STEP_HOURS) if ev["max_rate_kw"]>0 else 0
                    
                    # Can it afford to discharge? Need slack > steps_needed + some margin
                    # Also, only discharge if it has already charged some energy (energy_rem < energy_kwh)
                    if slack_steps > steps_needed + 4 and ev["energy_rem"] < ev["energy_kwh"]:
                        # Discharge at max rate
                        discharge_kw = min(ev["max_rate_kw"], (ev["energy_kwh"] - ev["energy_rem"]) / STEP_HOURS)
                        if discharge_kw > 0:
                            req[pid] = -discharge_kw
                            v2g_discharged += discharge_kw
                            # Re-evaluate total request
                            total_req = sum(req.values())
                            if total_req <= self.power_cap and not is_peak:
                                break # Relieved enough cap

        # Scale down proportionally if STILL over cap (only for positive requests)
        scale = 1.0
        pos_req = sum(r for r in req.values() if r > 0)
        if total_req > self.power_cap and pos_req > 0:
            allowed_pos = self.power_cap - sum(r for r in req.values() if r < 0)
            scale = max(0.0, allowed_pos / pos_req)

        actual_load = 0.0
        for pid, ev in self.active.items():
            r = req[pid]
            delivered_kw = r * scale if r > 0 else r
            delivered_kwh = delivered_kw * STEP_HOURS
            ev["energy_rem"] = max(0.0, ev["energy_rem"] - delivered_kwh)
            actual_load += delivered_kw

        # Remove fully charged EVs and free their ports
        done_ports = [pid for pid, ev in self.active.items()
                      if ev["energy_rem"] <= 1e-4]
        for pid in done_ports:
            self.completed.append(self.active.pop(pid))

        return actual_load

    # ── Observation ──────────────────────────────────────────────────────────
    def _observe(self):
        hour = (self.t * STEP_MINUTES / 60) % 24
        avg_slack = 0.0
        if self.queue:
            slacks = [(ev["deadline_step"] - self.t) * STEP_MINUTES
                      for ev in self.queue]
            avg_slack = max(0.0, float(np.mean(slacks)))

        load = self.history_load[-1] if self.history_load else 0.0

        return {
            "t":            self.t,
            "hour":         hour,
            "queue_len":    len(self.queue),
            "free_ports":   self.n_ports - len(self.active),
            "active_count": len(self.active),
            "avg_slack":    avg_slack,
            "load":         load,
            "load_norm":    load / self.power_cap,
        }

    # ── Reward ───────────────────────────────────────────────────────────────
    def _compute_reward(self, drops, load):
        prev_load = self.history_load[-2] if len(self.history_load) >= 2 else 0.0
        w_wait   = 1.0
        w_drop   = 5.0
        w_smooth = 0.5
        return -(
            w_wait   * len(self.queue)
            + w_drop   * drops
            + w_smooth * abs(load - prev_load)
        )

    # ── TOU rate ─────────────────────────────────────────────────────────────
    def _tou_rate(self, step):
        """Return INR/kWh rate for the given time step."""
        hour = (step * STEP_MINUTES / 60) % 24
        # Weekday on-peak: 14:00-20:00
        if 14 <= hour < 20:
            return self.tou["on_peak"]
        elif 9 <= hour < 14 or 20 <= hour < 23:
            return self.tou["mid_peak"]
        return self.tou["off_peak"]

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _max_step(self):
        return int(self.sessions["deadline_step"].max()) + 1

    # ── Metrics ──────────────────────────────────────────────────────────────
    def get_metrics(self):
        """Return all five evaluation metrics as a dict."""
        all_evs = self.completed + self.dropped
        n_total = len(all_evs)

        # Average waiting time
        wait_times = [ev.get("wait_time", 0) for ev in all_evs]
        avg_wait = float(np.mean(wait_times)) if wait_times else 0.0

        # Completion rate
        n_completed = len(self.completed)
        completion  = 100.0 * n_completed / n_total if n_total > 0 else 0.0

        # Station utilisation
        total_slots  = self.n_ports * len(self.history_active)
        used_slots   = sum(self.history_active)
        utilisation  = 100.0 * used_slots / total_slots if total_slots > 0 else 0.0

        # Grid load variance
        L = np.array(self.history_load)
        load_var = float(np.var(L)) if len(L) > 0 else 0.0

        # Energy cost
        energy_cost = self._total_cost + \
            self.tou["demand"] * self._max_load / 1000  # demand charge per month scaled

        return {
            "avg_waiting_time_min": round(avg_wait, 2),
            "completion_rate_pct":  round(completion, 2),
            "utilisation_pct":      round(utilisation, 2),
            "load_variance_kw2":    round(load_var, 2),
            "energy_cost_inr":      round(energy_cost, 2),
            "n_completed":          n_completed,
            "n_dropped":            len(self.dropped),
            "n_total":              n_total,
            "peak_load_kw":         round(self._max_load, 2),
        }


# ── Synthetic session generator (for testing without real data) ──────────────
def make_synthetic_sessions(n=500, n_steps=288, seed=42):
    """
    Generate synthetic EV sessions that mimic Caltech weekday patterns.
    n_steps=288 corresponds to 24 hours at 5-min resolution.
    """
    rng = np.random.default_rng(seed)

    # Morning arrival peak around step 96 (08:00)
    arrivals = np.clip(
        (rng.normal(96, 20, n)).astype(int), 0, n_steps - 10)

    # Parking duration: 4–10 hours (48–120 steps)
    durations = rng.integers(48, 121, size=n)
    deadlines = np.minimum(arrivals + durations, n_steps - 1)

    # Energy demand: 5–40 kWh
    energy = rng.uniform(5, 40, size=n)

    df = pd.DataFrame({
        "session_id":    [f"S{i:04d}" for i in range(n)],
        "arrival_step":  arrivals,
        "deadline_step": deadlines,
        "energy_kwh":    energy.round(2),
        "max_rate_kw":   np.full(n, 6.6),
        "userID":        [f"U{rng.integers(0,50):03d}" for _ in range(n)],
        "claimed":       rng.integers(0, 2, size=n),
        "min_steps_needed": np.ceil(energy / (6.6 * 5 / 60)).astype(int),
        "connectionTime": pd.date_range("2019-01-07 00:00",
                                        periods=n, freq="3min", tz="UTC"),
        "hist_mean_dur":      np.full(n, -1.0),
        "hist_mean_energy":   np.full(n, -1.0),
        "hist_mean_dep_hour": np.full(n, -1.0),
        "hour_sin": np.sin(2 * np.pi * (arrivals * 5 / 60 % 24) / 24),
        "hour_cos": np.cos(2 * np.pi * (arrivals * 5 / 60 % 24) / 24),
        "dow_sin":  np.zeros(n),
        "dow_cos":  np.ones(n),
    })

    return df, int(df["deadline_step"].max()) + 1
