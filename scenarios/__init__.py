"""
scenarios/case1_linear.py  — Constant-velocity obstacle (Case 1 from paper)
scenarios/case2_accel.py   — Constant-acceleration obstacle (Case 2)
scenarios/case3_figure8.py — Figure-8 obstacle (Case 3)

Each module exposes:
    obstacle_policy(t, p_obs) → new_p_obs  (callable for UAVNavEnv)
    SCENARIO_CONFIG            (dict with start, goal, init_obs_pos, obs_velocity)
"""
