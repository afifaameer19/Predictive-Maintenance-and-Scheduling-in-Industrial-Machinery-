import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import case_study_advanced as cs


def test_split_has_no_machine_leakage():
    tr, va, te = cs.split_data(cs.make_dataset(60))
    assert not set(tr.machine_id) & set(te.machine_id)
    assert not set(va.machine_id) & set(te.machine_id)


def test_worked_example_86_6():
    assert abs(cs.fuzzy_priority(.8, .9) - 86.6) < 1e-9


def test_memberships_sum_to_one():
    for x in np.linspace(0, 1, 101):
        assert abs(cs.memberships(x).sum() - 1) < 1e-12


def test_crossover_preserves_permutation():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = rng.permutation(12), rng.permutation(12)
        assert sorted(cs.ordered_crossover(a, b, rng)) == list(range(12))


def test_ga_not_worse_than_greedy_and_close_to_exact():
    f = cs.synthetic_fleet(12, np.random.default_rng(1))
    costs, _ = cs.build_costs(f)
    g = np.argsort(-f.priority.to_numpy())
    ga, _ = cs.genetic(lambda o: cs.objective(o, costs), 12, 0, g)
    ex = cs.exact_assignment(costs)
    assert cs.objective(ga, costs) <= cs.objective(g, costs) + 1e-9
    assert cs.objective(ga, costs) >= cs.objective(ex, costs) - 1e-9
