from offline_recovery_exchange import Job, State, exact_joint, feasible_schedule, greedy_retime


def test_success_releases_a_slot_but_greedy_can_make_the_same_exchange():
    # B has already succeeded; A failed and still owes its conventional path.
    ai = Job("ai", 78, 20, 100, value=1)
    fixed = State(78, (Job("A", 60, 25, 128, fixed_start_ms=78),), (ai,))
    movable = State(78, (Job("A", 60, 25, 128),), (ai,))
    assert exact_joint(fixed)["value"] == 0
    assert greedy_retime(movable, "deadline")["value"] == 1
    assert exact_joint(movable)["value"] == 1


def test_joint_subset_search_can_beat_all_three_greedy_orders():
    # This is an AI packing control case, not by itself AI-RAN novelty.
    recovery = Job("recovery", 0, 25, 65, fixed_start_ms=40)
    ai = (
        Job("A", 0, 25, 40, value=100),
        Job("B", 0, 20, 40, value=75),
        Job("C", 0, 20, 40, value=75),
        Job("D", 0, 5, 40, value=25),
    )
    state = State(0, (recovery,), ai)
    joint = exact_joint(state)
    assert joint["value"] == 150
    assert joint["admitted"] == ["B", "C"]
    for priority in ("density", "deadline", "value"):
        assert greedy_retime(state, priority)["value"] == 125
    assert joint["schedule"][-1]["job"] == "recovery"
    assert joint["schedule"][-1]["finish_ms"] == 65


def test_all_fail_recovery_is_mandatory_and_deadline_checked():
    recovery = Job("recovery", 0, 25, 30)
    ai = Job("ai", 0, 10, 20, value=3)
    state = State(0, (recovery,), (ai,))
    assert exact_joint(state)["value"] == 0
    assert feasible_schedule(0, (recovery, ai)) is None


def test_future_ai_is_not_visible_to_an_online_decision():
    try:
        State(10, (), (Job("future_ai", 11, 1, 20, value=1),))
    except ValueError as error:
        assert "future arrival" in str(error)
    else:
        raise AssertionError("future AI leaked into current state")
