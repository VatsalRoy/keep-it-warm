from env.grader import grade


def t(answer, match):
    return {"answer": answer, "match": match}


def test_exact_ignores_case_quotes_and_period():
    task = t(["opsctl heap dump --live"], "exact")
    assert grade(task, "opsctl heap dump --live")
    assert grade(task, "  `Opsctl heap dump --live`. ")
    assert grade(task, "Answer: opsctl heap dump --live")
    assert not grade(task, "opsctl heap dump")


def test_int_takes_first_integer():
    task = t(["7"], "int")
    assert grade(task, "7")
    assert grade(task, "There were 7 ERROR lines")
    assert not grade(task, "17")


def test_ordered_needs_same_order():
    task = t(["api-gateway", "auth", "orders"], "ordered")
    assert grade(task, "api-gateway, auth, orders")
    assert grade(task, "api-gateway -> auth -> orders")
    assert not grade(task, "auth, api-gateway, orders")


def test_set_ignores_order():
    task = t(["a", "b"], "set")
    assert grade(task, "b, a")
    assert not grade(task, "a")


def test_empty_answer_fails():
    assert not grade(t(["x"], "exact"), None)
    assert not grade(t(["x"], "exact"), "")