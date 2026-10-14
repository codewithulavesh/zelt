from zelt.answers import extract_answer, is_correct, same_answer


def test_extract_answer_boxed():
    assert extract_answer(r"The result is \boxed{42}.") == "42"
    assert extract_answer(r"Therefore, \boxed{3/4} is correct.") == "3/4"
    assert extract_answer(r"\boxed{-15}") == "-15"


def test_extract_answer_xml():
    assert extract_answer("<answer>72</answer>") == "72"
    assert extract_answer("Reasoning...\n<answer>391</answer>") == "391"


def test_extract_answer_fallback_trailing():
    assert extract_answer("The total is 500.") == "500"
    assert extract_answer("Natalia sold 72 clips") == "72"


def test_is_correct_exact_and_normalized():
    assert is_correct("42", "42")
    assert is_correct(" 42 ", "42")
    assert is_correct("42.0", "42")
    assert is_correct("1,000", "1000")
    assert is_correct("$100", "100")
    assert is_correct("100%", "100")
    assert not is_correct("43", "42")


def test_same_answer():
    assert same_answer("72", "72")
    assert same_answer("72.0", "72")
    assert not same_answer("72", "73")
