"""
grader.py — Private grading script for Python Skills Test
Usage:
    python grader.py <notebook.ipynb>           # grade one notebook
    python grader.py submissions/               # grade all .ipynb in a folder
"""

import csv
import sys
import types
import nbformat
from nbconvert import PythonExporter
from pathlib import Path
from datetime import datetime


# Load a notebook into an importable module 

def load_notebook(path: str) -> types.ModuleType:
    """Execute a student notebook and return its globals as a module."""
    nb = nbformat.read(path, as_version=4)
    src, _ = PythonExporter().from_notebook_node(nb)

    module = types.ModuleType("student")
    try:
        exec(compile(src, path, "exec"), module.__dict__)
    except Exception as e:
        print(f"  [ERROR] Notebook crashed during execution: {e}")
    return module


# Individual problem tests 

def test_problem_1(s) -> tuple[bool, str]:
    """calculate_age(name, birth_year, current_year) -> greeting string"""
    fn = getattr(s, "calculate_age", None)
    if fn is None:
        return False, "function not found"
    try:
        assert fn("Alex", 2010, 2026) == "Hello Alex, you are 16 years old!"
        assert fn("Sam",  2000, 2026) == "Hello Sam, you are 26 years old!"
        return True, "passed"
    except AssertionError as e:
        return False, f"wrong output: {e}"
    except Exception as e:
        return False, f"raised {type(e).__name__}: {e}"


def test_problem_2(s) -> tuple[bool, str]:
    """count_even_odd(lst) -> (even_count, odd_count)"""
    fn = getattr(s, "count_even_odd", None)
    if fn is None:
        return False, "function not found"
    try:
        assert fn([1, 2, 3, 4, 5]) == (2, 3)
        assert fn([0, 11, 22])      == (2, 1)
        assert fn([])               == (0, 0)
        return True, "passed"
    except AssertionError as e:
        return False, f"wrong output: {e}"
    except Exception as e:
        return False, f"raised {type(e).__name__}: {e}"


def test_problem_3(s) -> tuple[bool, str]:
    """track_word_frequency(words, target) -> int (case-insensitive)"""
    fn = getattr(s, "track_word_frequency", None)
    if fn is None:
        return False, "function not found"
    try:
        assert fn(["Apple", "banana", "APPLE", "cherry"], "apple")  == 2
        assert fn(["blue", "red", "green"], "yellow")               == 0
        assert fn(["Python", "python", "pYtHoN"], "PYTHON")         == 3
        return True, "passed"
    except AssertionError as e:
        return False, f"wrong output: {e}"
    except Exception as e:
        return False, f"raised {type(e).__name__}: {e}"


def test_problem_4(s) -> tuple[bool, str]:
    """find_second_largest(lst) -> int | None"""
    fn = getattr(s, "find_second_largest", None)
    if fn is None:
        return False, "function not found"
    try:
        assert fn([10, 20, 4, 45, 99, 99]) == 45
        assert fn([5, 5, 5])               is None
        assert fn([12])                    is None
        assert fn([-1, -2, -3, -4])        == -2
        return True, "passed"
    except AssertionError as e:
        return False, f"wrong output: {e}"
    except Exception as e:
        return False, f"raised {type(e).__name__}: {e}"


def test_problem_5(s) -> tuple[bool, str]:
    """find_target_pair(lst, target) -> (a, b) | None"""
    fn = getattr(s, "find_target_pair", None)
    if fn is None:
        return False, "function not found"
    try:
        assert fn([2, 7, 11, 15], 9) == (2, 7)
        assert fn([3, 2, 4], 6)      == (2, 4)
        assert fn([1, 2, 3], 7)      is None
        assert fn([3, 3], 6)         == (3, 3)
        assert fn([3], 6)            is None
        return True, "passed"
    except AssertionError as e:
        return False, f"wrong output: {e}"
    except Exception as e:
        return False, f"raised {type(e).__name__}: {e}"


TESTS = [
    ("Problem 1 — calculate_age",        test_problem_1),
    ("Problem 2 — count_even_odd",       test_problem_2),
    ("Problem 3 — track_word_frequency", test_problem_3),
    ("Problem 4 — find_second_largest",  test_problem_4),
    ("Problem 5 — find_target_pair",     test_problem_5),
]


# Extract student name from notebook globals

def get_student_name(module: types.ModuleType) -> tuple[str, str]:
    """Pull FIRST_NAME / LAST_NAME variables out of the executed notebook."""
    first = getattr(module, "FIRST_NAME", "").strip()
    last  = getattr(module, "LAST_NAME",  "").strip()

    placeholder = {"fill in first name", "fill in last name", ""}
    if first.lower() in placeholder:
        first = "UNKNOWN"
    if last.lower() in placeholder:
        last = "UNKNOWN"

    return first, last


# Grade a single notebook

def grade(path: str) -> dict:
    """Grade one notebook. Returns a result dict."""
    print(f"\n{'='*55}")
    print(f"  Grading: {path}")
    print(f"{'='*55}")

    student = load_notebook(path)
    first_name, last_name = get_student_name(student)
    print(f"  Student: {first_name} {last_name}")

    score = 0
    problem_results = {}

    for label, test_fn in TESTS:
        passed, msg = test_fn(student)
        status = "✓ Solved" if passed else "✗ Not solved"
        print(f"  {status}  {label}")
        if not passed:
            print(f"            └ {msg}")
        score += int(passed)
        problem_results[label] = "Solved" if passed else f"Not solved ({msg})"

    print(f"\n  Score: {score} / {len(TESTS)}")

    return {
        "first_name":  first_name,
        "last_name":   last_name,
        "score":       score,
        "total":       len(TESTS),
        "file":        Path(path).name,
        **{f"p{i+1}": problem_results[label] for i, (label, _) in enumerate(TESTS)},
    }


# Save results to CSV 

def save_csv(results: list[dict], out_path: Path):
    if not results:
        return

    fieldnames = [
        "last_name", "first_name", "score", "total", "file",
        *[f"p{i+1}" for i in range(len(TESTS))],
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: (r["last_name"], r["first_name"])))

    print(f"\n  Results saved to: {out_path}")


# Entry point 

def main():
    if len(sys.argv) < 2:
        print("Usage: python grader.py <notebook.ipynb | folder/>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_dir():
        notebooks = sorted(target.glob("**/*.ipynb"))
        if not notebooks:
            print(f"No .ipynb files found in {target}")
            sys.exit(1)

        print(f"\nFound {len(notebooks)} notebook(s) to grade.\n")
        all_results = [grade(str(nb)) for nb in notebooks]

        # Summary table
        print(f"\n{'='*55}")
        print("  SUMMARY")
        print(f"{'='*55}")
        print(f"  {'Name':<30} {'File':<25} Score")
        print(f"  {'-'*30} {'-'*25} -----")
        for r in sorted(all_results, key=lambda r: (r["last_name"], r["first_name"])):
            name = f"{r['last_name']}, {r['first_name']}"
            print(f"  {name:<30} {r['file']:<25} {r['score']} / {r['total']}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_csv(all_results, target / f"grades_{timestamp}.csv")

    elif target.is_file():
        result = grade(str(target))
        save_csv([result], target.parent / f"grades_{target.stem}.csv")

    else:
        print(f"Path not found: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()