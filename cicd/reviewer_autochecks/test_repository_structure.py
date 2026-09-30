"""Bojaxns layout and readable array-schema checks; no runtime imports."""

import ast
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "bojaxns"
REQUIRED_PATHS = (
    "AGENTS.md",
    "LEARNINGS.md",
    "COMMON_CONTEXT.md",
    "docs/design/INVARIANTS.md",
    "docs/design/REQUIREMENTS.md",
    "docs/design/DEPENDENCIES.md",
    "docs/design/SOURCE_LAYOUT.md",
    "docs/design/system_tests/bayesian_optimisation.md",
    "cicd/coverage_record.json",
    "cicd/non_invariant_test_coverage.json",
    "cicd/pre_release_autochecks/check_all_invariants_covered.py",
    "cicd/demos/README.md",
    "cicd/benchmarks/README.md",
    "cicd/system_tests/README.md",
    "cicd/testing/README.md",
)


def test_required_cicd_and_design_paths_exist() -> None:
    missing = [path for path in REQUIRED_PATHS if not (REPO_ROOT / path).is_file()]
    assert not missing, "Missing scaffold files:\n" + "\n".join(missing)


def test_package_discovery_uses_only_src() -> None:
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "bojaxns"
    assert metadata["project"]["license"] == "Apache-2.0"
    assert metadata["project"]["license-files"] == ["LICENSE"]
    license_text = (REPO_ROOT / "LICENSE").read_text()
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    assert "Academic and Non-Commercial Use License" not in license_text
    assert metadata["project"]["requires-python"] == ">=3.10"
    dependencies = metadata["project"]["dependencies"]
    assert "jaxns==3.0.0" in dependencies
    assert "tfp-nightly==0.26.0.dev20260930" in dependencies
    assert "scipy" in dependencies
    assert "jaxctx>=1.2.0" in dependencies
    assert "etils" not in dependencies
    names = {dependency.split("=")[0].lower().replace("_", "-") for dependency in dependencies}
    assert not names & {"pydantic", "pydoe2", "tensorflow-probability"}
    assert metadata["tool"]["setuptools"]["packages"]["find"] == {
        "where": ["src"], "include": ["bojaxns*"],
    }
    assert not (REPO_ROOT / "setup.py").exists()
    assert (SOURCE_ROOT / "__init__.py").is_file()
    assert not list((REPO_ROOT / "bojaxns").rglob("*.py"))


def test_production_does_not_contain_tests() -> None:
    assert SOURCE_ROOT.is_dir(), "The source tree must exist, not pass vacuously."
    offending = [
        str(path.relative_to(REPO_ROOT))
        for path in SOURCE_ROOT.rglob("*.py")
        if "tests" in path.relative_to(SOURCE_ROOT).parts
        or path.name.startswith("test_")
    ]
    assert not offending, offending


def _name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        return _name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def test_scientific_array_fields_have_shape_comments() -> None:
    """Check existing NamedTuples and future dataclasses, without constraining experiment-state mutability."""
    assert SOURCE_ROOT.is_dir()
    missing = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        source = path.read_text()
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not (
                any(_name(base) == "NamedTuple" for base in node.bases)
                or any(_name(dec) == "dataclass" for dec in node.decorator_list)
            ):
                continue
            for field in node.body:
                if not isinstance(field, ast.AnnAssign):
                    continue
                annotation = ast.get_source_segment(source, field.annotation) or ""
                if not any(word in annotation for word in ("Array", "ndarray", "PRNGKey")):
                    continue
                declaration = " ".join(lines[field.lineno - 1:field.end_lineno])
                if "# [" not in declaration:
                    missing.append(f"{path.relative_to(REPO_ROOT)}:{field.lineno}")
    assert not missing, "Array fields need symbolic shape comments:\n" + "\n".join(missing)
