from __future__ import annotations

import nox


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    session.install(".[tests]")
    session.run("pytest", *session.posargs)


@nox.session
def coverage_pytest(session: nox.Session) -> None:
    """
    Run and give a coverage report with pytest
    """
    session.install(".[tests]")
    session.run("coverage", "run", "-m", "pytest", "-vv")
    session.run("coverage", "report", "-m")
    session.run("coverage", "html")


@nox.session
def coverage_unittest(session: nox.Session) -> None:
    """
    Run unittest via coverage
    """
    session.install(".[tests]")
    session.run("coverage", "run", "-m", "unittest", "discover")
    session.run("coverage", "report", "-m")
    session.run("coverage", "html")


@nox.session
def unittest(session: nox.Session) -> None:
    """
    Run unittest
    """
    session.install(".[tests]")
    session.run("python", "-m", "unittest", "discover", "-v")


@nox.session
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass "--serve" to serve.
    """
    session.install(".[docs]")
    session.install("sphinx_rtd_theme")
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", ".", "build")
