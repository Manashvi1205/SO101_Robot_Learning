"""Hierarchical SO-101 research prototype.

Importing this package never opens a camera or connects to a robot.
"""

from .models import PlanStep, TaskPlan
from .registry import SkillRegistry, build_default_registry

__all__ = ["PlanStep", "SkillRegistry", "TaskPlan", "build_default_registry"]
__version__ = "0.2.0"

