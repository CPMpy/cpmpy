import ast
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import (
    Observer,
    HandlerObserver,
    LoggerObserver,
    IntermediateObjectivesObserver,
    MetadataSidecarObserver,
    HookPipeRelayObserver,
    ResourceLimitObserver,
    SolverArgsObserver,
    RuntimeObserver,
    SolutionCheckerObserver,
    WriteToFileObserver,
    WriteToStdoutObserver,
)


# Map of observer names to classes
# Note: WriteToFileObserver is handled specially because it needs output_file.
OBSERVER_CLASSES = { # Lazy: see load_observer
    "HandlerObserver": HandlerObserver,
    "LoggerObserver": LoggerObserver,
    "ResourceLimitObserver": ResourceLimitObserver,
    "SolverArgsObserver": SolverArgsObserver,
    "RuntimeObserver": RuntimeObserver,
    "SolutionCheckerObserver": SolutionCheckerObserver,
    "IntermediateObjectivesObserver": IntermediateObjectivesObserver,
    "MetadataSidecarObserver": MetadataSidecarObserver,
    "HookPipeRelayObserver": HookPipeRelayObserver,
    "WriteToStdoutObserver": WriteToStdoutObserver,
}

# Aliases for shorter names
OBSERVER_ALIASES = {
    "WriteToFile": "WriteToFileObserver",
    "WriteToStdout": "WriteToStdoutObserver",
    "Competition": "CompetitionPrintingObserver",
    "Handler": "HandlerObserver",
    "Logger": "LoggerObserver",
    "ResourceLimit": "ResourceLimitObserver",
    "SolverArgs": "SolverArgsObserver",
    "Runtime": "RuntimeObserver",
    "SolutionChecker": "SolutionCheckerObserver",
    "IntermediateObjectives": "IntermediateObjectivesObserver",
    "MetadataSidecar": "MetadataSidecarObserver",
    "HookPipeRelay": "HookPipeRelayObserver",
}


def parse_observer_with_args(observer_spec: str) -> tuple[str, Dict[str, Any]]:
    """
    Parse an observer specification that may include constructor arguments.
    """
    paren_pos = observer_spec.rfind("(")
    if paren_pos != -1 and observer_spec.endswith(")"):
        observer_path = observer_spec[:paren_pos]
        args_str = observer_spec[paren_pos + 1 : -1]

        kwargs: Dict[str, Any] = {}
        if args_str.strip():
            try:
                parsed = ast.literal_eval(f"{{{args_str}}}")
                if isinstance(parsed, dict):
                    kwargs = parsed
                else:
                    raise ValueError(f"Invalid argument format: {args_str}. Expected key=value pairs")
            except (ValueError, SyntaxError):
                for pair in args_str.split(","):
                    pair = pair.strip()
                    if "=" not in pair:
                        raise ValueError(f"Invalid argument format: {pair}. Expected 'key=value'")
                    eq_pos = pair.find("=")
                    key = pair[:eq_pos].strip()
                    value = pair[eq_pos + 1 :].strip()
                    try:
                        parsed_value = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        if (value.startswith('"') and value.endswith('"')) or (
                            value.startswith("'") and value.endswith("'")
                        ):
                            parsed_value = value[1:-1]
                        else:
                            parsed_value = value
                    kwargs[key] = parsed_value

        return observer_path, kwargs

    return observer_spec, {}


def load_observer(observer_name: str) -> Observer:
    """
    Load an observer by name/module/file path, optionally with constructor arguments.
    """
    observer_path, kwargs = parse_observer_with_args(observer_name)

    if observer_path in OBSERVER_ALIASES:
        observer_path = OBSERVER_ALIASES[observer_path]

    # File path format: /path/to/file.py:ClassName or path/to/file.py::ClassName
    if "::" in observer_path or ("::" not in observer_path and ".py:" in observer_path):
        if "::" in observer_path:
            file_part, class_name = observer_path.rsplit("::", 1)
        else:
            file_part, class_name = observer_path.rsplit(":", 1)

        if class_name in OBSERVER_ALIASES:
            class_name = OBSERVER_ALIASES[class_name]

        # module.path.file.py::ClassName -> module.path.file
        if ".py" in file_part and not file_part.startswith("/") and not file_part.startswith("."):
            module_path = file_part.replace(".py", "")
            try:
                module = importlib.import_module(module_path)
                observer_class = getattr(module, class_name)
                if not issubclass(observer_class, Observer):
                    raise ValueError(f"{observer_class} is not a subclass of Observer")
                if class_name == "WriteToFileObserver":
                    if "file_path" in kwargs and "output_file" not in kwargs:
                        kwargs["output_file"] = kwargs.pop("file_path")
                    if "output_file" not in kwargs:
                        kwargs["output_file"] = "results/output.txt"
                return observer_class(**kwargs)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Could not load observer '{observer_path}': {e}")

        file_path = Path(file_part).resolve()
        if file_path.exists():
            parent_dir = str(file_path.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            observer_class = getattr(module, class_name)
            if not issubclass(observer_class, Observer):
                raise ValueError(f"{observer_class} is not a subclass of Observer")
            if class_name == "WriteToFileObserver":
                if "file_path" in kwargs and "output_file" not in kwargs:
                    kwargs["output_file"] = kwargs.pop("file_path")
                if "output_file" not in kwargs:
                    kwargs["output_file"] = "results/output.txt"
            return observer_class(**kwargs)

    # Legacy and special handling for WriteToFileObserver.
    if observer_path.startswith("WriteToFileObserver") or observer_path.endswith("WriteToFileObserver"):
        if ":" in observer_path and "::" not in observer_path:
            _, file_path = observer_path.split(":", 1)
            kwargs["output_file"] = file_path
            return WriteToFileObserver(**kwargs)
        if "file_path" in kwargs and "output_file" not in kwargs:
            kwargs["output_file"] = kwargs.pop("file_path")
        if "output_file" not in kwargs:
            kwargs["output_file"] = "results/output.txt"
        return WriteToFileObserver(**kwargs)

    if observer_path in OBSERVER_CLASSES:
        obs_cls = OBSERVER_CLASSES[observer_path]
        return obs_cls(**kwargs)

    if "." in observer_path:
        module_path, class_name = observer_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            observer_class = getattr(module, class_name)
            if not issubclass(observer_class, Observer):
                raise ValueError(f"{observer_class} is not a subclass of Observer")
            if class_name == "WriteToFileObserver":
                if "file_path" in kwargs and "output_file" not in kwargs:
                    kwargs["output_file"] = kwargs.pop("file_path")
                if "output_file" not in kwargs:
                    kwargs["output_file"] = "results/output.txt"
            return observer_class(**kwargs)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Could not load observer '{observer_path}': {e}")

    raise ValueError(f"Unknown observer: {observer_path}. Available: {', '.join(OBSERVER_CLASSES.keys())}")


def load_observers(observer_names: Optional[List[str]]) -> List[Observer]:
    if not observer_names:
        return []
    return [load_observer(name) for name in observer_names]
