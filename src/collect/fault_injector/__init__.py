"""Synthetic build-failure fault injector for LaunchForge AI."""

from src.collect.fault_injector.faults import (
    FAULT_REGISTRY,
    BaseFault,
    FaultApplication,
    RemoveDependencyFault,
    TypeScriptTypeErrorFault,
    WrongImportPathFault,
    get_fault,
    register_fault,
)

__all__ = [
    "BaseFault",
    "FAULT_REGISTRY",
    "FaultApplication",
    "RemoveDependencyFault",
    "TypeScriptTypeErrorFault",
    "WrongImportPathFault",
    "get_fault",
    "register_fault",
]
