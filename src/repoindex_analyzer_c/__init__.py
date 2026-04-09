"""C language analyzer backed by tree-sitter.

Responsibilities
----------------
- Initialize the tree-sitter C parser and derive module and function stable IDs.
- Walk parse nodes to extract imports, declarations, call sites, and normalized artifacts.
- Translate the collected metadata into `AnalysisResult` objects for persistence.

Design principles
-----------------
The analyzer confines tree-sitter interaction to this module so language-specific logic stays isolated and deterministic.

Architectural role
------------------
This module belongs to the **language analyzer layer** and implements the C-family analysis path for ADR-004.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from repoindex.contracts import LanguageAnalyzer

from tree_sitter import Language, Node, Parser
from tree_sitter_c import language

from repoindex.models import (
    AnalysisResult,
    CallSite,
    DeclarationArtifact,
    DeclarationKind,
    FunctionArtifact,
    ImportArtifact,
    ImportKind,
    ModuleArtifact,
)

_C_SUFFIXES = {".c", ".h"}
_LANGUAGE = Language(language())
__all__ = ["CAnalyzer", "_disambiguate_function_stable_ids", "build_analyzer"]
_TYPE_LIKE_NAMES = {
    "bool",
    "char",
    "double",
    "float",
    "int",
    "long",
    "short",
    "size_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "void",
}


def _new_parser() -> Parser:
    """
    Create a parser configured for the C grammar.

    Parameters
    ----------
    None

    Returns
    -------
    tree_sitter.Parser
        Parser configured for ``tree-sitter-c``.
    """
    return Parser(_LANGUAGE)


def _decode_source_text(source: bytes) -> str:
    """
    Decode one source fragment with a deterministic legacy fallback.

    Parameters
    ----------
    source : bytes
        Raw source bytes to decode.

    Returns
    -------
    str
        Text decoded as UTF-8 when possible, otherwise Latin-1.
    """
    try:
        return source.decode("utf-8")
    except UnicodeDecodeError:
        return source.decode("latin-1")


def _module_name_for_path(path: Path, root: Path) -> str:
    """
    Derive the logical module name for one C source path.

    Parameters
    ----------
    path : pathlib.Path
        Source file being analyzed.
    root : pathlib.Path
        Repository root used for relative module naming.

    Returns
    -------
    str
        Dotted module identity derived from the relative file path.
    """
    relative = path.relative_to(root).with_suffix("")
    return ".".join(relative.parts)


def _module_stable_id(path: Path, root: Path) -> str:
    """
    Build the durable identity for one C-family module.

    Parameters
    ----------
    path : pathlib.Path
        Source path being analyzed.
    root : pathlib.Path
        Repository root used for relative identity derivation.

    Returns
    -------
    str
        Durable C-family module identity.
    """
    return f"c:module:{path.relative_to(root).as_posix()}"


def _symbol_owner_id(path: Path, root: Path) -> str:
    """
    Build the file-scoped owner identity for C symbols.

    Parameters
    ----------
    path : pathlib.Path
        Source path being analyzed.
    root : pathlib.Path
        Repository root used for relative identity derivation.

    Returns
    -------
    str
        Repo-relative owner identity that preserves the file suffix.
    """
    return path.relative_to(root).as_posix()


def _function_stable_id(owner_id: str, function_name: str) -> str:
    """
    Build the durable identity for one C function.

    Parameters
    ----------
    owner_id : str
        File-scoped owner identity preserving the source suffix.
    function_name : str
        Unqualified function name.

    Returns
    -------
    str
        Durable C function identity.
    """
    return f"c:function:{owner_id}:{function_name}"


def _declaration_stable_id(
    owner_id: str,
    kind: DeclarationKind,
    declaration_name: str,
) -> str:
    """
    Build the durable identity for one top-level C declaration.

    Parameters
    ----------
    owner_id : str
        File-scoped owner identity preserving the source suffix.
    kind : {"struct", "enum", "typedef"}
        Stable declaration classifier.
    declaration_name : str
        Exposed declaration name.

    Returns
    -------
    str
        Durable C declaration identity.
    """
    return f"c:{kind}:{owner_id}:{declaration_name}"


def _node_text(node: Node, source: bytes) -> str:
    """
    Decode the source text owned by one syntax node.

    Parameters
    ----------
    node : tree_sitter.Node
        Syntax node whose text should be decoded.
    source : bytes
        Full source buffer.

    Returns
    -------
    str
        Decoded node text with a deterministic legacy fallback.
    """
    return _decode_source_text(source[node.start_byte : node.end_byte])


def _comment_to_summary(text: str) -> str | None:
    """
    Normalize one raw C comment block into summary text.

    Parameters
    ----------
    text : str
        Raw comment text including delimiters.

    Returns
    -------
    str | None
        Normalized summary text, or ``None`` when no content remains.
    """
    stripped = text.strip()
    if stripped.startswith("/*"):
        body = stripped.removeprefix("/*").removesuffix("*/")
        lines = [line.strip().lstrip("*").strip() for line in body.splitlines()]
    else:
        lines = [
            line.strip().removeprefix("//").strip() for line in stripped.splitlines()
        ]

    normalized_lines = [line for line in lines if line]
    if not normalized_lines:
        return None
    return "\n".join(normalized_lines)


def _leading_module_comment(root: Node, source: bytes) -> str | None:
    """
    Extract the first leading file comment as module summary text.

    Parameters
    ----------
    root : tree_sitter.Node
        Translation-unit root node.
    source : bytes
        Full source buffer.

    Returns
    -------
    str | None
        Normalized leading comment summary, or ``None`` when absent.
    """
    for child in root.children:
        if child.type == "comment":
            return _comment_to_summary(_node_text(child, source))
        if child.type != "preproc_include":
            return None
    return None


def _attached_comment_map(root: Node, source: bytes) -> dict[int, str]:
    """
    Map declaration start bytes to nearby leading comment summaries.

    Parameters
    ----------
    root : tree_sitter.Node
        Translation-unit root node.
    source : bytes
        Full source buffer.

    Returns
    -------
    dict[int, str]
        Attached comment summaries keyed by declaration start byte.
    """
    attached: dict[int, str] = {}
    pending_comment: str | None = None
    pending_end_row: int | None = None

    for child in root.children:
        if child.type == "comment":
            pending_comment = _comment_to_summary(_node_text(child, source))
            pending_end_row = child.end_point.row
            continue

        if pending_comment is not None and pending_end_row is not None:
            if child.start_point.row - pending_end_row <= 2:
                attached[child.start_byte] = pending_comment
            pending_comment = None
            pending_end_row = None

    return attached


def _named_descendants(node: Node) -> list[Node]:
    """
    Collect named descendants of one syntax node in source order.

    Parameters
    ----------
    node : tree_sitter.Node
        Parent syntax node.

    Returns
    -------
    list[tree_sitter.Node]
        Named descendant nodes in deterministic source order.
    """
    descendants: list[Node] = []
    stack = list(reversed(node.named_children))

    while stack:
        current = stack.pop()
        descendants.append(current)
        stack.extend(reversed(current.named_children))

    return descendants


def _unwrap_declarator_name(node: Node, source: bytes) -> str | None:
    """
    Resolve the identifier owned by one declarator node.

    Parameters
    ----------
    node : tree_sitter.Node
        Declarator node that may nest pointers or arrays.
    source : bytes
        Full source buffer.

    Returns
    -------
    str | None
        Identifier text when resolvable.
    """
    if node.type == "function_declarator":
        function_name = _function_declarator_name(node, source)
        if function_name is not None:
            return function_name

    if node.type in {"identifier", "field_identifier"}:
        return _node_text(node, source)

    child = node.child_by_field_name("declarator")
    if child is not None:
        return _unwrap_declarator_name(child, source)

    for named_child in node.named_children:
        name = _unwrap_declarator_name(named_child, source)
        if name is not None:
            return name
    return None


def _function_declarator_name(node: Node, source: bytes) -> str | None:
    """
    Resolve the callable identifier owned by one function declarator.

    Parameters
    ----------
    node : tree_sitter.Node
        Function declarator that may include annotations or parse errors.
    source : bytes
        Full source buffer.

    Returns
    -------
    str | None
        Callable identifier text when resolvable.
    """
    direct_declarator = node.child_by_field_name("declarator")
    if direct_declarator is not None and direct_declarator.type == "identifier":
        direct_name = _node_text(direct_declarator, source)
        error_name = _error_identifier_name(node, source)
        if error_name is not None and _looks_like_macro_or_type_name(direct_name):
            return error_name

    macro_wrapped_name = _function_macro_wrapper_name(node, source)
    if macro_wrapped_name is not None:
        return macro_wrapped_name

    if direct_declarator is None or direct_declarator.type != "identifier":
        return None
    if not _looks_like_annotation_macro_name(_node_text(direct_declarator, source)):
        return None
    return _annotated_function_call_name(node, source)


def _annotated_function_call_name(node: Node, source: bytes) -> str | None:
    """
    Resolve a callable name from a macro-annotated call-like declarator.

    Parameters
    ----------
    node : tree_sitter.Node
        Function declarator whose real callable name may appear in a nested
        call expression.
    source : bytes
        Full source buffer.

    Returns
    -------
    str | None
        Callable identifier text when the annotation pattern is present.
    """
    for named_child in reversed(node.named_children):
        if named_child.type != "call_expression":
            continue
        arguments = named_child.child_by_field_name("arguments")
        if arguments is None or not arguments.named_children:
            continue
        function_node = named_child.child_by_field_name("function")
        if function_node is None:
            continue
        name = _unwrap_declarator_name(function_node, source)
        if name is not None:
            return name
    return None


def _looks_like_macro_or_type_name(name: str) -> bool:
    """
    Decide whether one declarator token looks like a macro or a type name.

    Parameters
    ----------
    name : str
        Candidate declarator token.

    Returns
    -------
    bool
        ``True`` when the token is unlikely to be the real callable name.
    """
    if name in _TYPE_LIKE_NAMES or name.endswith("_t"):
        return True
    return _looks_like_annotation_macro_name(name)


def _looks_like_annotation_macro_name(name: str) -> bool:
    """
    Decide whether one declarator token looks like an annotation macro.

    Parameters
    ----------
    name : str
        Candidate declarator token.

    Returns
    -------
    bool
        ``True`` when the token is all-uppercase or underscore-style text.
    """
    has_alpha = any(char.isalpha() for char in name)
    if not has_alpha:
        return False
    return all(not char.isalpha() or char.isupper() for char in name)


def _error_identifier_name(node: Node, source: bytes) -> str | None:
    """
    Extract one identifier-like token from an ``ERROR`` child when present.

    Parameters
    ----------
    node : tree_sitter.Node
        Declarator node that may include a parse-error placeholder.
    source : bytes
        Full source buffer.

    Returns
    -------
    str | None
        Recovered identifier text when the ``ERROR`` child looks usable.
    """
    for named_child in node.named_children:
        if named_child.type != "ERROR":
            continue
        text = _node_text(named_child, source).strip()
        if not text:
            continue
        if "::" in text or "(" in text or ")" in text or "[" in text or "]" in text:
            continue
        if not (text[0].isalpha() or text[0] == "_"):
            continue
        if any(not (char.isalnum() or char == "_") for char in text):
            continue
        return text
    return None


def _function_macro_wrapper_name(node: Node, source: bytes) -> str | None:
    """
    Resolve a function name wrapped by a macro-style declarator.

    Parameters
    ----------
    node : tree_sitter.Node
        Function declarator that may wrap the real name in a macro call.
    source : bytes
        Full source buffer.

    Returns
    -------
    str | None
        Wrapped function name when the declarator matches the macro pattern.
    """
    nested = node.child_by_field_name("declarator")
    if nested is None or nested.type != "function_declarator":
        return None

    parameter_list = nested.child_by_field_name("parameters")
    if parameter_list is None or len(parameter_list.named_children) != 1:
        return None

    parameter = parameter_list.named_children[0]
    if parameter.type != "parameter_declaration":
        return None
    if parameter.child_by_field_name("declarator") is not None:
        return None

    for named_child in parameter.named_children:
        if named_child.type in {"identifier", "type_identifier", "field_identifier"}:
            return _node_text(named_child, source)
    return None


def _extract_parameter_names(
    parameter_list: Node | None,
    source: bytes,
) -> tuple[str, ...]:
    """
    Extract deterministic parameter names from one parameter list.

    Parameters
    ----------
    parameter_list : tree_sitter.Node | None
        Parameter list node from a function declarator.
    source : bytes
        Full source buffer.

    Returns
    -------
    tuple[str, ...]
        Parameter names in declaration order.
    """
    if parameter_list is None:
        return ()

    parameters: list[str] = []
    for child in parameter_list.named_children:
        if child.type != "parameter_declaration":
            continue
        declarator = child.child_by_field_name("declarator")
        if declarator is None:
            continue
        name = _unwrap_declarator_name(declarator, source)
        if name is not None:
            parameters.append(name)
    return tuple(parameters)


def _find_parameter_list(node: Node | None) -> Node | None:
    """
    Find the parameter-list node nested inside one declarator tree.

    Parameters
    ----------
    node : tree_sitter.Node | None
        Declarator subtree that may own a function declarator.

    Returns
    -------
    tree_sitter.Node | None
        Nested ``parameter_list`` node when present.
    """
    if node is None:
        return None
    if node.type == "parameter_list":
        return node

    parameter_list = node.child_by_field_name("parameters")
    if parameter_list is not None:
        return parameter_list

    declarator = node.child_by_field_name("declarator")
    if declarator is not None:
        nested = _find_parameter_list(declarator)
        if nested is not None:
            return nested

    for child in node.named_children:
        nested = _find_parameter_list(child)
        if nested is not None:
            return nested
    return None


def _is_supported_function_definition(node: Node) -> bool:
    """
    Decide whether one parsed function-definition node is semantically usable.

    Parameters
    ----------
    node : tree_sitter.Node
        Candidate top-level ``function_definition`` node.

    Returns
    -------
    bool
        ``True`` when the node exposes a callable declarator with parameters
        and contains no top-level parse error markers.
    """
    if any(child.type == "ERROR" for child in node.named_children):
        return False

    declarator = node.child_by_field_name("declarator")
    return _find_parameter_list(declarator) is not None


def _call_site_from_expression(node: Node, source: bytes) -> CallSite | None:
    """
    Convert one tree-sitter call expression into a normalized call record.

    Parameters
    ----------
    node : tree_sitter.Node
        Call-expression node.
    source : bytes
        Full source buffer.

    Returns
    -------
    repoindex.models.CallSite | None
        Normalized call record, or ``None`` when no supported target exists.
    """
    function_node = node.child_by_field_name("function")
    if function_node is None:
        return None

    if function_node.type == "identifier":
        return CallSite(
            kind="name",
            target=_node_text(function_node, source),
            lineno=function_node.start_point.row + 1,
            col_offset=function_node.start_point.column,
        )

    if function_node.type == "field_expression":
        receiver = function_node.child_by_field_name("argument")
        field = function_node.child_by_field_name("field")
        if receiver is None or field is None:
            return None
        return CallSite(
            kind="attribute",
            target=_node_text(field, source),
            lineno=field.start_point.row + 1,
            col_offset=field.start_point.column,
            base=_node_text(receiver, source),
        )

    return CallSite(
        kind="unresolved",
        target="",
        lineno=function_node.start_point.row + 1,
        col_offset=function_node.start_point.column,
    )


def _extract_calls(body: Node | None, source: bytes) -> tuple[CallSite, ...]:
    """
    Extract normalized calls from one function body.

    Parameters
    ----------
    body : tree_sitter.Node | None
        Compound-statement node owning the function body.
    source : bytes
        Full source buffer.

    Returns
    -------
    tuple[repoindex.models.CallSite, ...]
        Call records in deterministic source order.
    """
    if body is None:
        return ()

    calls: list[CallSite] = []
    for node in _named_descendants(body):
        if node.type != "call_expression":
            continue
        call = _call_site_from_expression(node, source)
        if call is not None:
            calls.append(call)
    return tuple(calls)


def _returns_value(body: Node | None) -> int:
    """
    Detect whether one function body contains a value-returning statement.

    Parameters
    ----------
    body : tree_sitter.Node | None
        Compound-statement node owning the function body.

    Returns
    -------
    int
        ``1`` when the body contains ``return <expr>;``, else ``0``.
    """
    if body is None:
        return 0

    for node in _named_descendants(body):
        if node.type == "return_statement" and len(node.named_children) > 0:
            return 1
    return 0


def _extract_functions(
    root: Node,
    source: bytes,
    *,
    module_name: str,
    owner_id: str,
) -> tuple[FunctionArtifact, ...]:
    """
    Extract top-level C function definitions from one translation unit.

    Parameters
    ----------
    root : tree_sitter.Node
        Translation-unit root node.
    source : bytes
        Full source buffer.
    module_name : str
        Dotted owner module name.
    owner_id : str
        File-scoped owner identity preserving the source suffix.

    Returns
    -------
    tuple[repoindex.models.FunctionArtifact, ...]
        Deterministic function artifacts ordered by source position.
    """
    functions: list[FunctionArtifact] = []

    for child in root.children:
        if child.type != "function_definition":
            continue
        if not _is_supported_function_definition(child):
            continue

        declarator = child.child_by_field_name("declarator")
        body = child.child_by_field_name("body")
        if declarator is None:
            continue

        name = _unwrap_declarator_name(declarator, source)
        if name is None:
            continue

        parameter_list = _find_parameter_list(declarator)
        parameters = _extract_parameter_names(parameter_list, source)
        signature_end = body.start_byte if body is not None else declarator.end_byte
        signature = _decode_source_text(
            source[child.start_byte : signature_end]
        ).strip()
        is_public = int(
            not any(
                sub.type == "storage_class_specifier"
                and _node_text(sub, source) == "static"
                for sub in child.children
            )
        )

        functions.append(
            FunctionArtifact(
                name=name,
                stable_id=_function_stable_id(owner_id, name),
                lineno=child.start_point.row + 1,
                end_lineno=body.end_point.row + 1 if body is not None else None,
                signature=" ".join(signature.split()),
                docstring=None,
                has_docstring=0,
                is_method=0,
                is_public=is_public,
                parameters=parameters,
                returns_value=_returns_value(body),
                yields_value=0,
                raises=0,
                has_asserts=0,
                decorators=(),
                calls=_extract_calls(body, source),
                callable_refs=(),
            )
        )

    return _disambiguate_function_stable_ids(tuple(functions))


def _disambiguate_function_stable_ids(
    functions: tuple[FunctionArtifact, ...],
) -> tuple[FunctionArtifact, ...]:
    """
    Disambiguate duplicate C function stable IDs within one file analysis.

    Parameters
    ----------
    functions : tuple[repoindex.models.FunctionArtifact, ...]
        Extracted function artifacts in source order.

    Returns
    -------
    tuple[repoindex.models.FunctionArtifact, ...]
        Function artifacts with duplicate stable IDs rewritten deterministically.
    """
    counts: dict[str, int] = {}
    for function in functions:
        counts[function.stable_id] = counts.get(function.stable_id, 0) + 1

    if all(count == 1 for count in counts.values()):
        return functions

    used_ids: set[str] = set()
    disambiguated: list[FunctionArtifact] = []
    for function in functions:
        stable_id = function.stable_id
        if counts[stable_id] > 1:
            digest = hashlib.sha1(function.signature.encode("utf-8")).hexdigest()[:12]
            stable_id = f"{stable_id}:{digest}"
            if stable_id in used_ids:
                stable_id = f"{stable_id}:{function.lineno}"
            function = replace(function, stable_id=stable_id)
        used_ids.add(stable_id)
        disambiguated.append(function)
    return tuple(disambiguated)


def _declaration_name(node: Node, source: bytes) -> str | None:
    """
    Resolve the exposed declaration name for one top-level type node.

    Parameters
    ----------
    node : tree_sitter.Node
        Declaration node being normalized.
    source : bytes
        Full source buffer.

    Returns
    -------
    str | None
        Declaration name when one is present.
    """
    if node.type == "type_definition":
        named_children = list(node.named_children)
        if not named_children:
            return None
        alias_node = named_children[-1]
        if alias_node.type in {"type_identifier", "identifier", "primitive_type"}:
            return _node_text(alias_node, source)
        return None

    for named_child in node.named_children:
        if named_child.type in {"type_identifier", "identifier"}:
            return _node_text(named_child, source)
    return None


def _resolve_declaration_docstring(
    attached_comments: dict[int, str],
    node: Node,
    inherited_comment: str | None = None,
) -> str | None:
    """
    Resolve the best docstring to attach to one declaration node.

    Parameters
    ----------
    attached_comments : dict[int, str]
        Leading comment summaries keyed by declaration start byte.
    node : tree_sitter.Node
        Declaration node whose docstring should be resolved.
    inherited_comment : str | None, optional
        Leading comment summary inherited from an owning declaration node.

    Returns
    -------
    str | None
        Attached comment when present, otherwise the inherited comment.
    """
    return attached_comments.get(node.start_byte, inherited_comment)


def _declaration_artifact(
    node: Node,
    source: bytes,
    *,
    docstring: str | None,
    kind: DeclarationKind,
    owner_id: str,
) -> DeclarationArtifact | None:
    """
    Build one normalized declaration artifact when the node is named.

    Parameters
    ----------
    node : tree_sitter.Node
        Declaration node being normalized.
    source : bytes
        Full source buffer.
    docstring : str | None
        Docstring to attach to the declaration.
    kind : str
        Stable declaration classifier.
    owner_id : str
        File-scoped owner identity preserving the source suffix.

    Returns
    -------
    repoindex.models.DeclarationArtifact | None
        Normalized declaration artifact, or ``None`` when no usable name is
        present.
    """
    name = _declaration_name(node, source)
    if name is None:
        return None

    return DeclarationArtifact(
        name=name,
        stable_id=_declaration_stable_id(owner_id, kind, name),
        kind=kind,
        lineno=node.start_point.row + 1,
        signature=" ".join(_node_text(node, source).split()),
        docstring=docstring,
    )


def _extract_declarations(
    root: Node,
    source: bytes,
    *,
    owner_id: str,
) -> tuple[DeclarationArtifact, ...]:
    """
    Extract top-level C declarations useful for exact and semantic lookup.

    Parameters
    ----------
    root : tree_sitter.Node
        Translation-unit root node.
    source : bytes
        Full source buffer.
    owner_id : str
        File-scoped owner identity preserving the source suffix.

    Returns
    -------
    tuple[repoindex.models.DeclarationArtifact, ...]
        Deterministic declaration artifacts ordered by source position.
    """
    declarations_by_stable_id: dict[str, DeclarationArtifact] = {}
    attached_comments = _attached_comment_map(root, source)

    for child in root.children:
        if child.type == "struct_specifier":
            declaration = _declaration_artifact(
                child,
                source,
                docstring=_resolve_declaration_docstring(attached_comments, child),
                kind="struct",
                owner_id=owner_id,
            )
            if declaration is not None:
                declarations_by_stable_id[declaration.stable_id] = declaration
            continue

        if child.type == "enum_specifier":
            declaration = _declaration_artifact(
                child,
                source,
                docstring=_resolve_declaration_docstring(attached_comments, child),
                kind="enum",
                owner_id=owner_id,
            )
            if declaration is not None:
                declarations_by_stable_id[declaration.stable_id] = declaration
            continue

        if child.type != "type_definition":
            continue

        child_comment = attached_comments.get(child.start_byte)
        for named_child in child.named_children:
            if named_child.type == "struct_specifier":
                declaration = _declaration_artifact(
                    named_child,
                    source,
                    docstring=_resolve_declaration_docstring(
                        attached_comments,
                        named_child,
                        inherited_comment=child_comment,
                    ),
                    kind="struct",
                    owner_id=owner_id,
                )
                if declaration is not None:
                    declarations_by_stable_id[declaration.stable_id] = declaration
            elif named_child.type == "enum_specifier":
                declaration = _declaration_artifact(
                    named_child,
                    source,
                    docstring=_resolve_declaration_docstring(
                        attached_comments,
                        named_child,
                        inherited_comment=child_comment,
                    ),
                    kind="enum",
                    owner_id=owner_id,
                )
                if declaration is not None:
                    declarations_by_stable_id[declaration.stable_id] = declaration

        declaration = _declaration_artifact(
            child,
            source,
            docstring=_resolve_declaration_docstring(attached_comments, child),
            kind="typedef",
            owner_id=owner_id,
        )
        if declaration is not None:
            declarations_by_stable_id[declaration.stable_id] = declaration

    return tuple(
        sorted(
            declarations_by_stable_id.values(),
            key=lambda artifact: artifact.lineno,
        )
    )


def _extract_imports(root: Node, source: bytes) -> tuple[ImportArtifact, ...]:
    """
    Extract include rows from one translation unit.

    Parameters
    ----------
    root : tree_sitter.Node
        Translation-unit root node.
    source : bytes
        Full source buffer.

    Returns
    -------
    tuple[repoindex.models.ImportArtifact, ...]
        Deterministic include rows ordered by source position.
    """
    imports: list[ImportArtifact] = []

    for child in root.children:
        if child.type != "preproc_include":
            continue
        include_target = None
        include_kind: ImportKind = "include_local"
        for named_child in child.named_children:
            if named_child.type == "string_literal":
                include_target = _node_text(named_child, source).strip('"')
                include_kind = "include_local"
                break
            if named_child.type == "system_lib_string":
                include_target = _node_text(named_child, source).strip("<>")
                include_kind = "include_system"
                break
        if include_target is None:
            continue
        imports.append(
            ImportArtifact(
                name=include_target,
                alias=None,
                lineno=child.start_point.row + 1,
                kind=include_kind,
            )
        )

    return tuple(imports)


class CAnalyzer:
    """
    Concrete C analyzer for repository indexing.

    Parameters
    ----------
    None

    Notes
    -----
    This analyzer is backed by ``tree-sitter-c`` so further C extraction work
    can evolve from a real parse tree instead of regex heuristics.
    """

    name = "c"
    version = "2"
    discovery_globs: tuple[str, ...] = ("*.c", "*.h")

    def supports_path(self, path: Path) -> bool:
        """
        Decide whether the analyzer accepts a C-family source path.

        Parameters
        ----------
        path : pathlib.Path
            Candidate repository file.

        Returns
        -------
        bool
            ``True`` when the file is a ``.c`` or ``.h`` source file.
        """
        return path.suffix in _C_SUFFIXES

    def analyze_file(self, path: Path, root: Path) -> AnalysisResult:
        """
        Analyze one C-family source file into normalized artifacts.

        Parameters
        ----------
        path : pathlib.Path
            C-family source file to analyze.
        root : pathlib.Path
            Repository root used for module-name derivation.

        Returns
        -------
        repoindex.models.AnalysisResult
            Normalized analysis result for the file.
        """
        source = path.read_bytes()
        root_node = _new_parser().parse(source).root_node
        module_comment = _leading_module_comment(root_node, source)
        module_name = _module_name_for_path(path, root)
        owner_id = _symbol_owner_id(path, root)
        return AnalysisResult(
            source_path=path,
            module=ModuleArtifact(
                name=module_name,
                stable_id=_module_stable_id(path, root),
                docstring=module_comment,
                has_docstring=int(module_comment is not None),
            ),
            classes=(),
            functions=_extract_functions(
                root_node,
                source,
                module_name=module_name,
                owner_id=owner_id,
            ),
            declarations=_extract_declarations(
                root_node,
                source,
                owner_id=owner_id,
            ),
            imports=_extract_imports(root_node, source),
        )


def build_analyzer() -> LanguageAnalyzer:
    """
    Build the first-party C analyzer plugin instance.

    Parameters
    ----------
    None

    Returns
    -------
    repoindex.contracts.LanguageAnalyzer
        First-party C analyzer instance.
    """
    return CAnalyzer()
