"""Shared helpers for cleaning LLM-generated CAD code."""

from __future__ import annotations

import re


THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
BUILD_CALL_PATTERN = re.compile(r"(?m)^[ \t]*build\(\).*$")
NOISE_LINE_PATTERN = re.compile(r"^\s*(\d+\.|[-*]|\d+\)|\d+:)")


def remove_reasoning_blocks(text: str) -> str:
    return THINK_BLOCK_PATTERN.sub("", text).strip()


def extract_code_block(text: str) -> str:
    stripped = text.strip()
    fence_start = stripped.find("```")
    if fence_start != -1:
        stripped = stripped[fence_start + 3 :].lstrip()
        if stripped.lower().startswith("python"):
            stripped = stripped[6:].lstrip()
        fence_end = stripped.rfind("```")
        if fence_end != -1:
            stripped = stripped[:fence_end]
    return stripped.strip()


def strip_leading_instructions(text: str) -> str:
    pattern = re.compile(r"^(import\s+cadquery|from\s+cadquery|def\s+build)", re.MULTILINE | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return text[match.start():]
    return text


def ensure_cadquery_import(code: str) -> str:
    normalized = code.replace("\r", "")
    if "import cadquery" in normalized:
        return normalized
    return f"import cadquery as cq\n\n{normalized.lstrip()}"


def _truncate_noise_lines(code: str) -> str:
    lines: list[str] = []
    # We need to be smarter about noise lines.
    # If we are inside a docstring (multiline string), we should NOT truncate on noise.
    
    in_multiline = False
    delimiter = None
    
    for line in code.splitlines():
        # Check state changes
        # This is a simplified check, but likely sufficient for "whole line" noise checks
        # If the line *contains* a triple quote, we toggle state? 
        # It's safer to track state.
        
        # Count triple quotes
        triple_double = line.count('"""')
        triple_single = line.count("'''")
        
        # If we are not in multiline, checking for start
        if not in_multiline:
            if triple_double > 0 and (triple_double % 2 == 1):
                in_multiline = True
                delimiter = '"""'
            elif triple_single > 0 and (triple_single % 2 == 1):
                in_multiline = True
                delimiter = "'''"
        else:
            # In multiline, checking for end
            if delimiter in line:
                # If it ends here?
                # Simplified: just count again. If odd, we likely closed it.
                # But we need to match the specific delimiter.
                count = line.count(delimiter)
                if count > 0:
                     # This is tricky if there are multiple on a line, but usually just one closer.
                     # If we assume it closes:
                     # But wait, we need to know if it closes *before* or *after* potential noise pattern?
                     # Noise pattern `^\s*[-*]` matches things like `  - item`.
                     # Docstrings often contain lists!
                     pass
                
                # We need to verify if we closed it. 
                # Heuristic: if we found the delimiter, assume closed for the NEXT line?
                # Or closed on THIS line?
                # If closed on this line, the noise pattern match would be irrelevant for this line (since it contains quotes).
                # The noise pattern `^\s*(\d+\.|[-*]|\d+\)|\d+:)` is unlikely to match `  """` line unless it is `  - """` which is weird.
                
                # So if we are in multiline, we IGNORE noise check.
                # We just need to know when we exit.
                if line.rstrip().endswith(delimiter):
                    in_multiline = False
                    delimiter = None
        
        # If in multiline, do NOT truncate
        if in_multiline:
            lines.append(line)
            continue
            
        # Check for noise
        if NOISE_LINE_PATTERN.match(line):
            # Double check it's not valid code?
            # Code shouldn't start with `-` or `*`.
            # `1.` is invalid syntax in Python unless in string.
            # But wait, `1 + 1`? No, `1.` matches `\d+\.`. `1.` is float.
            # `1.0` is float.
            # `1.` at start of line is valid float.
            # `NOISE_LINE_PATTERN` is `^\s*(\d+\.|[-*]|\d+\)|\d+:)`
            # `1.` matches `\d+\.`.
            # So `x = 1.` is fine. `1.` on its own line is fine (expression statement).
            # But LLMs often output lists: `1. Step one`.
            
            # We should probably only truncate if it looks like a list item AND we are at the end of the file?
            # Or rely on `_collect_imports_and_build` to handle structure.
            # `_truncate_noise_lines` is aggressive.
            
            # Given `_collect_imports_and_build` is now smarter about parsing `def build():`, maybe we can RELAX `_truncate_noise_lines`?
            # Or remove it entirely and rely on `_collect_imports_and_build` extracting the function?
            # `_collect_imports_and_build` will stop at unindented noise lines.
            # But noise lines INSIDE docstrings were the issue.
            
            # If we just SKIP `_truncate_noise_lines`, `_collect_imports_and_build` handles the rest?
            # `sanitize_cad_code` calls `_truncate_noise_lines` BEFORE `_collect_imports_and_build`.
            # If I remove `_truncate_noise_lines`, `_collect_imports_and_build` will see:
            # def build():
            #     """
            #     ...
            #     - item
            #     ...
            #     """
            # It will correctly include `- item` because it's indented and inside `in_multiline_string`.
            
            # But what about noise AFTER the code?
            # def build(): ...
            # 
            # 1. Explanation...
            
            # `_collect_imports_and_build` stops when it sees `1. Explanation...` (unindented).
            # So `_truncate_noise_lines` might be REDUNDANT or HARMFUL if it triggers inside docstrings.
            
            # However, what if the noise is:
            # def build():
            #    ...
            #    return res
            # 
            # Explanation:
            # - point 1
            
            # `_collect_imports_and_build` stops at `Explanation:` (unindented).
            # So we seem safe to REMOVE or WEAKEN `_truncate_noise_lines`.
            
            # Let's try removing it from the pipeline?
            pass
        
        if NOISE_LINE_PATTERN.match(line):
             break
        lines.append(line)
    return "\n".join(lines)


def _collect_imports_and_build(code: str) -> str:
    imports: list[str] = []
    build_lines: list[str] = []
    in_build = False
    
    # Simple state machine to track if we are inside a multiline string
    in_multiline_string = False
    multiline_delimiter = None  # will be ''' or """

    for line in code.splitlines():
        stripped = line.lstrip()
        
        # Check for imports if we haven't started building yet
        if not in_build:
            if stripped.startswith("import") or stripped.startswith("from"):
                imports.append(line)
                continue
            if stripped.startswith("def build"):
                in_build = True
                build_lines.append(line)
                continue
            # Ignore lines before build that aren't imports
            continue

        if in_build:
            # If we are in a multiline string, we accept anything
            if in_multiline_string:
                build_lines.append(line)
                # Check if it closes
                # This is a basic check; it might fail on lines like: x = """ string """
                # But usually docstrings are """\n ... \n"""
                if multiline_delimiter in line:
                    # We might need to be more careful about counting occurrences or position
                    # But for now, if the line contains the delimiter, we assume it might close it.
                    # A better check: count occurrences of the delimiter in the line.
                    count = line.count(multiline_delimiter)
                    # If we started this block on a previous line, and we find an odd number of delimiters (usually 1), we close.
                    # If we find even, it might be "foo" + """bar""" -> open again? No.
                    # Let's assume standard docstring usage:
                    # """
                    # text
                    # """
                    if count > 0:
                        # Logic is tricky without full parsing. 
                        # Heuristic: if the line *ends* with delimiter, it likely closes.
                        if line.rstrip().endswith(multiline_delimiter):
                             in_multiline_string = False
                             multiline_delimiter = None
                continue

            # Not in multiline string
            # Check if this line starts a multiline string
            if '"""' in line:
                # Check if it's self-contained? e.g. x = """foo"""
                if line.count('"""') % 2 == 1:
                    in_multiline_string = True
                    multiline_delimiter = '"""'
            elif "'''" in line:
                if line.count("'''") % 2 == 1:
                    in_multiline_string = True
                    multiline_delimiter = "'''"

            # Indentation check
            if in_multiline_string:
                 build_lines.append(line)
                 continue

            if line.startswith(" ") or line.startswith("\t") or line.strip() == "":
                build_lines.append(line)
                continue
            
            # Stop when leaving the build block (unindented and not in multiline string)
            break

    if not build_lines:
        build_lines = [
            "def build():",
            "    return cq.Workplane()",
        ]
    merged: list[str] = []
    if imports:
        merged.extend(imports)
        merged.append("")
    merged.extend(build_lines)
    return "\n".join(merged)


def sanitize_cad_code(text: str) -> str:
    stripped = remove_reasoning_blocks(text)
    stripped = extract_code_block(stripped)
    stripped = strip_leading_instructions(stripped)
    stripped = stripped.replace("```", "").strip()
    sanitized = ensure_cadquery_import(stripped)
    match = BUILD_CALL_PATTERN.search(sanitized)
    if match:
        sanitized = sanitized[: match.start()].rstrip()
    # sanitized = _truncate_noise_lines(sanitized) # Removed as it triggers inside docstrings
    sanitized = sanitized.strip()
    sanitized = _collect_imports_and_build(sanitized)
    return sanitized
