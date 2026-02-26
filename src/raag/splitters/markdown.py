"""Markdown-aware document splitter."""

import re
from raag.splitters.base import SplitSection


_HEADING_RE = re.compile(r"^(#{1,6})\s+(?:(\d+(?:\.\d+)*\.?)\s+)?(.+)$")


class MarkdownSplitter:
    """Splits markdown documents by heading structure.

    Each heading starts a new section. The section includes
    all body content until the next heading (any level).
    Hierarchy metadata (parent_path, heading_level) is preserved
    on each section for downstream use.
    """

    def split(self, text: str, max_chars: int = 1000) -> list[SplitSection]:
        lines = text.split("\n")
        sections: list[SplitSection] = []
        hierarchy_stack: list[tuple[int, str]] = []

        current_heading: tuple[int, str | None, str | None, str] | None = None
        current_body_lines: list[str] = []
        current_parent_path: list[str] = []

        for line in lines:
            match = _HEADING_RE.match(line)

            if match:
                # Flush previous section
                if current_heading is not None or current_body_lines:
                    self._flush(
                        current_heading,
                        current_body_lines,
                        current_parent_path,
                        max_chars,
                        sections,
                    )

                level = len(match.group(1))
                number = match.group(2)
                title = match.group(3).strip()

                # Update hierarchy stack
                while hierarchy_stack and hierarchy_stack[-1][0] >= level:
                    hierarchy_stack.pop()

                current_parent_path = [h[1] for h in hierarchy_stack]

                heading_text = f"{number} {title}" if number else title
                hierarchy_stack.append((level, heading_text))

                current_heading = (level, number, title, line.strip())
                current_body_lines = []
            else:
                current_body_lines.append(line)

        # Flush last section
        if current_heading is not None or current_body_lines:
            self._flush(
                current_heading,
                current_body_lines,
                current_parent_path,
                max_chars,
                sections,
            )

        return sections

    def _flush(
        self,
        heading: tuple[int, str | None, str | None, str] | None,
        body_lines: list[str],
        parent_path: list[str],
        max_chars: int,
        sections: list[SplitSection],
    ):
        body_text = "\n".join(body_lines).strip()

        if heading:
            level, number, title, heading_line = heading
            full_text = f"{heading_line}\n\n{body_text}" if body_text else heading_line
        else:
            level, number, title, heading_line = 0, None, None, ""
            full_text = body_text

        if not full_text.strip():
            return

        # Skip heading-only sections (no body content).
        # These are already captured in child sections' parent_path.
        if heading and not body_text:
            return

        if len(full_text) <= max_chars or not body_text:
            sections.append(SplitSection(
                text=full_text,
                heading_level=level,
                section_number=number,
                section_title=title,
                parent_path=parent_path[:],
                chunk_index=0,
                total_chunks=1,
            ))
        else:
            sub_chunks = self._split_long(heading_line, body_text, max_chars)
            for text, idx, total in sub_chunks:
                sections.append(SplitSection(
                    text=text,
                    heading_level=level,
                    section_number=number,
                    section_title=title,
                    parent_path=parent_path[:],
                    chunk_index=idx,
                    total_chunks=total,
                ))

    def _split_long(
        self,
        heading_line: str,
        body_text: str,
        max_chars: int,
    ) -> list[tuple[str, int, int]]:
        paragraphs = [p for p in body_text.split("\n\n") if p.strip()]
        chunks: list[list[str]] = []
        current_paras: list[str] = []
        current_len = len(heading_line)

        for para in paragraphs:
            if current_len + len(para) > max_chars and current_paras:
                chunks.append(current_paras)
                current_paras = []
                current_len = len(heading_line)

            current_paras.append(para)
            current_len += len(para) + 2

        if current_paras:
            chunks.append(current_paras)

        total = len(chunks)
        results = []
        for i, para_group in enumerate(chunks):
            suffix = " (continued)" if i > 0 else ""
            if heading_line:
                text = f"{heading_line}{suffix}\n\n" + "\n\n".join(para_group)
            else:
                text = "\n\n".join(para_group)
            results.append((text, i, total))

        return results
