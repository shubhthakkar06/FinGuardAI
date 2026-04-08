from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PAGE_WIDTH = 1654
PAGE_HEIGHT = 2339
MARGIN_X = 120
MARGIN_Y = 120
CONTENT_WIDTH = PAGE_WIDTH - (2 * MARGIN_X)
BOTTOM_LIMIT = PAGE_HEIGHT - MARGIN_Y

BASE_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "docs"
SOURCE_MD = DOCS_DIR / "FinGuardAI_Report.md"
OUTPUT_PDF = DOCS_DIR / "FinGuardAI_Report.pdf"


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


BODY_FONT = load_font("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
BODY_BOLD_FONT = load_font("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
H1_FONT = load_font("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 42)
H2_FONT = load_font("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 34)
H3_FONT = load_font("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 30)
H4_FONT = load_font("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 26)
CODE_FONT = load_font("/System/Library/Fonts/Supplemental/Courier New.ttf", 21)
TABLE_FONT = load_font("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
TABLE_BOLD_FONT = load_font("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)

TEXT_COLOR = (20, 20, 20)
MUTED_COLOR = (90, 90, 90)
GRID_COLOR = (180, 180, 180)
SHADE_COLOR = (245, 247, 250)


@dataclass
class Block:
    kind: str
    value: object


class PdfRenderer:
    def __init__(self) -> None:
        self.pages: list[Image.Image] = []
        self.page = None
        self.draw = None
        self.y = MARGIN_Y
        self.keep_next_image_on_page = False
        self.new_page()

    def new_page(self) -> None:
        self.page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
        self.draw = ImageDraw.Draw(self.page)
        self.pages.append(self.page)
        self.y = MARGIN_Y

    def force_page_break(self) -> None:
        if self.y > MARGIN_Y + 5:
            self.new_page()

    def ensure_space(self, height: int) -> None:
        if self.y + height > BOTTOM_LIMIT:
            self.new_page()

    def line_height(self, font: ImageFont.ImageFont, spacing: int = 8) -> int:
        bbox = font.getbbox("Ag")
        return (bbox[3] - bbox[1]) + spacing

    def get_heading_font(self, level: int) -> ImageFont.ImageFont:
        return {1: H1_FONT, 2: H2_FONT, 3: H3_FONT}.get(level, H4_FONT)

    def estimate_heading_height(self, level: int, text: str) -> int:
        text = self.sanitize_inline(text)
        font = self.get_heading_font(level)
        spacing_before = {1: 20, 2: 18, 3: 14, 4: 10}.get(level, 10)
        spacing_after = {1: 20, 2: 14, 3: 10, 4: 8}.get(level, 8)
        lines = self.wrap_text(text, font, CONTENT_WIDTH)
        return spacing_before + (len(lines) * self.line_height(font, 10)) + spacing_after

    def estimate_image_height(self, image_path: Path, max_height: int = 950) -> int:
        with Image.open(image_path) as img:
            scale = min(CONTENT_WIDTH / img.width, max_height / img.height, 1.0)
            return int(img.height * scale) + 24

    def sanitize_inline(self, text: str) -> str:
        text = text.replace("`", "")
        text = text.replace("**", "")
        text = text.replace("​", "")
        return text

    def wrap_text(self, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
        text = self.sanitize_inline(text).strip()
        if not text:
            return [""]

        words = text.split()
        lines: list[str] = []
        current = words[0]

        for word in words[1:]:
            test = f"{current} {word}"
            if self.draw.textlength(test, font=font) <= width:
                current = test
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def draw_wrapped_lines(
        self,
        lines: list[str],
        font: ImageFont.ImageFont,
        *,
        fill: tuple[int, int, int] = TEXT_COLOR,
        indent: int = 0,
        spacing: int = 8,
    ) -> None:
        lh = self.line_height(font, spacing)
        block_height = max(lh, len(lines) * lh)
        self.ensure_space(block_height)
        x = MARGIN_X + indent
        for line in lines:
            self.draw.text((x, self.y), line, fill=fill, font=font)
            self.y += lh

    def draw_heading(self, level: int, text: str) -> None:
        text = self.sanitize_inline(text)
        font = self.get_heading_font(level)
        spacing_before = {1: 20, 2: 18, 3: 14, 4: 10}.get(level, 10)
        spacing_after = {1: 20, 2: 14, 3: 10, 4: 8}.get(level, 8)
        self.y += spacing_before
        lines = self.wrap_text(text, font, CONTENT_WIDTH)
        self.draw_wrapped_lines(lines, font, spacing=10)
        self.y += spacing_after

    def draw_paragraph(self, text: str) -> None:
        lines = self.wrap_text(text, BODY_FONT, CONTENT_WIDTH)
        self.draw_wrapped_lines(lines, BODY_FONT)
        self.y += 10

    def draw_bullet(self, text: str, bullet: str = "-") -> None:
        bullet_indent = 28
        text_indent = 56
        wrapped = self.wrap_text(text, BODY_FONT, CONTENT_WIDTH - text_indent)
        lh = self.line_height(BODY_FONT)
        height = max(lh, len(wrapped) * lh)
        self.ensure_space(height)
        self.draw.text((MARGIN_X + bullet_indent, self.y), bullet, fill=TEXT_COLOR, font=BODY_BOLD_FONT)
        for i, line in enumerate(wrapped):
            self.draw.text((MARGIN_X + text_indent, self.y + (i * lh)), line, fill=TEXT_COLOR, font=BODY_FONT)
        self.y += height + 6

    def draw_numbered(self, marker: str, text: str) -> None:
        marker_width = self.draw.textlength(marker, font=BODY_BOLD_FONT) + 12
        wrapped = self.wrap_text(text, BODY_FONT, CONTENT_WIDTH - int(marker_width) - 30)
        lh = self.line_height(BODY_FONT)
        height = max(lh, len(wrapped) * lh)
        self.ensure_space(height)
        self.draw.text((MARGIN_X + 12, self.y), marker, fill=TEXT_COLOR, font=BODY_BOLD_FONT)
        for i, line in enumerate(wrapped):
            self.draw.text((MARGIN_X + 12 + marker_width, self.y + (i * lh)), line, fill=TEXT_COLOR, font=BODY_FONT)
        self.y += height + 6

    def draw_code_block(self, lines: list[str]) -> None:
        if not lines:
            return
        sanitized = [line.rstrip() for line in lines]
        wrapped_lines: list[str] = []
        for line in sanitized:
            chunks = self.wrap_text(line if line else " ", CODE_FONT, CONTENT_WIDTH - 50)
            wrapped_lines.extend(chunks)
        lh = self.line_height(CODE_FONT, 6)
        box_height = (len(wrapped_lines) * lh) + 36
        self.ensure_space(box_height)
        self.draw.rounded_rectangle(
            (MARGIN_X, self.y, MARGIN_X + CONTENT_WIDTH, self.y + box_height),
            radius=18,
            outline=GRID_COLOR,
            fill=SHADE_COLOR,
            width=2,
        )
        cy = self.y + 18
        for line in wrapped_lines:
            self.draw.text((MARGIN_X + 24, cy), line, fill=MUTED_COLOR, font=CODE_FONT)
            cy += lh
        self.y += box_height + 12

    def draw_table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        rows = [[self.sanitize_inline(cell.strip()) for cell in row] for row in rows]
        col_count = max(len(row) for row in rows)
        norm_rows = [row + [""] * (col_count - len(row)) for row in rows]
        col_width = CONTENT_WIDTH / col_count
        table_x = MARGIN_X

        wrapped_rows: list[list[list[str]]] = []
        row_heights: list[int] = []
        for idx, row in enumerate(norm_rows):
            font = TABLE_BOLD_FONT if idx == 0 else TABLE_FONT
            wrapped_cells: list[list[str]] = []
            max_lines = 1
            for cell in row:
                lines = self.wrap_text(cell, font, int(col_width) - 18)
                wrapped_cells.append(lines)
                max_lines = max(max_lines, len(lines))
            wrapped_rows.append(wrapped_cells)
            row_heights.append(max_lines * self.line_height(font, 6) + 18)

        total_height = sum(row_heights)
        self.ensure_space(total_height + 10)

        y = self.y
        for row_index, wrapped_cells in enumerate(wrapped_rows):
            row_height = row_heights[row_index]
            fill = (235, 240, 248) if row_index == 0 else ((250, 250, 250) if row_index % 2 == 0 else (255, 255, 255))
            self.draw.rectangle((table_x, y, table_x + CONTENT_WIDTH, y + row_height), fill=fill, outline=GRID_COLOR, width=1)
            font = TABLE_BOLD_FONT if row_index == 0 else TABLE_FONT

            cx = table_x
            for cell_lines in wrapped_cells:
                self.draw.rectangle((cx, y, cx + col_width, y + row_height), outline=GRID_COLOR, width=1)
                cy = y + 9
                for line in cell_lines:
                    self.draw.text((cx + 9, cy), line, fill=TEXT_COLOR, font=font)
                    cy += self.line_height(font, 6)
                cx += col_width
            y += row_height

        self.y = y + 14

    def draw_image(self, image_path: Path, caption: str | None = None, shrink_to_fit_page: bool = False) -> None:
        if caption:
            self.draw_paragraph(caption)

        img = Image.open(image_path).convert("RGB")
        max_width = CONTENT_WIDTH
        max_height = 950
        if shrink_to_fit_page:
            max_height = max(300, min(max_height, BOTTOM_LIMIT - self.y - 24))
        scale = min(max_width / img.width, max_height / img.height, 1.0)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        needed_height = img.height + 24
        if not shrink_to_fit_page:
            self.ensure_space(needed_height)

        x = MARGIN_X + int((CONTENT_WIDTH - img.width) / 2)
        self.page.paste(img, (x, self.y))
        self.y += img.height + 24

    def save(self, out_path: Path) -> None:
        rgb_pages = [page.convert("RGB") for page in self.pages]
        rgb_pages[0].save(out_path, "PDF", resolution=150.0, save_all=True, append_images=rgb_pages[1:])


def parse_markdown(path: Path) -> list[Block]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[Block] = []
    i = 0
    in_code_block = False
    code_lines: list[str] = []

    while i < len(raw_lines):
        line = raw_lines[i].rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_lines = []
            else:
                blocks.append(Block("code", code_lines))
                in_code_block = False
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        if stripped == "<!-- pagebreak -->":
            blocks.append(Block("pagebreak", ""))
            i += 1
            continue

        if not stripped:
            blocks.append(Block("blank", ""))
            i += 1
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            blocks.append(Block("heading", (level, text)))
            i += 1
            continue

        if stripped.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if match:
                alt_text, rel_path = match.groups()
                blocks.append(Block("image", (alt_text, rel_path)))
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines: list[str] = []
            while i < len(raw_lines) and raw_lines[i].strip().startswith("|"):
                current = raw_lines[i].strip()
                if not re.fullmatch(r"\|\s*[-:| ]+\|?", current):
                    table_lines.append(current)
                i += 1
            rows = []
            for table_line in table_lines:
                pieces = [cell.strip() for cell in table_line.strip("|").split("|")]
                rows.append(pieces)
            blocks.append(Block("table", rows))
            continue

        bullet_match = re.match(r"^[-*]\s+(.*)$", stripped)
        if bullet_match:
            blocks.append(Block("bullet", bullet_match.group(1)))
            i += 1
            continue

        numbered_match = re.match(r"^(\d+\.)\s+(.*)$", stripped)
        if numbered_match:
            blocks.append(Block("numbered", (numbered_match.group(1), numbered_match.group(2))))
            i += 1
            continue

        paragraph_lines = [stripped]
        i += 1
        while i < len(raw_lines):
            candidate = raw_lines[i].rstrip().strip()
            if not candidate:
                break
            if candidate.startswith(("#", "![", "|", "```")):
                break
            if re.match(r"^[-*]\s+", candidate) or re.match(r"^\d+\.\s+", candidate):
                break
            paragraph_lines.append(candidate)
            i += 1
        blocks.append(Block("paragraph", " ".join(paragraph_lines)))

    return blocks


def render_markdown_to_pdf() -> None:
    renderer = PdfRenderer()
    blocks = parse_markdown(SOURCE_MD)

    def next_meaningful_block(start: int) -> Block | None:
        for j in range(start, len(blocks)):
            if blocks[j].kind != "blank":
                return blocks[j]
        return None

    for idx, block in enumerate(blocks):
        if block.kind == "pagebreak":
            renderer.force_page_break()
        elif block.kind == "blank":
            renderer.y += 8
        elif block.kind == "heading":
            level, text = block.value
            next_block = next_meaningful_block(idx + 1)
            if next_block and next_block.kind == "image":
                _, rel_path = next_block.value
                image_path = (SOURCE_MD.parent / rel_path).resolve()
                combined_height = renderer.estimate_heading_height(level, text) + renderer.estimate_image_height(image_path)
                if renderer.y + combined_height > BOTTOM_LIMIT:
                    renderer.force_page_break()
                renderer.keep_next_image_on_page = True
            renderer.draw_heading(level, text)
        elif block.kind == "paragraph":
            renderer.draw_paragraph(str(block.value))
        elif block.kind == "bullet":
            renderer.draw_bullet(str(block.value))
        elif block.kind == "numbered":
            marker, text = block.value
            renderer.draw_numbered(marker, text)
        elif block.kind == "code":
            renderer.draw_code_block(list(block.value))
        elif block.kind == "table":
            renderer.draw_table(list(block.value))
        elif block.kind == "image":
            alt_text, rel_path = block.value
            image_path = (SOURCE_MD.parent / rel_path).resolve()
            renderer.draw_image(
                image_path,
                None if not alt_text else "",
                shrink_to_fit_page=renderer.keep_next_image_on_page,
            )
            renderer.keep_next_image_on_page = False

    renderer.save(OUTPUT_PDF)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    render_markdown_to_pdf()
