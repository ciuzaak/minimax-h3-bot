"""Decode and split a Telegram batch prompt attachment."""

from codecs import BOM_UTF16_BE, BOM_UTF16_LE
from itertools import groupby
import re

from .jobs import extract_prompt_count


COUNT_TOKEN = re.compile(r"(?<![A-Za-z0-9_])[xX][+-]?[0-9]+$")


def parse_prompt_file(content):
    encoding = "utf-16" if content.startswith((BOM_UTF16_LE, BOM_UTF16_BE)) else "utf-8-sig"
    try:
        text = content.decode(encoding)
    except UnicodeError as error:
        raise ValueError("无法读取文本编码，请将文件另存为 UTF-8 后重发。") from error
    if "\0" in text:
        raise ValueError("文件包含空字符，请确认它是纯文本，并另存为 UTF-8 后重发。")

    prompts = []
    for has_text, lines in groupby(text.splitlines(), key=lambda line: bool(line.strip())):
        if not has_text:
            continue
        number = len(prompts) + 1
        try:
            prompt, count = extract_prompt_count("\n".join(lines))
        except ValueError as error:
            raise ValueError(f"第 {number} 组的生成数量无效，请在末尾写 xN（如 x3）。") from error
        if count is None and COUNT_TOKEN.search(prompt):
            raise ValueError(f"第 {number} 组缺少提示词或数量无效，请使用「提示词 x3」格式。")
        if count is not None and count > 2 ** 64:
            raise ValueError(f"第 {number} 组的生成数量超过可用的随机 seed 数量。")
        prompts.append((prompt, 1 if count is None else count))
    if not prompts:
        raise ValueError("文件中没有提示词。")
    return prompts
