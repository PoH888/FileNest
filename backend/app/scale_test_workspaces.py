"""为第 35 课生成可重复的规模测试工作区。"""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Literal


ScaleName = Literal["small", "medium", "large"]
DEFAULT_SEED = 3501
FIXED_MTIME_NS = 1_704_067_200_000_000_000


@dataclass(frozen=True, slots=True)
class ScaleProfile:
    """一个规模档位的固定工作区组成。"""

    name: ScaleName
    file_count: int
    document_file_count: int
    leaf_directory_count: int
    document_bytes: int = 1_024
    other_file_bytes: int = 256


SCALE_PROFILES: dict[ScaleName, ScaleProfile] = {
    "small": ScaleProfile(
        name="small",
        file_count=100,
        document_file_count=80,
        leaf_directory_count=10,
    ),
    "medium": ScaleProfile(
        name="medium",
        file_count=1_000,
        document_file_count=800,
        leaf_directory_count=50,
    ),
    "large": ScaleProfile(
        name="large",
        file_count=10_000,
        document_file_count=8_000,
        leaf_directory_count=200,
    ),
}


@dataclass(frozen=True, slots=True)
class ScaleWorkspaceManifest:
    """生成结果的路径无关摘要，便于后续评测记录。"""

    schema_version: str
    scale: ScaleName
    seed: int
    file_count: int
    document_file_count: int
    non_document_file_count: int
    leaf_directory_count: int
    total_bytes: int
    content_sha256: str


class ScaleWorkspaceError(ValueError):
    """规模测试工作区参数或目标目录不符合契约。"""


def get_scale_profile(scale: str) -> ScaleProfile:
    """返回固定规模档位，拒绝未登记的规模名称。"""

    try:
        return SCALE_PROFILES[scale]  # type: ignore[index]
    except KeyError as error:
        available = ", ".join(SCALE_PROFILES)
        raise ScaleWorkspaceError(
            f"scale must be one of: {available}"
        ) from error


def generate_scale_workspace(
    output_root: Path,
    scale: str,
    *,
    seed: int = DEFAULT_SEED,
) -> ScaleWorkspaceManifest:
    """在全新目录中生成一个固定规模的测试工作区。"""

    profile = get_scale_profile(scale)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ScaleWorkspaceError("seed must be an integer")

    output_root = Path(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise ScaleWorkspaceError(
            "output directory must not already exist; refusing to overwrite"
        )

    output_root.mkdir(parents=True)
    resolved_root = output_root.resolve(strict=True)
    leaf_directories = _leaf_directories(profile)
    for relative_directory in leaf_directories:
        (resolved_root / relative_directory).mkdir(parents=True)

    content_digest = sha256()
    total_bytes = 0

    for file_index in range(profile.file_count):
        relative_directory = leaf_directories[
            file_index % len(leaf_directories)
        ]
        is_document = file_index < profile.document_file_count
        extension = ".md" if is_document and file_index % 2 == 0 else ".txt"
        if not is_document:
            extension = ".dat"

        relative_path = (
            relative_directory / f"item-{file_index:06d}{extension}"
        )
        payload = _file_payload(
            profile,
            file_index,
            seed,
            is_document=is_document,
        )
        destination = resolved_root / relative_path
        destination.write_bytes(payload)
        os.utime(destination, ns=(FIXED_MTIME_NS, FIXED_MTIME_NS))

        relative_bytes = relative_path.as_posix().encode("ascii")
        content_digest.update(relative_bytes)
        content_digest.update(b"\0")
        content_digest.update(payload)
        total_bytes += len(payload)

    return ScaleWorkspaceManifest(
        schema_version="1.0",
        scale=profile.name,
        seed=seed,
        file_count=profile.file_count,
        document_file_count=profile.document_file_count,
        non_document_file_count=(
            profile.file_count - profile.document_file_count
        ),
        leaf_directory_count=profile.leaf_directory_count,
        total_bytes=total_bytes,
        content_sha256=content_digest.hexdigest(),
    )


def _leaf_directories(profile: ScaleProfile) -> tuple[Path, ...]:
    """生成两层目录，保持目录数量和路径顺序稳定。"""

    return tuple(
        Path(f"area-{index // 10:03d}")
        / f"bucket-{index:04d}"
        for index in range(profile.leaf_directory_count)
    )


def _file_payload(
    profile: ScaleProfile,
    file_index: int,
    seed: int,
    *,
    is_document: bool,
) -> bytes:
    """生成固定长度的 UTF-8 文本或非文档载荷。"""

    size = profile.document_bytes if is_document else profile.other_file_bytes
    if is_document:
        token = (
            "filenest-benchmark-token"
            if file_index % 10 == 0
            else "filenest-document-token"
        )
        prefix = (
            f"FileNest scale fixture\n"
            f"scale={profile.name} seed={seed} file={file_index:06d}\n"
            f"{token}\n"
        )
    else:
        prefix = (
            f"FileNest non-document fixture "
            f"scale={profile.name} seed={seed} file={file_index:06d}\n"
        )

    return (prefix + ("fixture-data\n" * size)).encode("ascii")[:size]


def build_parser() -> argparse.ArgumentParser:
    """构建规模工作区生成命令行。"""

    parser = argparse.ArgumentParser(
        description="生成 FileNest 第 35 课的可重复规模测试工作区。"
    )
    parser.add_argument(
        "--scale",
        choices=tuple(SCALE_PROFILES),
        required=True,
        help="规模档位：small、medium 或 large。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="必须尚不存在的工作区目录。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"固定内容 seed，默认 {DEFAULT_SEED}。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行生成命令并输出路径无关的 JSON 摘要。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = generate_scale_workspace(
            args.output_dir,
            args.scale,
            seed=args.seed,
        )
    except ScaleWorkspaceError as error:
        parser.error(str(error))

    print(json.dumps(asdict(manifest), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
