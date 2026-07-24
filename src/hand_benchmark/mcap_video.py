"""Extract one H.264 ``foxglove.CompressedVideo`` topic from an MCAP to MP4.

This is a focused adaptation of fsstudio's tested MCAP video extraction path.
It reads the MCAP summary and chunk indexes directly so the dataset workflow can
remain independent from the ViPE and WiLoR runtime.
"""

from __future__ import annotations

import shutil
import statistics
import struct
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

MCAP_MAGIC = b"\x89MCAP0\r\n"
COMPRESSED_VIDEO_SCHEMA = "foxglove.CompressedVideo"
DEFAULT_FPS = 30.0


@dataclass(frozen=True)
class McapSchema:
    """Schema metadata required to decode MCAP protobuf messages."""

    schema_id: int
    name: str
    encoding: str
    data: bytes


@dataclass(frozen=True)
class McapChannel:
    """Channel metadata required to select an MCAP topic."""

    channel_id: int
    schema_id: int
    topic: str


@dataclass(frozen=True)
class McapChunkIndex:
    """Chunk location and channel index metadata."""

    chunk_start_offset: int
    channel_offsets: dict[int, int]


@dataclass(frozen=True)
class McapSummary:
    """Read-only MCAP summary used for compressed-video extraction."""

    path: Path
    schemas: dict[int, McapSchema]
    channels: dict[int, McapChannel]
    chunk_indexes: list[McapChunkIndex]


@dataclass(frozen=True)
class VideoExportResult:
    """Provenance and diagnostics for one exported MCAP video stream."""

    output_video_path: Path
    source_mcap_path: Path
    video_topic: str
    frame_id: str | None
    frame_count: int
    time_range_sec: tuple[float, float]
    average_fps: float
    format: str
    warnings: list[str]

    def to_report(self) -> dict[str, Any]:
        """Return a JSON-serializable video-export report."""
        report = asdict(self)
        report["output_video_path"] = str(self.output_video_path)
        report["source_mcap_path"] = str(self.source_mcap_path)
        report["time_range_sec"] = list(self.time_range_sec)
        return report


@dataclass
class _BitstreamWriter:
    """State for writing a decodable H.264 GOP."""

    handle: BinaryIO
    packet_count: int = 0
    seen_idr: bool = False
    parameter_sets: bytes = field(default_factory=bytes)


def read_mcap_summary(path: Path) -> McapSummary:
    """Read schemas, channels, and chunk indexes without decoding video payloads."""
    schemas: dict[int, McapSchema] = {}
    channels: dict[int, McapChannel] = {}
    chunk_indexes: list[McapChunkIndex] = []
    with path.open("rb") as handle:
        if handle.read(8) != MCAP_MAGIC:
            raise ValueError(f"Not an MCAP file or unsupported magic: {path}")
        while record := _read_record(handle):
            opcode, data = record
            if opcode == 2:
                break
            if opcode == 3:
                schema = _parse_schema(data)
                schemas[schema.schema_id] = schema
            elif opcode == 4:
                channel = _parse_channel(data)
                channels[channel.channel_id] = channel
            elif opcode == 8:
                chunk_indexes.append(_parse_chunk_index(data))
    return McapSummary(
        path=path, schemas=schemas, channels=channels, chunk_indexes=chunk_indexes
    )


def export_video_topic(
    *,
    mcap_path: Path,
    output_video_path: Path,
    video_topic: str,
    overwrite: bool = False,
) -> VideoExportResult:
    """Export a selected compressed-video topic to a decodable MP4 file."""
    summary = read_mcap_summary(mcap_path)
    channel = next(
        (item for item in summary.channels.values() if item.topic == video_topic), None
    )
    if channel is None:
        raise ValueError(f"Video topic not found in {mcap_path}: {video_topic}")
    schema = summary.schemas.get(channel.schema_id)
    if schema is None or schema.name != COMPRESSED_VIDEO_SCHEMA:
        actual_schema = schema.name if schema else "unknown"
        raise ValueError(
            f"Expected {COMPRESSED_VIDEO_SCHEMA} for {video_topic}, got {actual_schema}"
        )
    if output_video_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists; pass --overwrite: {output_video_path}"
        )
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("Extracting MP4 video requires ffmpeg on PATH.")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_h264_path = output_video_path.with_suffix(".h264.tmp")
    try:
        timestamps, log_times, frame_id, video_format = _extract_h264_bitstream(
            summary, channel, temporary_h264_path
        )
        average_fps = _estimate_fps(log_times)
        _mux_h264(
            ffmpeg_path, temporary_h264_path, output_video_path, average_fps, overwrite
        )
    finally:
        temporary_h264_path.unlink(missing_ok=True)

    return VideoExportResult(
        output_video_path=output_video_path,
        source_mcap_path=mcap_path,
        video_topic=video_topic,
        frame_id=frame_id,
        frame_count=len(timestamps),
        time_range_sec=(timestamps[0], timestamps[-1]),
        average_fps=average_fps,
        format=video_format,
        warnings=[
            "Exported one selected MCAP camera stream; no rectification or multi-camera fusion was applied."
        ],
    )


def _extract_h264_bitstream(
    summary: McapSummary, channel: McapChannel, temporary_h264_path: Path
) -> tuple[list[float], list[int], str | None, str]:
    message_class = _protobuf_classes(summary.schemas).get(
        summary.schemas[channel.schema_id].name
    )
    if message_class is None:
        raise RuntimeError(
            f"Could not build protobuf class for {summary.schemas[channel.schema_id].name}"
        )
    timestamps: list[float] = []
    log_times: list[int] = []
    frame_id: str | None = None
    video_format = "unknown"
    writer = _BitstreamWriter(temporary_h264_path.open("wb"))
    try:
        for log_time_ns, payload in _iter_channel_messages(summary, channel.channel_id):
            message = message_class()
            message.ParseFromString(payload)
            data = bytes(message.data)
            if not writer.seen_idr:
                _begin_gop(writer, data)
                if not writer.seen_idr:
                    continue
            else:
                _write_packet(writer, data)
            timestamp = _message_timestamp(message, log_time_ns)
            timestamps.append(timestamp)
            log_times.append(log_time_ns)
            if frame_id is None:
                frame_id = str(message.frame_id) if message.frame_id else None
                video_format = str(message.format) if message.format else "unknown"
    finally:
        writer.handle.close()
    if not timestamps:
        temporary_h264_path.unlink(missing_ok=True)
        raise RuntimeError(f"No decodable H.264 packets extracted from {channel.topic}")
    return timestamps, log_times, frame_id, video_format


def _protobuf_classes(schemas: dict[int, McapSchema]) -> dict[str, Any]:
    from google.protobuf import descriptor_pb2, message_factory

    files_by_name: dict[str, Any] = {}
    for schema in schemas.values():
        if schema.encoding != "protobuf":
            continue
        descriptor_set = descriptor_pb2.FileDescriptorSet()
        descriptor_set.ParseFromString(schema.data)
        files_by_name.update(
            {descriptor.name: descriptor for descriptor in descriptor_set.file}
        )
    ordered_files: list[Any] = []
    seen: set[str] = set()

    def add_dependencies(descriptor: Any) -> None:
        if descriptor.name in seen:
            return
        for dependency in descriptor.dependency:
            if dependency in files_by_name:
                add_dependencies(files_by_name[dependency])
        seen.add(descriptor.name)
        ordered_files.append(descriptor)

    for descriptor in files_by_name.values():
        add_dependencies(descriptor)
    return message_factory.GetMessages(ordered_files)


def _iter_channel_messages(
    summary: McapSummary, channel_id: int
) -> Iterable[tuple[int, bytes]]:
    for chunk_index in summary.chunk_indexes:
        if channel_id not in chunk_index.channel_offsets:
            continue
        for message_channel_id, log_time_ns, payload in _iter_messages(
            _read_chunk(summary.path, chunk_index)
        ):
            if message_channel_id == channel_id:
                yield log_time_ns, payload


def _read_chunk(path: Path, chunk_index: McapChunkIndex) -> bytes:
    with path.open("rb") as handle:
        handle.seek(chunk_index.chunk_start_offset)
        record = _read_record(handle)
    if record is None or record[0] != 6:
        raise ValueError(
            f"Expected MCAP chunk at offset {chunk_index.chunk_start_offset}"
        )
    data = record[1]
    offset = 0
    _, offset = _read_u64(data, offset)
    _, offset = _read_u64(data, offset)
    uncompressed_size, offset = _read_u64(data, offset)
    _, offset = _read_u32(data, offset)
    compression, offset = _read_string(data, offset)
    payload, _ = _read_bytes64(data, offset)
    if not compression:
        return payload
    if compression == "zstd":
        import zstandard

        return zstandard.ZstdDecompressor().decompress(
            payload, max_output_size=uncompressed_size
        )
    if compression == "lz4":
        lz4_path = shutil.which("lz4")
        if lz4_path is None:
            raise RuntimeError("MCAP uses LZ4 chunks, but lz4 is not on PATH.")
        process = subprocess.run(
            [lz4_path, "-d", "-c"], input=payload, capture_output=True, check=False
        )
        if process.returncode:
            raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
        return process.stdout
    raise RuntimeError(f"Unsupported MCAP chunk compression: {compression}")


def _read_record(handle: BinaryIO) -> tuple[int, bytes] | None:
    header = handle.read(9)
    if not header:
        return None
    if len(header) != 9:
        raise ValueError("Truncated MCAP record header")
    length = struct.unpack_from("<Q", header, 1)[0]
    data = handle.read(length)
    if len(data) != length:
        raise ValueError("Truncated MCAP record payload")
    return header[0], data


def _parse_schema(data: bytes) -> McapSchema:
    schema_id, offset = _read_u16(data, 0)
    name, offset = _read_string(data, offset)
    encoding, offset = _read_string(data, offset)
    schema_data, _ = _read_bytes32(data, offset)
    return McapSchema(schema_id, name, encoding, schema_data)


def _parse_channel(data: bytes) -> McapChannel:
    channel_id, offset = _read_u16(data, 0)
    schema_id, offset = _read_u16(data, offset)
    topic, _ = _read_string(data, offset)
    return McapChannel(channel_id, schema_id, topic)


def _parse_chunk_index(data: bytes) -> McapChunkIndex:
    _, offset = _read_u64(data, 0)
    _, offset = _read_u64(data, offset)
    chunk_start_offset, offset = _read_u64(data, offset)
    _, offset = _read_u64(data, offset)
    channel_offsets, _ = _read_u16_u64_map(data, offset)
    return McapChunkIndex(chunk_start_offset, channel_offsets)


def _iter_messages(chunk: bytes) -> Iterable[tuple[int, int, bytes]]:
    offset = 0
    while offset < len(chunk):
        opcode = chunk[offset]
        length = struct.unpack_from("<Q", chunk, offset + 1)[0]
        data = chunk[offset + 9 : offset + 9 + length]
        if opcode == 5:
            channel_id, message_offset = _read_u16(data, 0)
            _, message_offset = _read_u32(data, message_offset)
            log_time_ns, message_offset = _read_u64(data, message_offset)
            _, message_offset = _read_u64(data, message_offset)
            yield channel_id, log_time_ns, data[message_offset:]
        offset += 9 + length


def _message_timestamp(message: Any, fallback_log_time_ns: int) -> float:
    if message.timestamp.seconds or message.timestamp.nanos:
        return (
            float(message.timestamp.seconds)
            + float(message.timestamp.nanos) / 1_000_000_000
        )
    return fallback_log_time_ns / 1_000_000_000


def _estimate_fps(log_times_ns: list[int]) -> float:
    deltas = [
        (log_times_ns[index] - log_times_ns[index - 1]) / 1_000_000_000
        for index in range(1, len(log_times_ns))
        if log_times_ns[index] > log_times_ns[index - 1]
    ]
    return 1.0 / statistics.median(deltas) if deltas else DEFAULT_FPS


def _nal_types(data: bytes) -> set[int]:
    types: set[int] = set()
    index = 0
    while index < len(data) - 3:
        start_length = (
            3
            if data[index : index + 3] == b"\x00\x00\x01"
            else 4
            if data[index : index + 4] == b"\x00\x00\x00\x01"
            else 0
        )
        if not start_length:
            index += 1
            continue
        if index + start_length >= len(data):
            break
        types.add(data[index + start_length] & 0x1F)
        index += start_length + 1
    return types


def _begin_gop(writer: _BitstreamWriter, data: bytes) -> None:
    types = _nal_types(data)
    if types and types.issubset({7, 8}):
        writer.parameter_sets += data
        return
    has_idr = 5 in types
    has_sps_and_idr = 7 in types and has_idr
    if not has_idr or (not has_sps_and_idr and not writer.parameter_sets):
        return
    writer.seen_idr = True
    if writer.parameter_sets and not has_sps_and_idr:
        _write_packet(writer, writer.parameter_sets)
    _write_packet(writer, data)


def _write_packet(writer: _BitstreamWriter, data: bytes) -> None:
    writer.handle.write(data)
    writer.packet_count += 1


def _mux_h264(
    ffmpeg_path: str, raw_path: Path, output_path: Path, fps: float, overwrite: bool
) -> None:
    process = subprocess.run(
        [
            ffmpeg_path,
            "-y" if overwrite else "-n",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "+genpts",
            "-f",
            "h264",
            "-r",
            f"{fps:.6f}",
            "-i",
            str(raw_path),
            "-c:v",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stdout)


def _read_u16(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<H", data, offset)[0], offset + 2


def _read_u32(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def _read_u64(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", data, offset)[0], offset + 8


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    length, offset = _read_u32(data, offset)
    return data[offset : offset + length].decode(
        "utf-8", errors="replace"
    ), offset + length


def _read_bytes32(data: bytes, offset: int) -> tuple[bytes, int]:
    length, offset = _read_u32(data, offset)
    return data[offset : offset + length], offset + length


def _read_bytes64(data: bytes, offset: int) -> tuple[bytes, int]:
    length, offset = _read_u64(data, offset)
    return data[offset : offset + length], offset + length


def _read_u16_u64_map(data: bytes, offset: int) -> tuple[dict[int, int], int]:
    length, offset = _read_u32(data, offset)
    end = offset + length
    values: dict[int, int] = {}
    while offset < end:
        key, offset = _read_u16(data, offset)
        value, offset = _read_u64(data, offset)
        values[key] = value
    return values, offset
