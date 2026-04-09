# py-heed

音视频转文字工具，基于 faster-whisper。

## 安装

### uvx 方式（无需安装）

```bash
uvx py-heed --help
```

### uv tool install 方式（永久安装）

```bash
uv tool install -e .
```

安装后可执行 `py-head` 命令。

## 使用方法

```bash
# 基本用法，默认输出 SRT 格式
py-head video.mp4 -m /path/to/model

# 指定模型路径
py-head video.mp4 --model /path/to/model

# 输出 TXT 格式
py-head video.mp4 -m /path/to/model -f txt

# 同时输出 SRT 和 TXT
py-head video.mp4 -m /path/to/model -f all

# 指定输出目录
py-head video.mp4 -m /path/to/model -o ./output/

# 指定输出文件路径
py-head video.mp4 -m /path/to/model -o ./output/result.srt

# 指定设备（默认 auto，自动检测 GPU）
py-head video.mp4 -m /path/to/model -d cuda
py-head video.mp4 -m /path/to/model -d cpu
py-head video.mp4 -m /path/to/model -d auto
```

## 参数说明

| 参数 | 短参数 | 说明 | 默认值 |
|------|--------|------|--------|
| `INPUT` | - | 音视频文件路径 | 必需 |
| `--model` / `-m` | 模型路径 | 指向 faster-whisper 模型目录 | 必需 |
| `--format` / `-f` | 输出格式 | `srt`, `txt`, `all` | `srt` |
| `--output` / `-o` | 输出路径 | 输出目录或文件路径 | 当前目录 |
| `--device` / `-d` | 计算设备 | `auto`, `cuda`, `cpu` | `auto` |

## 输出格式

### SRT（默认）

SubRip 字幕格式，包含时间戳：

```
1
00:00:00,000 --> 00:00:03,500
这是第一段语音

2
00:00:04,000 --> 00:00:07,200
这是第二段语音
```

### TXT

纯文本格式，每行包含时间区间和语音内容：

```
[00:00:00 - 00:00:03] 这是第一段语音
[00:00:04 - 00:00:07] 这是第二段语音
```

## 输出路径规则

- `--output` 指定**目录**时：输出文件名为源文件名（不含扩展名）+ `.srt` 或 `.txt`
- `--output` 指定**文件**时：必须指定单一格式（`srt` 或 `txt`），直接输出到该文件
- 目录不存在时自动创建

## 设备选择

- `auto`：自动检测系统是否有 GPU，有则使用 GPU（cuda），否则使用 CPU
- `cuda`：强制使用 GPU（需要 CUDA 环境）
- `cpu`：强制使用 CPU

## 依赖

- Python >= 3.9
- ffmpeg（系统已安装）
- faster-whisper 模型（需另行下载）

## License

MIT
