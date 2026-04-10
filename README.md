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

安装后可执行 `py-heed` 命令。

## 使用方法

### 查看版本

```bash
py-heed --version
```

### 音视频转文字 (text 子命令)

```bash
# 基本用法，默认输出 SRT 格式
py-heed text video.mp4 -m /path/to/model

# 指定模型路径
py-heed text video.mp4 --model /path/to/model

# 输出 TXT 格式
py-heed text video.mp4 -m /path/to/model -f txt

# 同时输出 SRT 和 TXT
py-heed text video.mp4 -m /path/to/model -f all

# 指定输出目录
py-heed text video.mp4 -m /path/to/model -o ./output/

# 指定输出文件路径
py-heed text video.mp4 -m /path/to/model -o ./output/result.srt

# 指定设备（默认 auto，自动检测 GPU）
py-heed text video.mp4 -m /path/to/model -d cuda
py-heed text video.mp4 -m /path/to/model -d cpu
py-heed text video.mp4 -m /path/to/model -d auto

# JSON 输出（用于程序处理）
py-heed text video.mp4 -m /path/to/model -j
```

### 视频转音频 (audio 子命令)

```bash
# 基本用法，默认转为 MP3
py-heed audio video.mp4

# 指定输出格式
py-heed audio video.mp4 -f wav
py-heed audio video.mp4 --format flac

# 指定输出目录
py-heed audio video.mp4 -o ./audio/

# 指定输出文件路径
py-heed audio video.mp4 -o ./audio/video.mp3

# 指定音质（默认 best）
py-heed audio video.mp4 -q high

# 指定设备（默认 gpu）
py-heed audio video.mp4 -d cpu

# JSON 输出（用于程序处理）
py-heed audio video.mp4 -j
```

## text 子命令参数说明

| 参数 | 短参数 | 说明 | 默认值 |
|------|--------|------|--------|
| `INPUT` | - | 音视频文件路径 | 必需 |
| `--model` / `-m` | 模型路径 | 指向 faster-whisper 模型目录 | 必需 |
| `--format` / `-f` | 输出格式 | `srt`, `txt`, `all` | `srt` |
| `--output` / `-o` | 输出路径 | 输出目录或文件路径 | 当前目录 |
| `--device` / `-d` | 计算设备 | `auto`, `cuda`, `cpu` | `auto` |
| `--json` / `-j` | JSON输出 | 输出 JSON 格式结果 | `false` |

## audio 子命令参数说明

| 参数 | 短参数 | 说明 | 默认值 |
|------|--------|------|--------|
| `INPUT` | - | 视频文件路径 | 必需 |
| `--output` / `-o` | 输出路径 | 输出目录或文件路径 | 当前目录 |
| `--format` / `-f` | 输出格式 | `mp3`, `wav`, `m4a`, `flac`, `ogg`, `aac`, `wma` | `mp3` |
| `--device` / `-d` | 计算设备 | `gpu`, `cpu` | `gpu` |
| `--quality` / `-q` | 音质 | `best`, `high`, `medium`, `low` | `best` |
| `--json` / `-j` | JSON输出 | 输出 JSON 格式结果 | `false` |

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

## 进度显示

所有命令都支持同行进度显示：

```
00:01:23 / 00:05:00 (41%)
```

使用 `--json` 参数可获取 JSON 格式的完整结果。

## 设备选择

- **text 子命令**：`auto` 自动检测 GPU，`cuda` 强制 GPU，`cpu` 强制 CPU
- **audio 子命令**：`gpu` 使用 GPU 加速（默认），`cpu` 使用 CPU

## 依赖

- Python >= 3.9
- ffmpeg（系统已安装）
- faster-whisper 模型（需另行下载 for text 子命令）

## License

MIT
