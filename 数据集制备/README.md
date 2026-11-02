# FFE 回波与 RCS 数据读取程序

本项目用于从 FEKO 导出的 `.ffe` 远场文件中读取复回波或 RCS 数据，完成频点整理、角度镜像扩展、MAT 文件保存、静态曲线绘制，以及宽带多频点 HRRP（高分辨距离像）计算与热力图绘制。

核心程序为 [`process_ffe_reusable.m`](./process_ffe_reusable.m)。程序采用“顶部集中配置 + 主处理流程 + 本地函数”的结构，修改数据文件或处理方式时，一般只需要调整脚本开头的 `cfg` 参数。

## 1. 运行环境

- MATLAB R2016b 或更高版本。
- 输入文件为 FEKO 导出的文本格式 `.ffe` 文件。
- 脚本应在本项目根目录下运行；相对路径均以 MATLAB 当前工作目录为基准。

之所以要求 MATLAB R2016b 及以上版本，是因为脚本末尾定义了本地函数。

## 2. 当前输入与输出

默认输入文件：

```text
窄带数据/C/Test4_FarField1_C.ffe
```

默认输出文件：

```text
qiudizhuizhuti.mat
```

默认输出变量名：

```matlab
data
```

当前文件包含 2 个频点，每个频点有 1801 个角度采样点。默认只读取第 1 个频点，并将 0～180° 数据镜像扩展为 0～360°，因此输出 `data` 的尺寸为：

```text
1 × 3601
```

当前默认配置仍是窄带单频点处理，因此 `hrrpEnabled = false`。切换到宽带文件并选择全部连续频点后即可启用 HRRP。

## 3. 快速使用

1. 在 MATLAB 中将当前工作目录切换到项目根目录。
2. 根据需要修改 `process_ffe_reusable.m` 开头的配置区。
3. 在命令窗口运行：

```matlab
process_ffe_reusable
```

运行成功后，程序会显示处理结果、绘制曲线，并根据配置生成 MAT 文件。

## 4. FFE 数据列说明

当前 FEKO 文件的数值记录共有 9 列：

| 列号 | FEKO 字段 | 含义 |
| ---: | --- | --- |
| 1 | `Theta` | 俯仰角 |
| 2 | `Phi` | 方位角 |
| 3 | `Re(Etheta)` | 垂直极化回波实部 |
| 4 | `Im(Etheta)` | 垂直极化回波虚部 |
| 5 | `Re(Ephi)` | 水平极化回波实部 |
| 6 | `Im(Ephi)` | 水平极化回波虚部 |
| 7 | `RCS(Theta)` | Theta 极化 RCS |
| 8 | `RCS(Phi)` | Phi 极化 RCS |
| 9 | `RCS(Total)` | 总 RCS |

程序默认通过第 5、6 列构造水平极化复回波：

```matlab
values = raw(:, 5) + 1i * raw(:, 6);
```

若需要垂直极化复回波，可配置为：

```matlab
cfg.echoRealColumn = 3;
cfg.echoImagColumn = 4;
```

## 5. 配置参数说明

### 5.1 输入与输出路径

```matlab
cfg.inputDir       = fullfile('窄带数据', 'C');
cfg.inputName      = 'Test4_FarField1_C.ffe';
cfg.outputDir      = '.';
cfg.outputName     = 'qiudizhuizhuti.mat';
cfg.outputVariable = 'data';
```

| 参数 | 说明 |
| --- | --- |
| `inputDir` | 输入文件所在目录 |
| `inputName` | FFE 文件名 |
| `outputDir` | MAT 文件输出目录，`.` 表示当前目录 |
| `outputName` | 输出 MAT 文件名 |
| `outputVariable` | MAT 文件中的数据变量名 |

使用 `fullfile` 组合路径，可以避免手工处理 Windows 的 `\` 和 Linux 的 `/`。

### 5.2 回波与 RCS 模式

```matlab
cfg.dataType       = 'echo';
cfg.echoRealColumn = 5;
cfg.echoImagColumn = 6;
cfg.rcsColumn      = 9;
```

- `dataType = 'echo'`：读取两列并构造复数回波。
- `dataType = 'rcs'`：直接读取 `rcsColumn` 指定的 RCS 列。
- `echoRealColumn`、`echoImagColumn`：复回波的实部和虚部列号。
- `rcsColumn`：RCS 数据列号，默认第 9 列总 RCS。

读取总 RCS 的配置示例：

```matlab
cfg.dataType = 'rcs';
cfg.rcsColumn = 9;
cfg.dbFactor = 10;
```

### 5.3 FFE 记录抽取规则

```matlab
cfg.initialSkipLines = 6;
cfg.linesPerRecord   = 12;
```

当前 FFE 文件开头包含 6 行文件级说明，因此程序先跳过 6 行。之后每个角度记录由 12 行组成，其中最后一行是 9 列数值数据，程序每隔 12 行读取一次该数值行。

这部分逻辑与原程序一致，但不再生成 `mk420.txt` 临时文件，而是将数值直接读入 MATLAB 数组。

如果其他 FFE 文件的每条记录占 10 行，可以设置：

```matlab
cfg.linesPerRecord = 10;
```

这里的 10 或 12 取决于 FFE 文件内容和导出格式，而不是 Windows/Linux 换行符本身。`fgetl` 会自动去除不同系统的换行符。

### 5.4 角度和频点参数

```matlab
cfg.angleStartDeg       = 0;
cfg.angleEndDeg         = 180;
cfg.samplesPerFrequency = 1801;
cfg.frequencyCount      = 2;
cfg.frequencyIndices    = 1;
cfg.useSymmetry         = true;
```

| 参数 | 说明 |
| --- | --- |
| `angleStartDeg` | 原始数据起始角度 |
| `angleEndDeg` | 原始数据终止角度 |
| `samplesPerFrequency` | 每个频点的角度采样数 |
| `frequencyCount` | FFE 文件中的总频点数 |
| `frequencyIndices` | 需要输出的频点序号 |
| `useSymmetry` | 是否将半周数据镜像扩展为整周 |

当前配置为 0～180°、1801 个点，所以角度间隔为：

```text
(180 - 0) / (1801 - 1) = 0.1°
```

如果不确定频点数，可以让程序根据记录总数自动推断：

```matlab
cfg.frequencyCount = [];
```

频点选择示例：

```matlab
cfg.frequencyIndices = 1;      % 只读取第 1 个频点
cfg.frequencyIndices = [1 2];  % 读取第 1、2 个频点
cfg.frequencyIndices = ':';    % 读取全部频点
```

### 5.5 保存与绘图参数

```matlab
cfg.saveResult   = true;
cfg.saveAngles   = false;
cfg.showPlot     = true;
cfg.dbFactor     = 10;
cfg.plotTitle    = '球顶锥体静态数据曲线（窄带）';
cfg.xTickStepDeg = 40;
```

| 参数 | 说明 |
| --- | --- |
| `saveResult` | 是否保存 MAT 文件 |
| `saveAngles` | 是否同时保存角度数组和配置结构体 |
| `showPlot` | 是否绘制结果曲线 |
| `dbFactor` | 线性量转换为 dB 时的倍率 |
| `plotTitle` | 图标题 |
| `xTickStepDeg` | 横坐标刻度间隔 |

如果 `saveAngles = true`，MAT 文件中还会包含：

- `anglesDeg`：与数据列一一对应的角度数组。
- `cfg`：本次处理所使用的全部配置。

### 5.6 宽带 HRRP 参数

```matlab
cfg.hrrpEnabled        = false;
cfg.hrrpShowPlot       = true;
cfg.hrrpExponent       = 0.2;
cfg.hrrpNormalize      = false;
cfg.hrrpColormap       = 'hot';
cfg.hrrpTitle          = '静态 HRRP';
cfg.hrrpXTickStepDeg   = 60;
cfg.hrrpYTickStep      = 20;
cfg.hrrpRangeBinLimits = [];
```

| 参数 | 说明 |
| --- | --- |
| `hrrpEnabled` | 是否计算 HRRP |
| `hrrpShowPlot` | 是否绘制 HRRP 热力图 |
| `hrrpExponent` | 幅度动态范围压缩指数，默认与原示例一致为 0.2 |
| `hrrpNormalize` | 是否把显示图像归一化到 0～1 |
| `hrrpColormap` | 热力图颜色映射 |
| `hrrpTitle` | HRRP 图标题 |
| `hrrpXTickStepDeg` | 角度轴刻度间隔 |
| `hrrpYTickStep` | 距离单元轴刻度间隔 |
| `hrrpRangeBinLimits` | 显示的距离单元范围，`[]` 表示全部 |

HRRP 要求至少读取两个频点，而且频点序号必须连续递增。例如：

```matlab
cfg.frequencyIndices = ':';
```

或者：

```matlab
cfg.frequencyIndices = 1:101;
```

不应使用 `[1 3 5]`、`[101 100 99]` 等非连续或逆序选择来生成 HRRP。

HRRP 还必须使用包含幅度和相位的复回波，因此应设置：

```matlab
cfg.dataType = 'echo';
```

不能直接对第 9 列 RCS 功率值进行 IFFT，因为该列不包含构建距离像所需的复相位信息。程序会对此进行检查。

## 6. 完整处理流程

### 第一步：生成完整路径

```matlab
inputFile  = fullfile(cfg.inputDir, cfg.inputName);
outputFile = fullfile(cfg.outputDir, cfg.outputName);
```

输入、输出目录和文件名分开配置，便于批量更换文件。

### 第二步：读取数值记录

```matlab
raw = readFixedStrideNumericRows( ...
    inputFile, cfg.initialSkipLines, cfg.linesPerRecord);
```

`readFixedStrideNumericRows` 的工作过程如下：

1. 使用 `fopen` 打开 FFE 文件。
2. 使用 `onCleanup` 保证发生错误时文件仍能被关闭。
3. 跳过文件开头的说明行。
4. 每隔 `linesPerRecord` 行读取一条数值记录。
5. 使用 `sscanf` 将字符串转换为数值。
6. 检查每条记录的列数是否一致。

对于当前文件，读取结果 `raw` 的尺寸为：

```text
3602 × 9
```

即 2 个频点 × 每频点 1801 个角度，共 3602 条记录，每条记录 9 列。

### 第三步：提取回波或 RCS

回波模式将实部和虚部组合为复数：

```matlab
values = raw(:, cfg.echoRealColumn) + ...
         1i * raw(:, cfg.echoImagColumn);
```

RCS 模式直接读取指定列：

```matlab
values = raw(:, cfg.rcsColumn);
```

程序会先检查 FFE 文件是否包含所要求的列，避免因列号错误而产生难以定位的结果。

### 第四步：整理频点维度

```matlab
halfData = reshape(values, ...
    cfg.samplesPerFrequency, frequencyCount);
```

整理后的 `halfData` 采用以下布局：

- 行：角度采样点。
- 列：频点。

当前配置下其尺寸为 `1801 × 2`。

### 第五步：选择频点

程序根据 `frequencyIndices` 从 `halfData` 中选择一个、多个或全部频点，并检查索引是否超出有效范围。

### 第六步：构造角度数组

```matlab
halfAnglesDeg = linspace( ...
    cfg.angleStartDeg, cfg.angleEndDeg, ...
    cfg.samplesPerFrequency).';
```

这样角度数组不再依赖写死的 `0:0.1:180`，更换采样数或角度范围时无需修改其他代码。

### 第七步：对称扩展至 360°

```matlab
processedData = [selectedHalfData; ...
                 flipud(selectedHalfData(1:end-1, :))];
```

原始数据包含 0～180°。镜像部分使用原始数据的 0～179.9°，翻转后对应 180.1～360°。这样：

- 180° 不会被重复。
- 360° 与原始 0° 对应。
- 输出点数为 `1801 + 1800 = 3601`。

该操作仅适用于目标和仿真条件确实满足对应对称关系的情况。若目标不对称，应设置：

```matlab
cfg.useSymmetry = false;
```

### 第八步：调整输出方向

```matlab
data = processedData.';
```

最终 `data` 的布局为：

- 每一行对应一个频点。
- 每一列对应一个角度。

这里使用 `.'` 普通转置，而不是 `'` 共轭转置。对于复回波，如果误用 `'`，虚部符号会改变。

### 第九步：保存 MAT 文件

程序先将数据放入结构体，再使用 `-struct` 保存，从而支持通过 `cfg.outputVariable` 动态设置变量名。

默认只保存：

```matlab
data
```

### 第十步：绘制 dB 曲线

```matlab
plotValuesDb = cfg.dbFactor .* ...
    log10(max(abs(processedData), realmin));
```

其中 `max(..., realmin)` 用于避免零值导致 `log10(0) = -Inf`。

常用 dB 计算方式：

- 复回波幅度：通常使用 `20*log10(abs(E))`，即 `dbFactor = 20`。
- 功率或线性 RCS：通常使用 `10*log10(abs(RCS))`，即 `dbFactor = 10`。

当前默认值为 10，是为了保持与原始程序一致。应根据数据的物理定义选择正确倍率。

### 第十一步：计算并绘制宽带 HRRP

输出 `data` 的布局是“频点 × 角度”，因此频率维是第 1 维。程序沿第 1 维执行 IFFT：

```matlab
dataHrrp = fftshift(ifft(data, [], 1), 1);
```

其中：

- `ifft(..., 1)` 将频域步进数据转换到距离像维度。
- `fftshift(..., 1)` 将距离维循环移位，与原始 HRRP 示例保持一致。
- HRRP 的每一列对应一个角度。
- HRRP 的每一行对应一个距离单元。

显示图像采用：

```matlab
hrrpImage = abs(dataHrrp) .^ cfg.hrrpExponent;
```

指数 0.2 用于压缩动态范围，使较弱散射点在热力图中更容易观察。该指数只影响显示图像，不改变用于计算的复数 `dataHrrp`。

HRRP 在运行过程中保存在工作区变量 `dataHrrp` 中，用于生成图像，但不会写入 MAT 文件。MAT 文件仍然只保存原来的 `data`；如果启用了 `saveAngles`，则还会按原配置保存角度和配置结构体。

绘图时直接使用实际角度数组和距离单元序号：

```matlab
rangeBins = (0:size(dataHrrp, 1)-1).';
imagesc(anglesDeg, rangeBins, hrrpImage);
set(gca, 'YDir', 'normal');
```

因此不需要把横轴、纵轴写死为 `[0 360]` 和 `[0 100]`。对于 101 个频点，距离单元会自动显示为 0～100。

## 7. 输出数据尺寸

设：

- 每频点原始角度数为 `N`。
- 选择的频点数为 `F`。

则输出尺寸如下：

| 镜像设置 | `data` 尺寸 |
| --- | --- |
| `useSymmetry = false` | `F × N` |
| `useSymmetry = true` | `F × (2N-1)` |

当前 `N = 1801`、`F = 1`，所以输出为 `1 × 3601`。

如果选择全部 2 个频点，则输出为 `2 × 3601`。

启用 HRRP 后，工作区变量 `dataHrrp` 与 `data` 尺寸相同：

```text
频点数（距离单元数） × 角度数
```

例如，当前工作区宽带文件包含 101 个频点。读取全部频点并进行角度镜像后：

```text
data      : 101 × 3601
dataHrrp  : 101 × 3601（仅用于计算和绘图，不写入 MAT 文件）
```

## 8. 常见配置示例

### 8.1 读取水平极化复回波

```matlab
cfg.dataType       = 'echo';
cfg.echoRealColumn = 5;
cfg.echoImagColumn = 6;
cfg.dbFactor       = 20;
```

### 8.2 读取垂直极化复回波

```matlab
cfg.dataType       = 'echo';
cfg.echoRealColumn = 3;
cfg.echoImagColumn = 4;
cfg.dbFactor       = 20;
```

### 8.3 读取总 RCS

```matlab
cfg.dataType  = 'rcs';
cfg.rcsColumn = 9;
cfg.dbFactor  = 10;
```

### 8.4 读取所有频点并保存角度

```matlab
cfg.frequencyIndices = ':';
cfg.saveAngles = true;
```

### 8.5 只读取 0～180°，不进行镜像

```matlab
cfg.useSymmetry = false;
```

### 8.6 更换输入文件

```matlab
cfg.inputDir  = fullfile('宽带数据', 'S');
cfg.inputName = 'Test1_FarField1_S.ffe';
```

同时需要根据新文件内容检查：

- `initialSkipLines`
- `linesPerRecord`
- `samplesPerFrequency`
- `frequencyCount`
- 数据列号

### 8.7 读取 101 频点宽带数据并绘制 HRRP

当前工作区的宽带文件包含 101 个频点，频率范围为 5.6～6.4 GHz，频率步进为 8 MHz。可使用以下配置：

```matlab
cfg.inputDir            = fullfile('宽带数据', 'C');
cfg.inputName           = 'Test4_FarField1_C.ffe';
cfg.outputName          = 'qiudizhuizhuti_800MHz.mat';

cfg.samplesPerFrequency = 1801;
cfg.frequencyCount      = 101;
cfg.frequencyIndices    = ':';
cfg.useSymmetry         = true;
cfg.dataType            = 'echo';

cfg.showPlot            = false;  % 可选：避免同时绘制 101 条频点曲线
cfg.hrrpEnabled         = true;
cfg.hrrpShowPlot        = true;
cfg.hrrpRangeBinLimits  = [0 100];
```

运行后：

- `data` 是 `101 × 3601` 的多频点复回波或 RCS 数据。
- 工作区中的 `dataHrrp` 是 `101 × 3601` 的复数距离像数据，但不会写入 MAT 文件。
- 输出 MAT 文件仍只保存原来的 `data`。
- HRRP 图横轴为 0～360°，纵轴为 0～100 距离单元。

## 9. 常见错误及处理方法

### 无法打开 FFE 文件

提示类似：

```text
无法打开 FFE 文件
```

检查 MATLAB 当前工作目录、`inputDir` 和 `inputName` 是否正确。建议继续使用 `fullfile`，不要手工拼接路径分隔符。

### 没有读取到数据

提示类似：

```text
未读取到任何数据，请检查 initialSkipLines 和 linesPerRecord
```

通常表示 FFE 文件结构与当前抽取规则不一致。打开文件查看第一条数值记录所在行，以及相邻数值记录之间相隔多少行。

### 某一行不是有效数值记录

通常说明 `linesPerRecord` 设置错误，程序选中了标题行或说明行，而不是数值行。

### 实际记录数与配置不一致

提示会同时显示实际记录数和配置期望值。检查：

```matlab
cfg.samplesPerFrequency
cfg.frequencyCount
cfg.linesPerRecord
```

如果每频点采样数已知、频点数未知，可以设置：

```matlab
cfg.frequencyCount = [];
```

### 频点索引超出范围

例如文件只有 2 个频点，却设置了：

```matlab
cfg.frequencyIndices = 3;
```

应将索引限制在 `1:frequencyCount` 范围内。

### dB 曲线数值与预期不一致

首先确认数据类型和倍率：

- 复幅度一般使用 `dbFactor = 20`。
- RCS/功率一般使用 `dbFactor = 10`。

其次确认 FFE 中的 RCS 是否已经是 dB 值。当前代码假定读出的数据是线性量；如果文件列本身已经是 dB，则不应再次执行对数转换，需要相应修改绘图部分。

### HRRP 提示至少需要两个频点

说明当前只选择了一个频点。宽带 HRRP 必须把多个频率样本沿频率维做 IFFT，应将 `frequencyIndices` 设置为 `':'` 或连续频点序号。

### HRRP 提示必须使用复回波

说明当前 `dataType` 被设置为 `'rcs'`。HRRP 需要复数频域回波的幅度和相位信息，应改为 `'echo'` 并正确配置实部、虚部列号。

### HRRP 提示频点必须连续递增

说明 `frequencyIndices` 存在跳号、重复或逆序。应使用类似 `1:101` 的连续递增序列。即使索引连续，实际 FEKO 频率也应等间隔，否则距离像坐标和聚焦结果可能不正确。

## 10. 与原始程序相比的改进

- 所有可变参数集中在脚本顶部，不再散落在处理流程中。
- 不再创建和读取 `mk420.txt` 临时文件。
- 使用 `fullfile` 管理路径。
- 支持回波/RCS 模式切换。
- 支持单频点、多频点和全部频点输出。
- 支持自动推断频点数。
- 自动计算角度数组，不再写死 `0:0.1:360`。
- 增加文件、列数、记录数和频点索引检查。
- 使用普通转置保护复回波虚部符号。
- 输出数据维度定义清晰，便于后续成像或特征处理。
- 使用 `onCleanup` 保证文件句柄可靠关闭。
- 支持沿频率维自动计算 HRRP，并绘制角度—距离单元热力图。
- 宽带读取采用分块预分配，避免逐行扩容造成明显性能损失。
- 检查 HRRP 是否使用多个、连续、递增的频点。

## 11. 使用注意事项

1. 镜像扩展的前提是目标和仿真设置具有相应对称性。
2. `reshape` 假定 FFE 记录按“一个频点的全部角度，然后下一个频点”排列；若文件采用不同排列顺序，需要调整整理逻辑。
3. 当前固定步长读取方式适用于本项目中的 FFE 结构。若以后 FEKO 导出格式发生变化，应优先检查文件头和记录块结构。
4. MAT 文件中的复回波保留复数形式，后续算法可以直接使用其幅度和相位。
5. 如果后续处理需要角度与数据严格对应，建议启用 `cfg.saveAngles = true`。
6. HRRP 默认纵轴是距离单元编号，不是米。若需要物理距离，应根据频率步进、带宽和电磁传播速度建立距离轴。
7. HRRP 假定参与 IFFT 的频率样本等间隔排列；仅检查索引连续并不能代替对实际频率值的确认。

## 12. 进动俯仰角批量生成

脚本位置：

```text
等效俯仰角生成/generate_precession_angles_reusable.m
```

该脚本调用同目录下的 `chouqujiaodu_jindong_600.m`，批量生成不同初始俯仰角、进动频率和进动角组合对应的等效俯仰角曲线。

所有可变参数均位于脚本顶部的 `cfg` 配置区，包括：

- 频段及各频段初始俯仰角范围。
- PRF 和观测时间。
- 自旋频率。
- 进动频率和进动角范围。
- 输出目录、文件名和变量名。
- 是否保存参数元数据、显示进度或预览曲线。

默认配置包含 13 个初始俯仰角、11 个进动频率和 11 个进动角，因此一共生成：

```text
13 × 11 × 11 = 1573 组曲线
```

采样时间包含 0 和观测终点，所以在 PRF 为 20 Hz、观测时间为 5 s 时，每条曲线包含：

```text
5 × 20 + 1 = 101 个采样点
```

最终 `angle_target` 尺寸为 `1573 × 101`。数据行顺序与原程序保持一致：初始俯仰角为最外层循环，进动频率为中间层，进动角为最内层。

程序已使用结构体字段替代 `eval`，并预分配结果矩阵。默认只保存 `angle_target`；若设置 `cfg.saveMetadata = true`，还会保存每一行对应的参数、时间轴和配置结构体。

## 13. 根据静态回波和动态角度生成 RCS/HRRP

可复用函数位置：

```text
数据集生成代码/generate_dynamic_rcs_hrrp_reusable.m
```

该函数读取：

- `窄带数据/<频段>/` 或 `宽带数据/<频段>/` 中的四类目标静态数据。
- `等效俯仰角生成/` 中对应目标和频段的 `angle_target` 动态角度数据。

然后根据每个慢时间点的动态俯仰角，从静态“频点×角度”矩阵中抽取对应回波。窄带输出为“运动参数组合×慢时间”，宽带数据沿频率维进行 IFFT 后输出为“距离单元×慢时间×运动参数组合”。

两个生成开关位于函数顶部：

```matlab
cfg.generateNarrowband = true;
cfg.generateWideband   = true;
```

只生成窄带：

```matlab
cfg.generateNarrowband = true;
cfg.generateWideband   = false;
```

只生成宽带 HRRP：

```matlab
cfg.generateNarrowband = false;
cfg.generateWideband   = true;
```

窄带和宽带可以分别选择频段：

```matlab
cfg.narrowband.band = 'C';
cfg.wideband.band   = 'C';
```

默认输出至项目根目录的 `生成数据` 文件夹：

```text
rcs_C.mat   → rcs_zhaidai_cell
hrrp_C.mat  → hrrp_kuandai_cell
```

函数可以直接运行：

```matlab
addpath('数据集生成代码');
generate_dynamic_rcs_hrrp_reusable;
```

也可以通过结构体临时覆盖顶部参数，而不修改函数文件：

```matlab
userConfig.generateNarrowband = false;
userConfig.generateWideband = true;
userConfig.wideband.band = 'S';
userConfig.maximumCases = 100;

outputs = generate_dynamic_rcs_hrrp_reusable(userConfig);
```

程序会在开始计算前检查所有静态数据和动态角度文件，并检查变量名、慢时间采样数及动态角度是否超出静态数据角度范围。当前工作区缺少 `angle_target4_C.mat`、`angle_target4_S.mat` 和 `angle_target4_X.mat`，生成对应频段的完整四目标数据前需要先运行第4类目标的角度生成脚本。

## 14. 划分分类数据并逐样本添加AWGN

可复用函数位置：

```text
数据集生成代码/build_noisy_classification_dataset_reusable.m
```

该函数读取 `生成数据/rcs_<频段>.mat` 和 `生成数据/hrrp_<频段>.mat`，按照每类前1210组训练、后363组测试进行固定划分，然后逐样本添加AWGN。RCS始终使用整条样本的实测平均功率；HRRP可以选择三种SNR参考功率定义。

默认的整幅平均功率模式仍调用：

```matlab
awgn(cleanSample, snrDb, 'measured')
```

RCS的每一行作为一个独立样本；HRRP的每个第三维切片作为一个独立样本。随机种子只在整批数据开始前设置一次，因此同一批中的每个样本噪声不同，同时重新运行又能复现完全相同的结果。

主要配置：

```matlab
cfg.buildRcs  = true;
cfg.buildHrrp = true;
cfg.band = 'C';
cfg.snrDbList = -10:5:20;
cfg.noiseRealizations = 1;
cfg.trainNoiseSeedBase = 1001;
cfg.testNoiseSeedBase  = 2001;
cfg.reuseNoiseAcrossSnr = true;
```

HRRP噪声模式通过以下配置选择：

```matlab
cfg.hrrpNoise.mode = 'peak';
cfg.hrrpNoise.energyFraction = 0.95;
cfg.hrrpNoise.inputIsFftShifted = true;
```

| `hrrpNoise.mode` | SNR参考信号功率 |
|---|---|
| `'whole_sample'` | 整幅HRRP的均方功率，即原有方式 |
| `'peak'` | 将干净HRRP通过 `ifftshift → fft` 恢复到频域，按频域最大瞬时功率添加噪声，再通过 `ifft → fftshift` 返回HRRP；这是当前默认值 |
| `'energy_fraction'` | 累计达到 `energyFraction` 能量的主距离单元平均功率 |

`peak`模式的SNR定义在IFFT前的频域中，不要求IFFT后的HRRP峰值SNR仍等于设定值。`inputIsFftShifted=true`表示输入HRRP在生成时执行过距离维 `fftshift`；如果原始HRRP没有执行该操作，应设置为 `false`。

95%能量模式先沿慢时间计算每个距离单元的平均功率，按功率从大到小选择累计达到95%总能量的距离单元，再对这些距离单元的功率求平均。噪声依然添加到整幅HRRP，并不是只添加到主距离单元。

例如使用峰值SNR：

```matlab
userConfig.hrrpNoise.mode = 'peak';
```

使用95%能量主距离单元SNR：

```matlab
userConfig.hrrpNoise.mode = 'energy_fraction';
userConfig.hrrpNoise.energyFraction = 0.95;
```

可选的测试集实际SNR核验：

```matlab
cfg.noiseTest.enabled = true;
cfg.noiseTest.mode = 'random';
cfg.noiseTest.randomSampleCount = 100;
cfg.noiseTest.randomSeedBase = 3001;
cfg.noiseTest.printEachSample = false;
cfg.noiseTest.warnMeanAbsoluteErrorDb = [];
```

`mode = 'random'` 会从测试集中随机抽取指定数量的样本；`mode = 'all'` 会遍历全部测试样本。程序利用 `noisySample-cleanSample` 得到实际噪声，逐样本计算：

```matlab
measuredSnrDb = 10*log10(signalPower/noisePower);
```

核验结果会按照所选择的HRRP参考功率定义输出目标SNR、实测均值、标准差、范围及平均绝对误差，并保存在每个MAT文件的 `noise_report` 结构体中。`snr_reference_mode` 保存实际使用的参考模式，`snr_domain` 标明SNR定义在频域还是HRRP域；95%能量模式还会在 `reference_range_bin_counts` 中保存每个核验样本所选择的主距离单元数量。随机抽样种子只与噪声实现序号有关，因此同一噪声实现的不同SNR会检查相同的测试样本。

当 `reuseNoiseAcrossSnr = true` 时，同一噪声实现下不同SNR使用相同的基础噪声形状，仅改变噪声幅度，有助于减少识别率—SNR曲线的随机抖动。训练集和测试集始终使用不同随机种子。

直接运行：

```matlab
addpath('数据集生成代码');
build_noisy_classification_dataset_reusable;
```

一次生成多个SNR：

```matlab
userConfig.snrDbList = -10:5:20;
userConfig.noiseRealizations = 5;
result = build_noisy_classification_dataset_reusable(userConfig);
```

负SNR文件使用 `m` 表示负号，例如：

```text
RCS_data_C_m10dB.mat
HRRP_data_C_m10dB.mat
```

每个输出文件包含 `train_data`、`train_labels`、`test_data`、`test_labels`、`snr_db`、`noise_report` 和 `dataset_cfg`。如需报告稳定的识别率曲线，可设置多个 `noiseRealizations`，并对相同SNR下多次噪声实现的识别率计算均值和标准差。

## 15. 可视化生成的分类数据集

可复用函数位置：

```text
数据集生成代码/visualize_generated_dataset_reusable.m
```

该函数直接读取 `识别数据集` 文件夹中的 `RCS_data_*` 和 `HRRP_data_*` 文件，可选择训练集或测试集、SNR、噪声实现，以及每个类别内部需要显示的样本序号。

默认显示四类目标在测试集中的第1个样本：

```matlab
addpath('数据集生成代码');
visualize_generated_dataset_reusable;
```

常用配置：

```matlab
cfg.visualizeRcs  = true;
cfg.visualizeSpectrum = true;
cfg.visualizeHrrp = true;
cfg.band = 'C';
cfg.snrDb = 5;
cfg.noiseRealizationIndex = [];
cfg.split = 'test';
cfg.sampleIndicesWithinClass = 1;
```

当数据文件名包含 `_r01` 时，应设置：

```matlab
cfg.noiseRealizationIndex = 1;
```

显示多个样本并播放：

```matlab
userConfig.sampleIndicesWithinClass = 1:100;
userConfig.animate = true;
userConfig.pauseSeconds = 0.1;
visualize_generated_dataset_reusable(userConfig);
```

导出四类对比图和每类单独图像：

```matlab
userConfig.exportComparisonImages = true;
userConfig.exportIndividualImages = true;
userConfig.outputDir = fullfile(pwd, '可视化结果');
visualize_generated_dataset_reusable(userConfig);
```

RCS支持幅度或dB显示；HRRP默认显示 `abs(HRRP).^0.2`，四类图可共用颜色范围，纵轴默认从上到下递增。默认关闭批量导出，避免误生成大量图片。

频谱可视化会自动读取：

```text
RCS_data_fft_<频段>_<SNR><噪声实现>.mat
```

例如：

```text
RCS_data_fft_C_5dB.mat
RCS_data_fft_C_5dB_r01.mat
```

横轴优先采用频谱文件中保存的 `frequency_axis_hz`。显示方式可配置为：

```matlab
cfg.spectrumDisplayMode = 'stored';     % 按文件中保存的形式显示
cfg.spectrumDisplayMode = 'magnitude';  % 强制显示幅度
cfg.spectrumDisplayMode = 'db';         % 转为dB显示
```

只显示频谱：

```matlab
userConfig.visualizeRcs = false;
userConfig.visualizeSpectrum = true;
userConfig.visualizeHrrp = false;
visualize_generated_dataset_reusable(userConfig);
```

## 16. 从RCS时域数据集生成频谱数据集

可复用函数位置：

```text
数据集生成代码/build_rcs_spectrum_dataset_reusable.m
```

该函数默认扫描 `识别数据集` 中的 `RCS_data_C_*.mat`，沿每条样本的慢时间维进行FFT，并保留训练/测试划分和标签。输出文件名在原文件前缀中插入 `fft`，其余频段、SNR、噪声实现或种子后缀保持不变：

```text
RCS_data_C_5dB.mat
→ RCS_data_fft_C_5dB.mat

RCS_data_C_5dB_r01.mat
→ RCS_data_fft_C_5dB_r01.mat
```

直接运行：

```matlab
addpath('数据集生成代码');
build_rcs_spectrum_dataset_reusable;
```

主要配置：

```matlab
cfg.band = 'C';
cfg.prfHz = 20;
cfg.removeMean = false;
cfg.windowType = 'none';
cfg.fftLength = [];
cfg.applyFftShift = true;
cfg.scaleByFftLength = true;
cfg.spectrumRepresentation = 'magnitude';
```

支持的频谱表示包括：

- `complex`：复数FFT结果。
- `magnitude`：幅度谱，默认选项。
- `power`：功率谱。
- `db`：dB频谱。

输入文件留空时自动扫描；也可以只转换指定文件：

```matlab
userConfig.inputFiles = { ...
    'RCS_data_C_5dB.mat', ...
    'RCS_data_C_5dB_r01.mat'};
build_rcs_spectrum_dataset_reusable(userConfig);
```

每个频谱数据集继续使用 `train_data`、`train_labels`、`test_data` 和 `test_labels`，方便复用原训练程序，并额外保存 `frequency_axis_hz`、`spectrum_cfg`、`source_file` 及源数据集中的SNR和噪声核验元数据。默认不覆盖已存在的频谱文件。

## 17. 查看一个样本的对应数据

可复用函数位置：

```text
数据集生成代码/visualize_single_sample_reusable.m
```

该函数锁定一个分类样本，分别创建三个窗口显示其RCS慢时间序列、RCS频谱和HRRP。程序会检查三个数据集的标签及样本顺序是否完全一致，避免显示不同样本；大型HRRP文件只读取选中的一个二维样本，不会整体载入内存。

按“类别标签 + 类内样本序号”选择，例如查看测试集中类别2的第10个样本：

```matlab
cfg.band = 'C';
cfg.snrDb = 5;
cfg.noiseRealizationIndex = 1;
cfg.split = 'test';
cfg.selectionMode = 'class';
cfg.classLabel = 2;
cfg.sampleIndexWithinClass = 10;
visualize_single_sample_reusable(cfg);
```

也可以按数据集中的全局序号选择：

```matlab
cfg.selectionMode = 'global';
cfg.globalSampleIndex = 100;
visualize_single_sample_reusable(cfg);
```

三种显示内容可以分别控制：

```matlab
cfg.visualizeRcs = true;
cfg.visualizeSpectrum = true;
cfg.visualizeHrrp = true;
```

如果文件不在默认目录中，可以设置 `cfg.inputDir`，也可以直接通过 `cfg.rcsFile`、`cfg.spectrumFile` 和 `cfg.hrrpFile` 指定完整文件路径。设置 `cfg.exportFigure=true` 可以把三个窗口分别保存到 `可视化结果/单样本`。
