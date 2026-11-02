%% FFE 回波 / RCS 数据读取与处理
% MATLAB R2016b 及以上版本可直接运行本脚本（脚本末尾含本地函数）。

clc;
clear;
close all;

%% ======================== 集中配置区 ========================
cfg.inputDir            = fullfile('宽带数据', 'C');
cfg.inputName           = 'Test1_FarField1_C.ffe';
cfg.outputDir           = '.';
cfg.outputName          = 'pingdizhui.mat';
cfg.outputVariable      = 'data';

% 数据类型：'echo' 读取复回波；'rcs' 读取 RCS 列
cfg.dataType            = 'echo';
cfg.echoRealColumn      = 5;
cfg.echoImagColumn      = 6;
cfg.rcsColumn           = 9;

% FFE 文件抽取规则
% 原程序先跳过 6 行，之后每 12 行取最后一行。
% 若另一种 FFE 格式每个记录占 10 行，只需把 linesPerRecord 改为 10。
cfg.initialSkipLines    = 6;
cfg.linesPerRecord      = 12;

% 扫描参数
cfg.angleStartDeg       = 0;
cfg.angleEndDeg         = 180;
cfg.samplesPerFrequency = 1801;
cfg.frequencyCount      = 101;       % 设为 [] 时根据记录总数自动推断
cfg.frequencyIndices    = ':';       % 例如 1、[1 3 5] 或 ':'
cfg.useSymmetry         = true;    % 将 0~180° 镜像扩展为 0~360°

% 保存与绘图
cfg.saveResult          = true;
cfg.saveAngles          = false;   % true 时同时保存 anglesDeg 和 cfg
cfg.showPlot            = true;
cfg.dbFactor            = 10;      % 保留原程序算法；复幅度常用 20，功率/RCS 常用 10
cfg.plotTitle           = '球顶锥体静态数据曲线（窄带）';
cfg.xTickStepDeg        = 40;

% 宽带 HRRP（高分辨距离像）
% 使用宽带数据时，需同时将 frequencyCount 设为实际频点数，
% 并将 frequencyIndices 设为 ':' 或连续递增的频点序号。
cfg.hrrpEnabled         = true;
cfg.hrrpShowPlot        = true;
cfg.hrrpExponent        = 0.2;        % 图像动态范围压缩指数
cfg.hrrpNormalize       = false;      % 是否将图像归一化至 0~1
cfg.hrrpColormap        = 'hot';
cfg.hrrpTitle           = '静态 HRRP';
cfg.hrrpXTickStepDeg    = 60;
cfg.hrrpYTickStep       = 20;
cfg.hrrpRangeBinLimits  = [];         % [] 表示自动显示全部距离单元

%% ======================== 主处理流程 ========================
inputFile  = fullfile(cfg.inputDir, cfg.inputName);
outputFile = fullfile(cfg.outputDir, cfg.outputName);

raw = readFixedStrideNumericRows( ...
    inputFile, cfg.initialSkipLines, cfg.linesPerRecord);

switch lower(cfg.dataType)
    case 'echo'
        requiredColumn = max(cfg.echoRealColumn, cfg.echoImagColumn);
        assert(size(raw, 2) >= requiredColumn, ...
            'FFE 数据只有 %d 列，无法读取第 %d 列。', size(raw, 2), requiredColumn);
        values = raw(:, cfg.echoRealColumn) + 1i * raw(:, cfg.echoImagColumn);

    case 'rcs'
        assert(size(raw, 2) >= cfg.rcsColumn, ...
            'FFE 数据只有 %d 列，无法读取第 %d 列。', size(raw, 2), cfg.rcsColumn);
        values = raw(:, cfg.rcsColumn);

    otherwise
        error('cfg.dataType 只能是 ''echo'' 或 ''rcs''。');
end

recordCount = numel(values);
if isempty(cfg.frequencyCount)
    assert(mod(recordCount, cfg.samplesPerFrequency) == 0, ...
        ['记录数 %d 不能被每频点采样数 %d 整除，无法自动推断频点数。' ...
         '请检查 FFE 抽取参数。'], recordCount, cfg.samplesPerFrequency);
    frequencyCount = recordCount / cfg.samplesPerFrequency;
else
    frequencyCount = cfg.frequencyCount;
    expectedCount = cfg.samplesPerFrequency * frequencyCount;
    assert(recordCount == expectedCount, ...
        ['实际抽取 %d 条记录，但配置要求 %d×%d=%d 条。' ...
         '请检查 samplesPerFrequency、frequencyCount 或 linesPerRecord。'], ...
        recordCount, cfg.samplesPerFrequency, frequencyCount, expectedCount);
end

% 每列对应一个频点，每行对应一个角度。
halfData = reshape(values, cfg.samplesPerFrequency, frequencyCount);

if ischar(cfg.frequencyIndices) || isstring(cfg.frequencyIndices)
    assert(strcmp(char(cfg.frequencyIndices), ':'), ...
        '字符型 frequencyIndices 只支持 '':''。');
    selectedFrequencyIndices = 1:frequencyCount;
    selectedHalfData = halfData;
else
    validateattributes(cfg.frequencyIndices, {'numeric'}, ...
        {'vector', 'integer', 'positive'});
    selectedFrequencyIndices = cfg.frequencyIndices(:).';
    assert(all(selectedFrequencyIndices <= frequencyCount), ...
        'frequencyIndices 超出有效范围 1~%d。', frequencyCount);
    assert(numel(unique(selectedFrequencyIndices)) == ...
        numel(selectedFrequencyIndices), 'frequencyIndices 中不能包含重复序号。');
    selectedHalfData = halfData(:, selectedFrequencyIndices);
end

halfAnglesDeg = linspace( ...
    cfg.angleStartDeg, cfg.angleEndDeg, cfg.samplesPerFrequency).';

if cfg.useSymmetry
    % 不重复 180°，并令 360° 对应原始 0°。
    processedData = [selectedHalfData; flipud(selectedHalfData(1:end-1, :))];
    angleStepDeg = (cfg.angleEndDeg - cfg.angleStartDeg) / ...
        (cfg.samplesPerFrequency - 1);
    anglesDeg = cfg.angleStartDeg + ...
        (0:size(processedData, 1)-1).' * angleStepDeg;
else
    processedData = selectedHalfData;
    anglesDeg = halfAnglesDeg;
end

% 与原程序保持兼容：每个频点占一行，角度沿列方向排列。
% .' 是普通转置，不会像 ' 那样对复数取共轭。
data = processedData.';

% HRRP 沿频率维（data 的第 1 维）进行 IFFT。
dataHrrp = [];
hrrpImage = [];
if cfg.hrrpEnabled
    assert(strcmpi(cfg.dataType, 'echo'), ...
        ['HRRP 必须使用包含幅度和相位的复回波，请将 dataType 设为 ''echo''。' ...
         '不能直接对 RCS 功率列进行 IFFT。']);
    assert(size(data, 1) >= 2, ...
        'HRRP 至少需要 2 个频点；请读取宽带多频点数据。');
    assert(all(diff(selectedFrequencyIndices) == 1), ...
        ['HRRP 要求 frequencyIndices 为连续递增序号，例如 '':'' 或 1:101。' ...
         '非连续或乱序频点会造成距离像失真。']);
    validateattributes(cfg.hrrpExponent, {'numeric'}, ...
        {'scalar', 'real', 'positive'});

    dataHrrp = fftshift(ifft(data, [], 1), 1);
    hrrpImage = abs(dataHrrp) .^ cfg.hrrpExponent;

    if cfg.hrrpNormalize
        imageMaximum = max(hrrpImage(:));
        if imageMaximum > 0
            hrrpImage = hrrpImage ./ imageMaximum;
        end
    end
end

if cfg.saveResult
    if ~isfolder(cfg.outputDir)
        mkdir(cfg.outputDir);
    end

    saveStruct = struct();
    saveStruct.(cfg.outputVariable) = data;
    if cfg.saveAngles
        saveStruct.anglesDeg = anglesDeg.';
        saveStruct.cfg = cfg;
    end
    save(outputFile, '-struct', 'saveStruct');
end

if cfg.showPlot
    plotValuesDb = cfg.dbFactor .* log10(max(abs(processedData), realmin));

    figure('Color', 'w');
    plot(anglesDeg, plotValuesDb, 'LineWidth', 1.1);
    grid on;
    xlabel('俯仰角（°）', 'FontSize', 14);
    ylabel(sprintf('%s（dB）', upper(cfg.dataType)), 'FontSize', 14);
    title(cfg.plotTitle, 'FontSize', 14);
    xlim([anglesDeg(1), anglesDeg(end)]);
    xticks(anglesDeg(1):cfg.xTickStepDeg:anglesDeg(end));

    if size(processedData, 2) > 1
        legend(compose('频点 %d', selectedFrequencyIndices), ...
            'Location', 'best');
    end
end

if cfg.hrrpEnabled && cfg.hrrpShowPlot
    rangeBins = (0:size(dataHrrp, 1)-1).';

    figure('Color', 'w');
    imagesc(anglesDeg, rangeBins, hrrpImage);
    set(gca, 'YDir', 'reverse');
    colormap(cfg.hrrpColormap);
    colorbar;
    xlabel('俯仰角（°）', 'FontSize', 14);
    ylabel('距离单元（Range bins）', 'FontSize', 14);
    title(cfg.hrrpTitle, 'FontSize', 14);
    xlim([anglesDeg(1), anglesDeg(end)]);
    xticks(anglesDeg(1):cfg.hrrpXTickStepDeg:anglesDeg(end));

    if isempty(cfg.hrrpRangeBinLimits)
        rangeBinLimits = [rangeBins(1), rangeBins(end)];
    else
        validateattributes(cfg.hrrpRangeBinLimits, {'numeric'}, ...
            {'vector', 'numel', 2, 'real', 'finite', 'nonnan'});
        rangeBinLimits = cfg.hrrpRangeBinLimits(:).';
        assert(rangeBinLimits(1) < rangeBinLimits(2), ...
            'hrrpRangeBinLimits 必须是递增的二元素向量。');
        assert(rangeBinLimits(1) >= rangeBins(1) && ...
            rangeBinLimits(2) <= rangeBins(end), ...
            'hrrpRangeBinLimits 必须位于 %.0f~%.0f 范围内。', ...
            rangeBins(1), rangeBins(end));
    end

    ylim(rangeBinLimits);
    yticks(rangeBinLimits(1):cfg.hrrpYTickStep:rangeBinLimits(2));
end

fprintf('处理完成：%d 个频点，每个频点 %d 个输出角度点。\n', ...
    size(data, 1), size(data, 2));
if cfg.saveResult
    fprintf('结果已保存至：%s\n', outputFile);
end
if cfg.hrrpEnabled
    fprintf('HRRP 已生成：%d 个距离单元 × %d 个角度点。\n', ...
        size(dataHrrp, 1), size(dataHrrp, 2));
end

%% ======================== 本地函数 ========================
function raw = readFixedStrideNumericRows(fileName, initialSkipLines, linesPerRecord)
%READFIXEDSTRIDENUMERICROWS 按固定行间隔直接读取数值记录，无需中间 txt 文件。

    validateattributes(initialSkipLines, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative'});
    validateattributes(linesPerRecord, {'numeric'}, ...
        {'scalar', 'integer', 'positive'});

    fid = fopen(fileName, 'rt');
    assert(fid ~= -1, '无法打开 FFE 文件：%s', fileName);
    cleaner = onCleanup(@() fclose(fid));

    lineNumber = 0;
    for k = 1:initialSkipLines
        line = fgetl(fid);
        lineNumber = lineNumber + 1;
        assert(ischar(line), '文件不足 %d 行：%s', initialSkipLines, fileName);
    end

    raw = [];
    rowCount = 0;
    allocatedRows = 0;
    columnCount = [];
    reachedEOF = false;
    while ~reachedEOF
        selectedLine = '';
        linesRead = 0;
        for k = 1:linesPerRecord
            currentLine = fgetl(fid);
            if isequal(currentLine, -1)
                reachedEOF = true;
                break;
            end
            selectedLine = currentLine;
            lineNumber = lineNumber + 1;
            linesRead = linesRead + 1;
        end

        if reachedEOF
            if linesRead > 0
                warning('文件末尾存在不完整记录：只读取了 %d/%d 行。', ...
                    linesRead, linesPerRecord);
            end
            break;
        end

        numericRow = sscanf(selectedLine, '%f').';
        assert(~isempty(numericRow), ...
            '第 %d 行不是有效数值记录，请检查 linesPerRecord。', lineNumber);

        if isempty(columnCount)
            columnCount = numel(numericRow);
            allocatedRows = 1024;
            raw = zeros(allocatedRows, columnCount);
        else
            assert(numel(numericRow) == columnCount, ...
                '第 %d 行有 %d 列，预期 %d 列。', ...
                lineNumber, numel(numericRow), columnCount);
        end

        rowCount = rowCount + 1;
        if rowCount > allocatedRows
            allocatedRows = allocatedRows * 2;
            raw(allocatedRows, columnCount) = 0; %#ok<AGROW>
        end
        raw(rowCount, :) = numericRow; %#ok<AGROW>
    end

    assert(rowCount > 0, ...
        '未读取到任何数据，请检查 initialSkipLines 和 linesPerRecord。');
    raw = raw(1:rowCount, :);
end
