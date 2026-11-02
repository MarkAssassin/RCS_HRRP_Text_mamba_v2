function result = visualize_single_sample_reusable(userConfig)
%% 选定一个分类样本并查看对应的RCS、频谱和HRRP
% 直接运行：
%   visualize_single_sample_reusable
%
% 按类别选择示例：
%   cfg.classLabel = 2;
%   cfg.sampleIndexWithinClass = 10;
%   visualize_single_sample_reusable(cfg);
%
% 按数据集全局行号选择示例：
%   cfg.selectionMode = 'global';
%   cfg.globalSampleIndex = 100;
%   visualize_single_sample_reusable(cfg);

%% ======================== 集中配置区 ========================
projectRoot = fileparts(fileparts(mfilename('fullpath')));

% 需要显示的数据类型，可分别开启或关闭。
cfg.visualizeRcs      = true;
cfg.visualizeSpectrum = true;
cfg.visualizeHrrp     = true;

% 数据集位置和文件标识。
cfg.inputDir = fullfile(projectRoot, '识别数据集','dataset');
cfg.band = 'X';
cfg.snrDb = 20;
% []表示文件名没有_r01后缀；1表示_r01，2表示_r02，以此类推。
cfg.noiseRealizationIndex = [];
cfg.split = 'train';              % 'train'或'test'

% 留空时根据inputDir、band、snrDb和噪声实现自动确定文件。
cfg.rcsFile = '';
cfg.spectrumFile = '';
cfg.hrrpFile = '';

% 样本选择方式：'class'按类别选择，'global'按数据集全局行号选择。
cfg.selectionMode = 'class';
cfg.classLabel = 1;
cfg.sampleIndexWithinClass = 50;
cfg.globalSampleIndex = 100;

% 标签名称只用于标题显示，不影响样本选择。
cfg.classLabels = 0:3;
cfg.classNames = {'大圆锥', '锥柱体', '球底锥', '球底锥柱体'};

% RCS显示参数。
cfg.prfHz = 20;
cfg.useTimeSeconds = true;
cfg.rcsDisplayMode = 'magnitude'; % 'magnitude'或'db'
cfg.rcsDbFactor = 10;

% 频谱显示参数。
% 'stored'使用文件保存形式；也可设为'magnitude'或'db'。
cfg.spectrumDisplayMode = 'stored';
cfg.spectrumDbFactor = 20;

% HRRP显示参数。
cfg.hrrpDisplayPower = 1;
cfg.hrrpNormalize = false;
cfg.hrrpColormap = 'hot';
% 'reverse'使距离单元刻度从上向下递增。
cfg.hrrpYDirection = 'reverse';

% 窗口和导出。
cfg.showFigure = true;
cfg.exportFigure = false;
cfg.outputDir = fullfile(projectRoot, '可视化结果', '单样本');
cfg.imageFormat = 'png';

%% ======================== 合并并检查配置 ========================
if nargin < 1 || isempty(userConfig)
    userConfig = struct();
end
assert(isstruct(userConfig), 'userConfig必须是结构体。');
cfg = mergeStructRecursive(cfg, userConfig);

cfg.band = upper(strtrim(char(cfg.band)));
cfg.split = lower(strtrim(char(cfg.split)));
cfg.selectionMode = lower(strtrim(char(cfg.selectionMode)));
validateConfiguration(cfg);

snrTag = formatSnrTag(cfg.snrDb);
realizationTag = formatRealizationTag(cfg.noiseRealizationIndex);
if isempty(cfg.rcsFile)
    cfg.rcsFile = fullfile(cfg.inputDir, sprintf( ...
        'RCS_data_%s_%s%s.mat', cfg.band, snrTag, realizationTag));
end
if isempty(cfg.spectrumFile)
    cfg.spectrumFile = fullfile(cfg.inputDir, sprintf( ...
        'RCS_data_fft_%s_%s%s.mat', ...
        cfg.band, snrTag, realizationTag));
end
if isempty(cfg.hrrpFile)
    cfg.hrrpFile = fullfile(cfg.inputDir, sprintf( ...
        'HRRP_data_%s_%s%s.mat', cfg.band, snrTag, realizationTag));
end

%% ======================== 确定同一个样本 ========================
activeFiles = {};
activeNames = {};
if cfg.visualizeRcs
    activeFiles{end + 1} = cfg.rcsFile;
    activeNames{end + 1} = 'RCS';
end
if cfg.visualizeSpectrum
    activeFiles{end + 1} = cfg.spectrumFile;
    activeNames{end + 1} = 'RCS频谱';
end
if cfg.visualizeHrrp
    activeFiles{end + 1} = cfg.hrrpFile;
    activeNames{end + 1} = 'HRRP';
end

referenceLabels = [];
for fileIndex = 1:numel(activeFiles)
    labels = loadLabels(activeFiles{fileIndex}, cfg.split);
    if isempty(referenceLabels)
        referenceLabels = labels;
    else
        assert(isequal(labels, referenceLabels), ...
            ['%s文件的标签或样本顺序与其他数据文件不一致，' ...
             '不能保证显示的是同一个样本：%s'], ...
            activeNames{fileIndex}, activeFiles{fileIndex});
    end
end

[globalSampleIndex, selectedLabel, classSampleIndex] = ...
    resolveSelectedSample(referenceLabels, cfg);
selectedClassName = resolveClassName(selectedLabel, cfg);

%% ======================== 按需读取选定样本 ========================
rcsSample = [];
spectrumSample = [];
hrrpSample = [];
spectrumMetadata = struct();

if cfg.visualizeRcs
    rcsSample = readSelectedSample( ...
        cfg.rcsFile, cfg.split, globalSampleIndex, 'row');
    assert(isvector(rcsSample), '读取到的RCS样本不是一维序列。');
    rcsSample = reshape(rcsSample, 1, []);
end
if cfg.visualizeSpectrum
    spectrumSample = readSelectedSample( ...
        cfg.spectrumFile, cfg.split, globalSampleIndex, 'row');
    assert(isvector(spectrumSample), '读取到的频谱样本不是一维序列。');
    spectrumSample = reshape(spectrumSample, 1, []);
    spectrumMetadata = loadSpectrumMetadata(cfg.spectrumFile);
end
if cfg.visualizeHrrp
    hrrpSample = readSelectedSample( ...
        cfg.hrrpFile, cfg.split, globalSampleIndex, 'page');
    assert(ismatrix(hrrpSample), '读取到的HRRP样本不是二维矩阵。');
end

%% ======================== 绘制单样本对应数据 ========================
figureVisibility = resolveFigureVisibility(cfg.showFigure);
figureHandles = struct('rcs', [], 'spectrum', [], 'hrrp', []);
exportedFiles = {};
titleText = sprintf( ...
    '%s集全局样本%d，%s（标签%d，类内样本%d），%s，SNR=%g dB', ...
    upper(cfg.split), globalSampleIndex, selectedClassName, ...
    selectedLabel, classSampleIndex, cfg.band, cfg.snrDb);

if cfg.exportFigure && ~isfolder(cfg.outputDir)
    mkdir(cfg.outputDir);
end

if cfg.visualizeRcs
    rcsFigure = figure('Color', 'w', 'Visible', figureVisibility, ...
        'Position', [100, 100, 760, 500], ...
        'Name', '单样本RCS慢时间序列');
    axesHandle = axes(rcsFigure);
    timeAxis = buildTimeAxis(numel(rcsSample), cfg);
    rcsDisplay = prepareRcsForDisplay(rcsSample, cfg);
    plot(axesHandle, timeAxis, rcsDisplay, 'LineWidth', 1.1);
    grid(axesHandle, 'on');
    xlabel(axesHandle, resolveTimeLabel(cfg));
    ylabel(axesHandle, resolveRcsLabel(cfg));
    title(axesHandle, {'RCS慢时间序列', titleText});
    xlim(axesHandle, [timeAxis(1), timeAxis(end)]);
    figureHandles.rcs = rcsFigure;
    drawnow;

    if cfg.exportFigure
        outputFile = buildExportFileName( ...
            cfg, snrTag, realizationTag, selectedLabel, ...
            classSampleIndex, 'rcs');
        saveas(rcsFigure, outputFile);
        exportedFiles{end + 1} = outputFile;
    end
end

if cfg.visualizeSpectrum
    spectrumFigure = figure('Color', 'w', 'Visible', figureVisibility, ...
        'Position', [140, 140, 760, 500], ...
        'Name', '单样本RCS频谱');
    axesHandle = axes(spectrumFigure);
    frequencyAxisHz = resolveFrequencyAxis( ...
        spectrumMetadata, numel(spectrumSample), cfg.prfHz);
    spectrumDisplay = prepareSpectrumForDisplay( ...
        spectrumSample, spectrumMetadata, cfg);
    plot(axesHandle, frequencyAxisHz, spectrumDisplay, 'LineWidth', 1.1);
    grid(axesHandle, 'on');
    xlabel(axesHandle, '微多普勒频率（Hz）');
    ylabel(axesHandle, resolveSpectrumLabel(spectrumMetadata, cfg));
    title(axesHandle, {'RCS频谱', titleText});
    xlim(axesHandle, [frequencyAxisHz(1), frequencyAxisHz(end)]);
    figureHandles.spectrum = spectrumFigure;
    drawnow;

    if cfg.exportFigure
        outputFile = buildExportFileName( ...
            cfg, snrTag, realizationTag, selectedLabel, ...
            classSampleIndex, 'spectrum');
        saveas(spectrumFigure, outputFile);
        exportedFiles{end + 1} = outputFile;
    end
end

if cfg.visualizeHrrp
%     hrrpFigure = figure('Color', 'w', 'Visible', figureVisibility, ...
%         'Position', [180, 180, 760, 560], ...
%         'Name', '单样本HRRP');
    hrrpFigure = figure('Color', 'w', 'Visible', figureVisibility, ...
        'Name', '单样本HRRP');
    axesHandle = axes(hrrpFigure);
    hrrpDisplay = abs(hrrpSample) .^ cfg.hrrpDisplayPower;
    if cfg.hrrpNormalize
        maximumValue = max(hrrpDisplay(:));
        if maximumValue > 0
            hrrpDisplay = hrrpDisplay ./ maximumValue;
        end
    end
    hrrpTimeAxis = buildTimeAxis(size(hrrpSample, 2), cfg);
    rangeBins = 0:size(hrrpSample, 1)-1;
    imagesc(axesHandle, hrrpTimeAxis, rangeBins, hrrpDisplay);
    set(axesHandle, 'YDir', cfg.hrrpYDirection);
    colormap(axesHandle, cfg.hrrpColormap);
%     colorbar(axesHandle);
    xlabel(axesHandle, resolveTimeLabel(cfg));
    ylabel(axesHandle, '距离单元');
    yticks(0:10:100);
    title(axesHandle, {'HRRP', titleText});
    figureHandles.hrrp = hrrpFigure;
    drawnow;

    if cfg.exportFigure
        outputFile = buildExportFileName( ...
            cfg, snrTag, realizationTag, selectedLabel, ...
            classSampleIndex, 'hrrp');
        saveas(hrrpFigure, outputFile);
        exportedFiles{end + 1} = outputFile;
    end
end

%% ======================== 返回结果 ========================
result = struct();
result.globalSampleIndex = globalSampleIndex;
result.classLabel = selectedLabel;
result.className = selectedClassName;
result.sampleIndexWithinClass = classSampleIndex;
result.rcsSample = rcsSample;
result.spectrumSample = spectrumSample;
result.hrrpSample = hrrpSample;
result.figureHandles = figureHandles;
result.exportedFiles = exportedFiles;
result.cfg = cfg;

fprintf(['单样本可视化完成：%s集全局样本%d，%s，' ...
    '标签%d，类内样本%d。\n'], upper(cfg.split), globalSampleIndex, ...
    selectedClassName, selectedLabel, classSampleIndex);
end

%% ======================== 本地函数 ========================
function validateConfiguration(cfg)
    assert(cfg.visualizeRcs || cfg.visualizeSpectrum || ...
        cfg.visualizeHrrp, '三个可视化开关不能同时为false。');
    assert(numel(cfg.band) == 1, 'band必须是单个频段字符。');
    assert(any(strcmp(cfg.split, {'train', 'test'})), ...
        'split只能是''train''或''test''。');
    assert(any(strcmp(cfg.selectionMode, {'class', 'global'})), ...
        'selectionMode只能是''class''或''global''。');
    validateattributes(cfg.snrDb, {'numeric'}, ...
        {'scalar', 'real', 'finite'});
    if ~isempty(cfg.noiseRealizationIndex)
        validateattributes(cfg.noiseRealizationIndex, {'numeric'}, ...
            {'scalar', 'integer', 'positive'});
    end
    validateattributes(cfg.classLabel, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative'});
    validateattributes(cfg.sampleIndexWithinClass, {'numeric'}, ...
        {'scalar', 'integer', 'positive'});
    validateattributes(cfg.globalSampleIndex, {'numeric'}, ...
        {'scalar', 'integer', 'positive'});
    assert(numel(cfg.classLabels) == numel(cfg.classNames), ...
        'classLabels和classNames数量必须一致。');
    validateattributes(cfg.prfHz, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'});
    assert(any(strcmpi(cfg.rcsDisplayMode, {'magnitude', 'db'})), ...
        'rcsDisplayMode只能是''magnitude''或''db''。');
    assert(any(strcmpi(cfg.spectrumDisplayMode, ...
        {'stored', 'magnitude', 'db'})), ...
        'spectrumDisplayMode只能是''stored''、''magnitude''或''db''。');
    validateattributes(cfg.rcsDbFactor, {'numeric'}, ...
        {'scalar', 'real', 'positive'});
    validateattributes(cfg.spectrumDbFactor, {'numeric'}, ...
        {'scalar', 'real', 'positive'});
    validateattributes(cfg.hrrpDisplayPower, {'numeric'}, ...
        {'scalar', 'real', 'positive'});
    assert(any(strcmpi(cfg.hrrpYDirection, {'normal', 'reverse'})), ...
        'hrrpYDirection只能是''normal''或''reverse''。');
    assert(any(strcmpi(cfg.imageFormat, {'png', 'jpg', 'tif', 'pdf'})), ...
        'imageFormat只支持png、jpg、tif或pdf。');
end

function labels = loadLabels(fileName, split)
    assert(isfile(fileName), '数据集文件不存在：%s', fileName);
    labelVariable = [split, '_labels'];
    fileVariables = whos('-file', fileName);
    availableNames = {fileVariables.name};
    assert(ismember(labelVariable, availableNames), ...
        '文件中不存在变量%s：%s', labelVariable, fileName);
    loaded = load(fileName, labelVariable);
    labels = double(loaded.(labelVariable)(:).');
    assert(~isempty(labels) && all(isfinite(labels)), ...
        '标签必须是非空有限数值：%s', fileName);
end

function [globalIndex, classLabel, classIndex] = ...
        resolveSelectedSample(labels, cfg)
    switch cfg.selectionMode
        case 'class'
            classMembers = find(labels == cfg.classLabel);
            assert(~isempty(classMembers), ...
                '数据集中不存在标签%d。', cfg.classLabel);
            assert(cfg.sampleIndexWithinClass <= numel(classMembers), ...
                '标签%d只有%d个样本，无法选择类内样本%d。', ...
                cfg.classLabel, numel(classMembers), ...
                cfg.sampleIndexWithinClass);
            globalIndex = classMembers(cfg.sampleIndexWithinClass);
            classLabel = cfg.classLabel;
            classIndex = cfg.sampleIndexWithinClass;
        case 'global'
            assert(cfg.globalSampleIndex <= numel(labels), ...
                '全局样本序号%d超过数据集样本数%d。', ...
                cfg.globalSampleIndex, numel(labels));
            globalIndex = cfg.globalSampleIndex;
            classLabel = labels(globalIndex);
            classMembers = find(labels == classLabel);
            classIndex = find(classMembers == globalIndex, 1, 'first');
    end
end

function className = resolveClassName(classLabel, cfg)
    classPosition = find(cfg.classLabels == classLabel, 1, 'first');
    if isempty(classPosition)
        className = sprintf('类别%d', classLabel);
    else
        className = char(cfg.classNames{classPosition});
    end
end

function sample = readSelectedSample(fileName, split, sampleIndex, layout)
    dataVariable = [split, '_data'];
    fileVariables = whos('-file', fileName);
    availableNames = {fileVariables.name};
    variablePosition = find(strcmp(availableNames, dataVariable), 1);
    assert(~isempty(variablePosition), ...
        '文件中不存在变量%s：%s', dataVariable, fileName);
    dataSize = fileVariables(variablePosition).size;

    switch layout
        case 'row'
            assert(numel(dataSize) == 2 && sampleIndex <= dataSize(1), ...
                '%s必须是“样本×特征”的二维矩阵。', dataVariable);
            subscripts = {sampleIndex, ':'};
        case 'page'
            assert(numel(dataSize) <= 3 && sampleIndex <= dataSize(3), ...
                '%s必须是“距离单元×慢时间×样本”的三维数组。', ...
                dataVariable);
            subscripts = {':', ':', sampleIndex};
        otherwise
            error('未知的数据布局：%s', layout);
    end

    % matfile只读取所选行或页，避免把大型HRRP文件整体载入内存。
    try
        fileObject = matfile(fileName);
        sample = subsref(fileObject, ...
            substruct('.', dataVariable, '()', subscripts));
    catch readError
        warning('singleSample:PartialReadFailed', ...
            '按需读取失败，将载入变量后提取样本：%s', ...
            readError.message);
        loaded = load(fileName, dataVariable);
        completeData = loaded.(dataVariable);
        if strcmp(layout, 'row')
            sample = completeData(sampleIndex, :);
        else
            sample = completeData(:, :, sampleIndex);
        end
    end
    assert(isnumeric(sample) && all(isfinite(sample(:))), ...
        '选定样本包含NaN或Inf：%s', fileName);
end

function metadata = loadSpectrumMetadata(fileName)
    metadata = struct();
    fileVariables = whos('-file', fileName);
    availableNames = {fileVariables.name};
    optionalNames = {'frequency_axis_hz', 'spectrum_cfg'};
    namesToLoad = intersect(optionalNames, availableNames, 'stable');
    if isempty(namesToLoad)
        return;
    end
    loaded = load(fileName, namesToLoad{:});
    for nameIndex = 1:numel(namesToLoad)
        name = namesToLoad{nameIndex};
        metadata.(name) = loaded.(name);
    end
end

function timeAxis = buildTimeAxis(timeCount, cfg)
    if cfg.useTimeSeconds
        timeAxis = (0:timeCount-1) ./ cfg.prfHz;
    else
        timeAxis = 0:timeCount-1;
    end
end

function label = resolveTimeLabel(cfg)
    if cfg.useTimeSeconds
        label = '慢时间（s）';
    else
        label = '慢时间采样点';
    end
end

function displayValue = prepareRcsForDisplay(sample, cfg)
    if strcmpi(cfg.rcsDisplayMode, 'db')
        displayValue = cfg.rcsDbFactor .* ...
            log10(max(abs(sample), realmin));
    else
        displayValue = abs(sample);
    end
end

function label = resolveRcsLabel(cfg)
    if strcmpi(cfg.rcsDisplayMode, 'db')
        label = 'RCS/回波（dB）';
    else
        label = 'RCS/回波幅度';
    end
end

function frequencyAxisHz = resolveFrequencyAxis(metadata, count, defaultPrf)
    if isfield(metadata, 'frequency_axis_hz')
        frequencyAxisHz = double(metadata.frequency_axis_hz(:).');
        assert(numel(frequencyAxisHz) == count, ...
            'frequency_axis_hz长度%d与频谱长度%d不一致。', ...
            numel(frequencyAxisHz), count);
        return;
    end

    prfHz = defaultPrf;
    applyFftShift = true;
    if isfield(metadata, 'spectrum_cfg')
        if isfield(metadata.spectrum_cfg, 'prfHz')
            prfHz = metadata.spectrum_cfg.prfHz;
        end
        if isfield(metadata.spectrum_cfg, 'applyFftShift')
            applyFftShift = metadata.spectrum_cfg.applyFftShift;
        end
    end
    if applyFftShift
        frequencyBins = -floor(count/2):ceil(count/2)-1;
    else
        frequencyBins = 0:count-1;
    end
    frequencyAxisHz = frequencyBins .* (prfHz / count);
end

function displayValue = prepareSpectrumForDisplay(sample, metadata, cfg)
    storedRepresentation = resolveStoredSpectrumRepresentation(metadata);
    switch lower(cfg.spectrumDisplayMode)
        case 'stored'
            if ~isreal(sample) || strcmp(storedRepresentation, 'complex')
                displayValue = abs(sample);
            else
                displayValue = real(sample);
            end
        case 'magnitude'
            displayValue = abs(sample);
        case 'db'
            if strcmp(storedRepresentation, 'db') && isreal(sample)
                displayValue = real(sample);
            else
                displayValue = cfg.spectrumDbFactor .* ...
                    log10(max(abs(sample), realmin));
            end
    end
end

function representation = resolveStoredSpectrumRepresentation(metadata)
    representation = 'unknown';
    if isfield(metadata, 'spectrum_cfg') && ...
            isfield(metadata.spectrum_cfg, 'spectrumRepresentation')
        representation = lower(char( ...
            metadata.spectrum_cfg.spectrumRepresentation));
    end
end

function label = resolveSpectrumLabel(metadata, cfg)
    if strcmpi(cfg.spectrumDisplayMode, 'db')
        label = '频谱（dB）';
    elseif strcmpi(cfg.spectrumDisplayMode, 'magnitude')
        label = '频谱幅度';
    else
        switch resolveStoredSpectrumRepresentation(metadata)
            case 'power'
                label = '频谱功率';
            case 'db'
                label = '频谱（dB）';
            otherwise
                label = '频谱幅度';
        end
    end
end

function visibility = resolveFigureVisibility(showFigure)
    if showFigure
        visibility = 'on';
    else
        visibility = 'off';
    end
end

function outputFile = buildExportFileName( ...
        cfg, snrTag, realizationTag, classLabel, classIndex, dataType)
    outputFile = fullfile(cfg.outputDir, sprintf( ...
        'single_%s_%s%s_%s_label%d_class%04d_%s.%s', ...
        cfg.band, snrTag, realizationTag, cfg.split, classLabel, ...
        classIndex, dataType, cfg.imageFormat));
end

function tag = formatSnrTag(snrDb)
    absoluteText = sprintf('%g', abs(snrDb));
    absoluteText = strrep(absoluteText, '.', 'p');
    if snrDb < 0
        tag = ['m', absoluteText, 'dB'];
    else
        tag = [absoluteText, 'dB'];
    end
end

function tag = formatRealizationTag(index)
    if isempty(index)
        tag = '';
    else
        tag = sprintf('_r%02d', index);
    end
end

function merged = mergeStructRecursive(base, override)
    merged = base;
    names = fieldnames(override);
    for index = 1:numel(names)
        name = names{index};
        if isfield(merged, name) && isstruct(merged.(name)) && ...
                isstruct(override.(name))
            merged.(name) = mergeStructRecursive( ...
                merged.(name), override.(name));
        else
            merged.(name) = override.(name);
        end
    end
end
