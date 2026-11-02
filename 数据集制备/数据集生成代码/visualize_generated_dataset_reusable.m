function result = visualize_generated_dataset_reusable(userConfig)
%% 读取并可视化生成的RCS、RCS频谱与HRRP分类数据集
% 直接运行：
%   visualize_generated_dataset_reusable
%
% 外部覆盖示例：
%   userConfig.snrDb = 10;
%   userConfig.split = 'test';
%   userConfig.sampleIndicesWithinClass = 1:10;
%   userConfig.animate = true;
%   visualize_generated_dataset_reusable(userConfig);

%% ======================== 集中配置区 ========================
projectRoot = fileparts(fileparts(mfilename('fullpath')));

% 数据类型开关。
cfg.visualizeRcs  = true;
cfg.visualizeSpectrum = false;
cfg.visualizeHrrp = false;

% 数据集文件配置。
cfg.inputDir = fullfile(projectRoot, '识别数据集','data')
cfg.band = 'X';
cfg.snrDb = 5;
% [] 对应无_r01后缀的单噪声实现文件；1对应_r01，2对应_r02。
cfg.noiseRealizationIndex = [];
% 留空时根据频段、SNR和噪声实现自动生成文件名。
cfg.rcsFile  = '';
cfg.spectrumFile = '';
cfg.hrrpFile = '';

% 'train' 或 'test'。
cfg.split = 'train';

% 类别配置。
cfg.classLabels = 0:3;
cfg.classNames = {'0','1','2','3'};

% 每个类别内部需要显示的样本序号，例如1、[1 10 100]或1:20。
cfg.sampleIndicesWithinClass = 1:363;

% 坐标和显示设置。
cfg.prfHz = 20;
cfg.useTimeSeconds = true;
cfg.rcsDisplayMode = 'magnitude';  % 'magnitude' 或 'db'
cfg.rcsDbFactor = 20;
cfg.rcsCommonYLimits = true;
% 'stored'按文件中保存的形式显示；也可强制'magnitude'或'db'。
cfg.spectrumDisplayMode = 'stored';
cfg.spectrumDbFactor = 20;
cfg.spectrumCommonYLimits = true;
cfg.hrrpDisplayPower = 1;
cfg.hrrpNormalizeEachImage = false;
cfg.hrrpCommonColorLimits = false;
cfg.hrrpColormap = 'hot';
cfg.hrrpYDirection = 'reverse';    % 距离单元从上到下递增

% 动画与窗口。
cfg.showFigures = true;
cfg.animate = true;
cfg.pauseSeconds = 0.5;

% 图像导出。默认关闭，避免一次生成大量文件。
cfg.exportComparisonImages = false;
cfg.exportIndividualImages = false;
cfg.outputDir = fullfile(projectRoot, '可视化结果');
cfg.imageFormat = 'png';

%% ======================== 合并并检查配置 ========================
if nargin < 1 || isempty(userConfig)
    userConfig = struct();
end
assert(isstruct(userConfig), 'userConfig必须是结构体。');
cfg = mergeStructRecursive(cfg, userConfig);

cfg.band = upper(strtrim(char(cfg.band)));
cfg.split = lower(strtrim(char(cfg.split)));
validateConfiguration(cfg);

snrTag = formatSnrTag(cfg.snrDb);
realizationTag = formatRealizationTag(cfg.noiseRealizationIndex);

if isempty(cfg.rcsFile)
    cfg.rcsFile = fullfile(cfg.inputDir, sprintf( ...
        'RCS_data_%s_%s%s.mat', cfg.band, snrTag, realizationTag));
end
if isempty(cfg.spectrumFile)
    cfg.spectrumFile = fullfile(cfg.inputDir, sprintf( ...
        'RCS_data_fft_%s_%s%s.mat', cfg.band, snrTag, realizationTag));
end
if isempty(cfg.hrrpFile)
    cfg.hrrpFile = fullfile(cfg.inputDir, sprintf( ...
        'HRRP_data_%s_%s%s.mat', cfg.band, snrTag, realizationTag));
end

if cfg.exportComparisonImages || cfg.exportIndividualImages
    if ~isfolder(cfg.outputDir)
        mkdir(cfg.outputDir);
    end
end

result = struct();
result.rcsFile = '';
result.spectrumFile = '';
result.hrrpFile = '';
result.exportedFiles = {};

%% ======================== 可视化RCS ========================
if cfg.visualizeRcs
    [rcsData, rcsLabels] = loadDatasetSplit(cfg.rcsFile, cfg.split);
    assert(ismatrix(rcsData), ...
        'RCS数据必须是“样本×慢时间”的二维矩阵。');
    assert(size(rcsData, 1) == numel(rcsLabels), ...
        'RCS样本数%d与标签数%d不一致。', ...
        size(rcsData, 1), numel(rcsLabels));

    rcsClassIndices = resolveClassSampleIndices( ...
        rcsLabels, cfg.classLabels, cfg.sampleIndicesWithinClass);
    rcsFiles = visualizeRcsData(rcsData, rcsClassIndices, cfg);
    result.rcsFile = cfg.rcsFile;
    result.exportedFiles = [result.exportedFiles, rcsFiles];
end

%% ======================== 可视化RCS频谱 ========================
if cfg.visualizeSpectrum
    [spectrumData, spectrumLabels, spectrumMetadata] = ...
        loadDatasetSplit(cfg.spectrumFile, cfg.split);
    assert(ismatrix(spectrumData), ...
        'RCS频谱数据必须是“样本×频率”的二维矩阵。');
    assert(size(spectrumData, 1) == numel(spectrumLabels), ...
        '频谱样本数%d与标签数%d不一致。', ...
        size(spectrumData, 1), numel(spectrumLabels));

    frequencyAxisHz = resolveFrequencyAxis( ...
        spectrumMetadata, size(spectrumData, 2), cfg.prfHz);
    spectrumClassIndices = resolveClassSampleIndices( ...
        spectrumLabels, cfg.classLabels, cfg.sampleIndicesWithinClass);
    spectrumFiles = visualizeSpectrumData( ...
        spectrumData, spectrumClassIndices, frequencyAxisHz, ...
        spectrumMetadata, cfg);
    result.spectrumFile = cfg.spectrumFile;
    result.exportedFiles = [result.exportedFiles, spectrumFiles];
end

%% ======================== 可视化HRRP ========================
if cfg.visualizeHrrp
    [hrrpData, hrrpLabels] = loadDatasetSplit(cfg.hrrpFile, cfg.split);
    assert(ndims(hrrpData) == 3, ...
        'HRRP数据必须是“距离单元×慢时间×样本”的三维数组。');
    assert(size(hrrpData, 3) == numel(hrrpLabels), ...
        'HRRP样本数%d与标签数%d不一致。', ...
        size(hrrpData, 3), numel(hrrpLabels));

    hrrpClassIndices = resolveClassSampleIndices( ...
        hrrpLabels, cfg.classLabels, cfg.sampleIndicesWithinClass);
    hrrpFiles = visualizeHrrpData(hrrpData, hrrpClassIndices, cfg);
    result.hrrpFile = cfg.hrrpFile;
    result.exportedFiles = [result.exportedFiles, hrrpFiles];
end

result.cfg = cfg;
fprintf('数据集可视化完成。\n');
end

%% ======================== 本地函数 ========================
function validateConfiguration(cfg)
    assert(cfg.visualizeRcs || cfg.visualizeSpectrum || cfg.visualizeHrrp, ...
        'RCS、频谱和HRRP可视化开关不能同时为false。');
    assert(numel(cfg.band) == 1, 'band必须是单个频段字符。');
    assert(any(strcmp(cfg.split, {'train', 'test'})), ...
        'split只能是''train''或''test''。');
    validateattributes(cfg.snrDb, {'numeric'}, ...
        {'scalar', 'real', 'finite'});
    if ~isempty(cfg.noiseRealizationIndex)
        validateattributes(cfg.noiseRealizationIndex, {'numeric'}, ...
            {'scalar', 'integer', 'positive'});
    end
    validateattributes(cfg.classLabels, {'numeric'}, ...
        {'vector', 'integer', 'nonnegative', 'nonempty'});
    assert(numel(cfg.classLabels) == numel(cfg.classNames), ...
        'classLabels和classNames数量必须一致。');
    assert(numel(cfg.classLabels) <= 4, ...
        '当前2×2对比布局最多支持4个类别。');
    validateattributes(cfg.sampleIndicesWithinClass, {'numeric'}, ...
        {'vector', 'integer', 'positive', 'nonempty'});
    validateattributes(cfg.prfHz, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'});
    assert(any(strcmpi(cfg.rcsDisplayMode, {'magnitude', 'db'})), ...
        'rcsDisplayMode只能是''magnitude''或''db''。');
    validateattributes(cfg.rcsDbFactor, {'numeric'}, ...
        {'scalar', 'real', 'positive'});
    assert(any(strcmpi(cfg.spectrumDisplayMode, ...
        {'stored', 'magnitude', 'db'})), ...
        'spectrumDisplayMode只能是''stored''、''magnitude''或''db''。');
    validateattributes(cfg.spectrumDbFactor, {'numeric'}, ...
        {'scalar', 'real', 'positive'});
    validateattributes(cfg.hrrpDisplayPower, {'numeric'}, ...
        {'scalar', 'real', 'positive'});
    assert(any(strcmpi(cfg.hrrpYDirection, {'normal', 'reverse'})), ...
        'hrrpYDirection只能是''normal''或''reverse''。');
    validateattributes(cfg.pauseSeconds, {'numeric'}, ...
        {'scalar', 'real', 'nonnegative'});
    assert(any(strcmpi(cfg.imageFormat, {'png', 'jpg', 'tif', 'pdf'})), ...
        'imageFormat只支持png、jpg、tif或pdf。');
end

function [data, labels, metadata] = loadDatasetSplit(fileName, split)
    assert(isfile(fileName), '数据集文件不存在：%s', fileName);
    dataVariable = [split, '_data'];
    labelVariable = [split, '_labels'];
    fileInfo = whos('-file', fileName);
    availableNames = {fileInfo.name};
    assert(ismember(dataVariable, availableNames), ...
        '文件中不存在变量%s：%s', dataVariable, fileName);
    assert(ismember(labelVariable, availableNames), ...
        '文件中不存在变量%s：%s', labelVariable, fileName);
    variablesToLoad = {dataVariable, labelVariable};
    optionalNames = {'frequency_axis_hz', 'spectrum_cfg'};
    for index = 1:numel(optionalNames)
        if ismember(optionalNames{index}, availableNames)
            variablesToLoad{end + 1} = optionalNames{index}; %#ok<AGROW>
        end
    end
    loaded = load(fileName, variablesToLoad{:});
    data = loaded.(dataVariable);
    loadedLabels = loaded.(labelVariable);
    labels = double(loadedLabels(:).');
    assert(isnumeric(data) && ~isempty(data), ...
        '变量%s必须是非空数值数组。', dataVariable);
    assert(all(isfinite(data(:))), '数据中包含NaN或Inf：%s', fileName);
    metadata = struct();
    for index = 1:numel(optionalNames)
        if isfield(loaded, optionalNames{index})
            metadata.(optionalNames{index}) = loaded.(optionalNames{index});
        end
    end
end

function classIndices = resolveClassSampleIndices( ...
        labels, classLabels, sampleIndicesWithinClass)
    classIndices = zeros(numel(classLabels), numel(sampleIndicesWithinClass));
    maximumRequestedIndex = max(sampleIndicesWithinClass);
    for classPosition = 1:numel(classLabels)
        availableIndices = find(labels == classLabels(classPosition));
        assert(numel(availableIndices) >= maximumRequestedIndex, ...
            '标签%d只有%d个样本，无法读取类内第%d个样本。', ...
            classLabels(classPosition), numel(availableIndices), ...
            maximumRequestedIndex);
        classIndices(classPosition, :) = ...
            availableIndices(sampleIndicesWithinClass);
    end
end

function exportedFiles = visualizeRcsData(data, classIndices, cfg)
    exportedFiles = {};
    sampleCount = size(classIndices, 2);
    classCount = size(classIndices, 1);
    timeAxis = buildTimeAxis(size(data, 2), cfg);
    figureVisibility = resolveFigureVisibility(cfg.showFigures);
    comparisonFigure = figure('Color', 'w', 'Visible', figureVisibility, ...
        'Name', 'RCS数据集可视化');

    for samplePosition = 1:sampleCount
        clf(comparisonFigure);
        displayValues = cell(1, classCount);
        globalMinimum = inf;
        globalMaximum = -inf;

        for classPosition = 1:classCount
            sampleIndex = classIndices(classPosition, samplePosition);
            magnitude = abs(data(sampleIndex, :));
            if strcmpi(cfg.rcsDisplayMode, 'db')
                displayValues{classPosition} = cfg.rcsDbFactor .* ...
                    log10(max(magnitude, realmin));
            else
                displayValues{classPosition} = magnitude;
            end
            globalMinimum = min(globalMinimum, min(displayValues{classPosition}));
            globalMaximum = max(globalMaximum, max(displayValues{classPosition}));
        end

        for classPosition = 1:classCount
            subplot(2, 2, classPosition);
            plot(timeAxis, displayValues{classPosition}, 'LineWidth', 1.1);
            grid on;
            xlabel(resolveTimeLabel(cfg));
            if strcmpi(cfg.rcsDisplayMode, 'db')
                ylabel('回波幅度（dB）');
            else
                ylabel('回波幅度');
            end
            title(sprintf('%s：类内样本%d', ...
                cfg.classNames{classPosition}, ...
                cfg.sampleIndicesWithinClass(samplePosition)));
            xlim([timeAxis(1), timeAxis(end)]);
            if cfg.rcsCommonYLimits && globalMaximum > globalMinimum
                ylim([globalMinimum, globalMaximum]);
            end
        end
        sgtitle(sprintf('%s集 RCS，%s，SNR=%g dB', ...
            upper(cfg.split), cfg.band, cfg.snrDb));
        drawnow;

        if cfg.exportComparisonImages
            outputFolder = fullfile(cfg.outputDir, 'RCS', 'comparison');
            ensureFolder(outputFolder);
            outputFile = fullfile(outputFolder, sprintf( ...
                'RCS_%s_%s_%s_sample%04d.%s', ...
                cfg.band, formatSnrTag(cfg.snrDb), cfg.split, ...
                cfg.sampleIndicesWithinClass(samplePosition), cfg.imageFormat));
            saveas(comparisonFigure, outputFile);
            exportedFiles{end + 1} = outputFile; %#ok<AGROW>
        end

        if cfg.exportIndividualImages
            individualFiles = exportIndividualRcs( ...
                timeAxis, displayValues, samplePosition, cfg);
            exportedFiles = [exportedFiles, individualFiles]; %#ok<AGROW>
        end

        if cfg.animate && samplePosition < sampleCount
            pause(cfg.pauseSeconds);
        end
    end
end

function frequencyAxisHz = resolveFrequencyAxis(metadata, frequencyCount, prfHz)
    if isfield(metadata, 'frequency_axis_hz')
        frequencyAxisHz = double(metadata.frequency_axis_hz(:).');
        assert(numel(frequencyAxisHz) == frequencyCount, ...
            'frequency_axis_hz长度%d与频谱列数%d不一致。', ...
            numel(frequencyAxisHz), frequencyCount);
        return;
    end

    applyFftShift = true;
    if isfield(metadata, 'spectrum_cfg') && ...
            isfield(metadata.spectrum_cfg, 'applyFftShift')
        applyFftShift = metadata.spectrum_cfg.applyFftShift;
    end
    if applyFftShift
        frequencyBins = -floor(frequencyCount/2):ceil(frequencyCount/2)-1;
    else
        frequencyBins = 0:frequencyCount-1;
    end
    frequencyAxisHz = frequencyBins .* (prfHz / frequencyCount);
end

function exportedFiles = visualizeSpectrumData( ...
        data, classIndices, frequencyAxisHz, metadata, cfg)
    exportedFiles = {};
    sampleCount = size(classIndices, 2);
    classCount = size(classIndices, 1);
    figureVisibility = resolveFigureVisibility(cfg.showFigures);
    comparisonFigure = figure('Color', 'w', 'Visible', figureVisibility, ...
        'Name', 'RCS频谱数据集可视化');

    for samplePosition = 1:sampleCount
        clf(comparisonFigure);
        displayValues = cell(1, classCount);
        globalMinimum = inf;
        globalMaximum = -inf;

        for classPosition = 1:classCount
            sampleIndex = classIndices(classPosition, samplePosition);
            displayValues{classPosition} = prepareSpectrumForDisplay( ...
                data(sampleIndex, :), metadata, cfg);
            globalMinimum = min(globalMinimum, min(displayValues{classPosition}));
            globalMaximum = max(globalMaximum, max(displayValues{classPosition}));
        end

        for classPosition = 1:classCount
            subplot(2, 2, classPosition);
            plot(frequencyAxisHz, displayValues{classPosition}, ...
                'LineWidth', 1.1);
            grid on;
            xlabel('微多普勒频率（Hz）');
            ylabel(resolveSpectrumDisplayLabel(metadata, cfg));
            title(sprintf('%s：类内样本%d', ...
                cfg.classNames{classPosition}, ...
                cfg.sampleIndicesWithinClass(samplePosition)));
            xlim([frequencyAxisHz(1), frequencyAxisHz(end)]);
            if cfg.spectrumCommonYLimits && globalMaximum > globalMinimum
                ylim([globalMinimum, globalMaximum]);
            end
        end
        sgtitle(sprintf('%s集 RCS频谱，%s，SNR=%g dB', ...
            upper(cfg.split), cfg.band, cfg.snrDb));
        drawnow;

        if cfg.exportComparisonImages
            outputFolder = fullfile(cfg.outputDir, 'Spectrum', 'comparison');
            ensureFolder(outputFolder);
            outputFile = fullfile(outputFolder, sprintf( ...
                'RCS_fft_%s_%s_%s_sample%04d.%s', ...
                cfg.band, formatSnrTag(cfg.snrDb), cfg.split, ...
                cfg.sampleIndicesWithinClass(samplePosition), cfg.imageFormat));
            saveas(comparisonFigure, outputFile);
            exportedFiles{end + 1} = outputFile; %#ok<AGROW>
        end

        if cfg.exportIndividualImages
            individualFiles = exportIndividualSpectrum( ...
                frequencyAxisHz, displayValues, samplePosition, ...
                metadata, cfg);
            exportedFiles = [exportedFiles, individualFiles]; %#ok<AGROW>
        end

        if cfg.animate && samplePosition < sampleCount
            pause(cfg.pauseSeconds);
        end
    end
end

function displayValue = prepareSpectrumForDisplay(sample, metadata, cfg)
    mode = lower(cfg.spectrumDisplayMode);
    storedRepresentation = resolveStoredSpectrumRepresentation(metadata);
    switch mode
        case 'stored'
            if ~isreal(sample) || strcmp(storedRepresentation, 'complex')
                displayValue = abs(sample);
            else
                displayValue = sample;
            end
        case 'magnitude'
            displayValue = abs(sample);
        case 'db'
            if strcmp(storedRepresentation, 'db') && isreal(sample)
                displayValue = sample;
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

function label = resolveSpectrumDisplayLabel(metadata, cfg)
    if strcmpi(cfg.spectrumDisplayMode, 'db')
        label = '频谱（dB）';
        return;
    elseif strcmpi(cfg.spectrumDisplayMode, 'magnitude')
        label = '频谱幅度';
        return;
    end

    switch resolveStoredSpectrumRepresentation(metadata)
        case 'power'
            label = '频谱功率';
        case 'db'
            label = '频谱（dB）';
        otherwise
            label = '频谱幅度';
    end
end

function exportedFiles = visualizeHrrpData(data, classIndices, cfg)
    exportedFiles = {};
    sampleCount = size(classIndices, 2);
    classCount = size(classIndices, 1);
    timeAxis = buildTimeAxis(size(data, 2), cfg);
    rangeBins = 0:size(data, 1)-1;
    figureVisibility = resolveFigureVisibility(cfg.showFigures);
    comparisonFigure = figure('Color', 'w', 'Visible', figureVisibility, ...
        'Name', 'HRRP数据集可视化');

    for samplePosition = 1:sampleCount
        clf(comparisonFigure);
        displayImages = cell(1, classCount);
        globalMinimum = inf;
        globalMaximum = -inf;

        for classPosition = 1:classCount
            sampleIndex = classIndices(classPosition, samplePosition);
            imageValue = abs(data(:, :, sampleIndex)) .^ cfg.hrrpDisplayPower;
            if cfg.hrrpNormalizeEachImage
                imageMaximum = max(imageValue(:));
                if imageMaximum > 0
                    imageValue = imageValue ./ imageMaximum;
                end
            end
            displayImages{classPosition} = imageValue;
            globalMinimum = min(globalMinimum, min(imageValue(:)));
            globalMaximum = max(globalMaximum, max(imageValue(:)));
        end

        for classPosition = 1:classCount
            subplot(2, 2, classPosition);
            imagesc(timeAxis, rangeBins, displayImages{classPosition});
            set(gca, 'YDir', cfg.hrrpYDirection);
            xlabel(resolveTimeLabel(cfg));
            ylabel('距离单元');
            title(sprintf('%s：类内样本%d', ...
                cfg.classNames{classPosition}, ...
                cfg.sampleIndicesWithinClass(samplePosition)));
            if cfg.hrrpCommonColorLimits && globalMaximum > globalMinimum
                caxis([globalMinimum, globalMaximum]);
            end
        end
        colormap(comparisonFigure, cfg.hrrpColormap);
        sgtitle(sprintf('%s集 HRRP，%s，SNR=%g dB', ...
            upper(cfg.split), cfg.band, cfg.snrDb));
        drawnow;

        if cfg.exportComparisonImages
            outputFolder = fullfile(cfg.outputDir, 'HRRP', 'comparison');
            ensureFolder(outputFolder);
            outputFile = fullfile(outputFolder, sprintf( ...
                'HRRP_%s_%s_%s_sample%04d.%s', ...
                cfg.band, formatSnrTag(cfg.snrDb), cfg.split, ...
                cfg.sampleIndicesWithinClass(samplePosition), cfg.imageFormat));
            saveas(comparisonFigure, outputFile);
            exportedFiles{end + 1} = outputFile; %#ok<AGROW>
        end

        if cfg.exportIndividualImages
            individualFiles = exportIndividualHrrp( ...
                timeAxis, rangeBins, displayImages, samplePosition, cfg, ...
                globalMinimum, globalMaximum);
            exportedFiles = [exportedFiles, individualFiles]; %#ok<AGROW>
        end

        if cfg.animate && samplePosition < sampleCount
            pause(cfg.pauseSeconds);
        end
    end
end

function exportedFiles = exportIndividualRcs( ...
        timeAxis, displayValues, samplePosition, cfg)
    exportedFiles = cell(1, numel(displayValues));
    for classPosition = 1:numel(displayValues)
        dataSetFolder = sprintf('%s_%s_%s', ...
            cfg.band, formatSnrTag(cfg.snrDb), cfg.split);
        outputFolder = fullfile(cfg.outputDir, 'RCS', dataSetFolder, ...
            sprintf('target%d', classPosition));
        ensureFolder(outputFolder);
        outputFile = fullfile(outputFolder, sprintf( ...
            'sample%04d.%s', ...
            cfg.sampleIndicesWithinClass(samplePosition), cfg.imageFormat));
        imageFigure = figure('Color', 'w', 'Visible', 'off');
        plot(timeAxis, displayValues{classPosition}, 'LineWidth', 1.1);
        grid on;
        xlabel(resolveTimeLabel(cfg));
        if strcmpi(cfg.rcsDisplayMode, 'db')
            ylabel('回波幅度（dB）');
        else
            ylabel('回波幅度');
        end
        title(cfg.classNames{classPosition});
        saveas(imageFigure, outputFile);
        close(imageFigure);
        exportedFiles{classPosition} = outputFile;
    end
end

function exportedFiles = exportIndividualSpectrum( ...
        frequencyAxisHz, displayValues, samplePosition, metadata, cfg)
    exportedFiles = cell(1, numel(displayValues));
    dataSetFolder = sprintf('%s_%s_%s', ...
        cfg.band, formatSnrTag(cfg.snrDb), cfg.split);
    for classPosition = 1:numel(displayValues)
        outputFolder = fullfile(cfg.outputDir, 'Spectrum', dataSetFolder, ...
            sprintf('target%d', classPosition));
        ensureFolder(outputFolder);
        outputFile = fullfile(outputFolder, sprintf( ...
            'sample%04d.%s', ...
            cfg.sampleIndicesWithinClass(samplePosition), cfg.imageFormat));
        imageFigure = figure('Color', 'w', 'Visible', 'off');
        plot(frequencyAxisHz, displayValues{classPosition}, ...
            'LineWidth', 1.1);
        grid on;
        xlabel('微多普勒频率（Hz）');
        ylabel(resolveSpectrumDisplayLabel(metadata, cfg));
        title(cfg.classNames{classPosition});
        saveas(imageFigure, outputFile);
        close(imageFigure);
        exportedFiles{classPosition} = outputFile;
    end
end

function exportedFiles = exportIndividualHrrp( ...
        timeAxis, rangeBins, displayImages, samplePosition, cfg, ...
        globalMinimum, globalMaximum)
    exportedFiles = cell(1, numel(displayImages));
    for classPosition = 1:numel(displayImages)
        dataSetFolder = sprintf('%s_%s_%s', ...
            cfg.band, formatSnrTag(cfg.snrDb), cfg.split);
        outputFolder = fullfile(cfg.outputDir, 'HRRP', dataSetFolder, ...
            sprintf('target%d', classPosition));
        ensureFolder(outputFolder);
        outputFile = fullfile(outputFolder, sprintf( ...
            'sample%04d.%s', ...
            cfg.sampleIndicesWithinClass(samplePosition), cfg.imageFormat));
        imageFigure = figure('Color', 'w', 'Visible', 'off');
        imagesc(timeAxis, rangeBins, displayImages{classPosition});
        set(gca, 'YDir', cfg.hrrpYDirection);
        colormap(imageFigure, cfg.hrrpColormap);
        colorbar;
        if cfg.hrrpCommonColorLimits && globalMaximum > globalMinimum
            caxis([globalMinimum, globalMaximum]);
        end
        xlabel(resolveTimeLabel(cfg));
        ylabel('距离单元');
        title(cfg.classNames{classPosition});
        saveas(imageFigure, outputFile);
        close(imageFigure);
        exportedFiles{classPosition} = outputFile;
    end
end

function axisValues = buildTimeAxis(timeCount, cfg)
    if cfg.useTimeSeconds
        axisValues = (0:timeCount-1) ./ cfg.prfHz;
    else
        axisValues = 0:timeCount-1;
    end
end

function label = resolveTimeLabel(cfg)
    if cfg.useTimeSeconds
        label = '慢时间（s）';
    else
        label = '慢时间采样点';
    end
end

function visibility = resolveFigureVisibility(showFigures)
    if showFigures
        visibility = 'on';
    else
        visibility = 'off';
    end
end

function ensureFolder(folderName)
    if ~isfolder(folderName)
        mkdir(folderName);
    end
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
