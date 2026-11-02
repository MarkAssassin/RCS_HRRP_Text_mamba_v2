function result = build_rcs_spectrum_dataset_reusable(userConfig)
%% 将已生成的RCS时域分类数据集转换为频谱数据集
% 直接运行：
%   build_rcs_spectrum_dataset_reusable
%
% 默认扫描“识别数据集”中的RCS_data_C_*.mat，并生成：
%   RCS_data_C_5dB.mat     -> RCS_data_fft_C_5dB.mat
%   RCS_data_C_5dB_r01.mat -> RCS_data_fft_C_5dB_r01.mat
% 其他种子或后缀也会原样保留。

%% ======================== 集中配置区 ========================
projectRoot = fileparts(fileparts(mfilename('fullpath')));

% 输入与输出。
cfg.inputDir  = fullfile(projectRoot, '识别数据集','data');
cfg.outputDir = fullfile(projectRoot, '识别数据集','data');
cfg.band = 'X';
% 留空时自动扫描；也可指定文件名或完整路径Cell，例如：
% {'RCS_data_C_5dB.mat', 'RCS_data_C_5dB_r01.mat'}
cfg.inputFiles = {};
cfg.overwriteExisting = false;

% FFT参数。RCS数据布局必须为“样本×慢时间”。
cfg.prfHz = 20;
cfg.removeMean = false;           % 是否去除每条样本的慢时间均值
cfg.windowType = 'none';          % 'none'、'hann'或'hamming'
cfg.fftLength = [];               % [] 使用原始慢时间长度
cfg.applyFftShift = true;
cfg.scaleByFftLength = false;

% 输出表示形式：'complex'、'magnitude'、'power'或'db'。
cfg.spectrumRepresentation = 'complex';
cfg.dbFactor = 20;
cfg.dbNormalizeToPeak = false;    % true时每个样本的峰值归一化为0 dB
cfg.normalizePerSample = false;   % 对最终频谱逐样本最大值归一化

% 保存和预览。
cfg.matFileVersion = '-v7';
cfg.showProgress = true;
cfg.showPreview = false;
cfg.previewSplit = 'test';
cfg.previewSampleIndex = 1;

%% ======================== 合并并检查配置 ========================
if nargin < 1 || isempty(userConfig)
    userConfig = struct();
end
assert(isstruct(userConfig), 'userConfig必须是结构体。');
cfg = mergeStructRecursive(cfg, userConfig);

cfg.band = upper(strtrim(char(cfg.band)));
cfg.windowType = lower(strtrim(char(cfg.windowType)));
cfg.spectrumRepresentation = lower(strtrim( ...
    char(cfg.spectrumRepresentation)));
cfg.previewSplit = lower(strtrim(char(cfg.previewSplit)));
validateConfiguration(cfg);

if ~isfolder(cfg.outputDir)
    mkdir(cfg.outputDir);
end

inputFiles = resolveInputFiles(cfg);
assert(~isempty(inputFiles), ...
    '没有找到需要转换的RCS数据集文件：%s', cfg.inputDir);

result = struct();
result.inputFiles = inputFiles;
result.outputFiles = {};
result.skippedFiles = {};

%% ======================== 逐文件生成频谱数据集 ========================
for fileIndex = 1:numel(inputFiles)
    inputFile = inputFiles{fileIndex};
    [~, inputName, inputExtension] = fileparts(inputFile);
    inputFileName = [inputName, inputExtension];
    outputFileName = buildOutputFileName(inputFileName);
    outputFile = fullfile(cfg.outputDir, outputFileName);

    if isfile(outputFile) && ~cfg.overwriteExisting
        result.skippedFiles{end + 1} = outputFile;
        if cfg.showProgress
            fprintf('已存在，跳过：%s\n', outputFile);
        end
        continue;
    end

    source = loadRcsDataset(inputFile);
    [trainSpectrum, frequencyAxisHz] = transformRcsToSpectrum( ...
        source.train_data, cfg);
    [testSpectrum, testFrequencyAxisHz] = transformRcsToSpectrum( ...
        source.test_data, cfg);
    assert(isequal(frequencyAxisHz, testFrequencyAxisHz), ...
        '训练集和测试集的频率轴不一致。');

    train_data = trainSpectrum;
    train_labels = source.train_labels;
    test_data = testSpectrum;
    test_labels = source.test_labels;
    frequency_axis_hz = frequencyAxisHz;
    spectrum_cfg = cfg;
    source_file = inputFile;

    optionalMetadata = copyOptionalMetadata(source);
    saveSpectrumDataset(outputFile, train_data, train_labels, ...
        test_data, test_labels, frequency_axis_hz, spectrum_cfg, ...
        source_file, optionalMetadata, cfg.matFileVersion);

    result.outputFiles{end + 1} = outputFile;
    if cfg.showProgress
        fprintf(['频谱数据集已生成：%s\n' ...
            '  训练尺寸：%s，测试尺寸：%s，频率范围：%.3f~%.3f Hz。\n'], ...
            outputFile, mat2str(size(train_data)), mat2str(size(test_data)), ...
            frequency_axis_hz(1), frequency_axis_hz(end));
    end

    if cfg.showPreview
        previewSpectrum(inputFileName, train_data, test_data, ...
            frequency_axis_hz, cfg);
    end
end

result.cfg = cfg;
fprintf('RCS频谱数据集转换完成。\n');
end

%% ======================== 本地函数 ========================
function validateConfiguration(cfg)
    assert(numel(cfg.band) == 1 || strcmp(cfg.band, '*'), ...
        'band必须是单个频段字符或''*''。');
    validateattributes(cfg.prfHz, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'});
    assert(any(strcmp(cfg.windowType, {'none', 'hann', 'hamming'})), ...
        'windowType只能是''none''、''hann''或''hamming''。');
    if ~isempty(cfg.fftLength)
        validateattributes(cfg.fftLength, {'numeric'}, ...
            {'scalar', 'integer', 'positive'});
    end
    assert(any(strcmp(cfg.spectrumRepresentation, ...
        {'complex', 'magnitude', 'power', 'db'})), ...
        'spectrumRepresentation配置不正确。');
    validateattributes(cfg.dbFactor, {'numeric'}, ...
        {'scalar', 'real', 'positive'});
    assert(any(strcmp(cfg.previewSplit, {'train', 'test'})), ...
        'previewSplit只能是''train''或''test''。');
    validateattributes(cfg.previewSampleIndex, {'numeric'}, ...
        {'scalar', 'integer', 'positive'});
    assert(any(strcmp(cfg.matFileVersion, {'-v7', '-v7.3'})), ...
        'matFileVersion只能是''-v7''或''-v7.3''。');
end

function inputFiles = resolveInputFiles(cfg)
    if isempty(cfg.inputFiles)
        if strcmp(cfg.band, '*')
            pattern = 'RCS_data_*.mat';
        else
            pattern = sprintf('RCS_data_%s_*.mat', cfg.band);
        end
        fileListing = dir(fullfile(cfg.inputDir, pattern));
        fileNames = {fileListing.name};
        isSpectrumOutput = startsWith(fileNames, 'RCS_data_fft_');
        fileListing = fileListing(~isSpectrumOutput);
        [~, order] = sort({fileListing.name});
        fileListing = fileListing(order);
        inputFiles = arrayfun(@(item) fullfile(item.folder, item.name), ...
            fileListing, 'UniformOutput', false);
    else
        assert(iscell(cfg.inputFiles), 'inputFiles必须是Cell数组。');
        inputFiles = cell(size(cfg.inputFiles));
        for index = 1:numel(cfg.inputFiles)
            configuredFile = char(cfg.inputFiles{index});
            if isfile(configuredFile)
                inputFiles{index} = configuredFile;
            else
                inputFiles{index} = fullfile(cfg.inputDir, configuredFile);
            end
            assert(isfile(inputFiles{index}), ...
                '指定的输入文件不存在：%s', inputFiles{index});
        end
    end
end

function outputFileName = buildOutputFileName(inputFileName)
    sourcePrefix = 'RCS_data_';
    assert(startsWith(inputFileName, sourcePrefix), ...
        '输入文件名必须以%s开头：%s', sourcePrefix, inputFileName);
    remainingName = inputFileName(numel(sourcePrefix) + 1:end);
    outputFileName = ['RCS_data_fft_', remainingName];
end

function source = loadRcsDataset(fileName)
    requiredNames = {'train_data', 'train_labels', ...
                     'test_data', 'test_labels'};
    fileInfo = whos('-file', fileName);
    availableNames = {fileInfo.name};
    missingNames = setdiff(requiredNames, availableNames);
    assert(isempty(missingNames), '文件%s缺少变量：%s', ...
        fileName, strjoin(missingNames, ', '));
    source = load(fileName);
    assert(ismatrix(source.train_data) && ismatrix(source.test_data), ...
        'RCS训练集和测试集必须是“样本×慢时间”的二维矩阵。');
    assert(size(source.train_data, 1) == numel(source.train_labels), ...
        '训练样本数与训练标签数不一致：%s', fileName);
    assert(size(source.test_data, 1) == numel(source.test_labels), ...
        '测试样本数与测试标签数不一致：%s', fileName);
    assert(size(source.train_data, 2) == size(source.test_data, 2), ...
        '训练集和测试集的慢时间长度不一致：%s', fileName);
    assert(all(isfinite(source.train_data(:))) && ...
           all(isfinite(source.test_data(:))), ...
        'RCS数据中包含NaN或Inf：%s', fileName);
end

function [spectrumData, frequencyAxisHz] = ...
        transformRcsToSpectrum(timeData, cfg)
    timeCount = size(timeData, 2);
    fftLength = cfg.fftLength;
    if isempty(fftLength)
        fftLength = timeCount;
    end
    assert(fftLength >= timeCount, ...
        'fftLength不能小于原始慢时间长度%d。', timeCount);

    processedData = timeData;
    if cfg.removeMean
        processedData = processedData - mean(processedData, 2);
    end

    window = buildWindow(timeCount, cfg.windowType);
    processedData = processedData .* window;
    complexSpectrum = fft(processedData, fftLength, 2);
    if cfg.scaleByFftLength
        complexSpectrum = complexSpectrum ./ fftLength;
    end
    if cfg.applyFftShift
        complexSpectrum = fftshift(complexSpectrum, 2);
    end

    switch cfg.spectrumRepresentation
        case 'complex'
            spectrumData = complexSpectrum;
        case 'magnitude'
            spectrumData = abs(complexSpectrum);
        case 'power'
            spectrumData = abs(complexSpectrum).^2;
        case 'db'
            spectrumData = cfg.dbFactor .* ...
                log10(max(abs(complexSpectrum), realmin));
            if cfg.dbNormalizeToPeak
                spectrumData = spectrumData - max(spectrumData, [], 2);
            end
    end

    if cfg.normalizePerSample
        maximumValues = max(abs(spectrumData), [], 2);
        maximumValues(maximumValues == 0) = 1;
        spectrumData = spectrumData ./ maximumValues;
    end

    if cfg.applyFftShift
        frequencyBins = -floor(fftLength/2):ceil(fftLength/2)-1;
    else
        frequencyBins = 0:fftLength-1;
    end
    frequencyAxisHz = frequencyBins .* (cfg.prfHz / fftLength);
end

function window = buildWindow(sampleCount, windowType)
    sampleIndex = 0:sampleCount-1;
    switch windowType
        case 'none'
            window = ones(1, sampleCount);
        case 'hann'
            window = 0.5 - 0.5 .* cos(2*pi .* sampleIndex ./ sampleCount);
        case 'hamming'
            window = 0.54 - 0.46 .* cos(2*pi .* sampleIndex ./ sampleCount);
        otherwise
            error('不支持的窗函数：%s', windowType);
    end
end

function optionalMetadata = copyOptionalMetadata(source)
    optionalMetadata = struct();
    names = {'snr_db', 'noise_report', 'dataset_cfg'};
    outputNames = {'source_snr_db', 'source_noise_report', ...
                   'source_dataset_cfg'};
    for index = 1:numel(names)
        if isfield(source, names{index})
            optionalMetadata.(outputNames{index}) = source.(names{index});
        end
    end
end

function saveSpectrumDataset(fileName, trainData, trainLabels, ...
        testData, testLabels, frequencyAxisHz, spectrumCfg, ...
        sourceFile, optionalMetadata, matVersion)
    train_data = trainData;
    train_labels = trainLabels;
    test_data = testData;
    test_labels = testLabels;
    frequency_axis_hz = frequencyAxisHz;
    spectrum_cfg = spectrumCfg;
    source_file = sourceFile;
    saveStruct = struct();
    saveStruct.train_data = train_data;
    saveStruct.train_labels = train_labels;
    saveStruct.test_data = test_data;
    saveStruct.test_labels = test_labels;
    saveStruct.frequency_axis_hz = frequency_axis_hz;
    saveStruct.spectrum_cfg = spectrum_cfg;
    saveStruct.source_file = source_file;
    metadataNames = fieldnames(optionalMetadata);
    for index = 1:numel(metadataNames)
        name = metadataNames{index};
        saveStruct.(name) = optionalMetadata.(name);
    end
    save(fileName, '-struct', 'saveStruct', matVersion);
end

function previewSpectrum(inputFileName, trainData, testData, ...
        frequencyAxisHz, cfg)
    if strcmp(cfg.previewSplit, 'train')
        previewData = trainData;
    else
        previewData = testData;
    end
    assert(cfg.previewSampleIndex <= size(previewData, 1), ...
        'previewSampleIndex超出%s集样本数。', cfg.previewSplit);
    figure('Color', 'w');
    plot(frequencyAxisHz, ...
        abs(previewData(cfg.previewSampleIndex, :)), 'LineWidth', 1.1);
    grid on;
    xlabel('微多普勒频率（Hz）');
    ylabel(resolveSpectrumLabel(cfg.spectrumRepresentation));
    title(sprintf('%s：%s集第%d个样本', ...
        inputFileName, upper(cfg.previewSplit), cfg.previewSampleIndex), ...
        'Interpreter', 'none');
end

function label = resolveSpectrumLabel(representation)
    switch representation
        case 'complex'
            label = '频谱幅度';
        case 'magnitude'
            label = '频谱幅度';
        case 'power'
            label = '频谱功率';
        case 'db'
            label = '频谱（dB）';
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
