function outputs = generate_dynamic_rcs_hrrp_reusable(userConfig)
%% 根据静态角度回波和动态等效俯仰角生成窄带序列与宽带 HRRP
% 直接运行：
%   generate_dynamic_rcs_hrrp_reusable
%
% 使用外部配置覆盖默认参数：
%   userConfig.generateNarrowband = false;
%   userConfig.generateWideband = true;
%   userConfig.wideband.band = 'S';
%   generate_dynamic_rcs_hrrp_reusable(userConfig);

%% ======================== 集中配置区 ========================
projectRoot = fileparts(fileparts(mfilename('fullpath')));

% 生成开关，可分别或同时启用。
cfg.generateNarrowband = true;
cfg.generateWideband   = true;

% 目标配置。序号必须与 angle_target<序号>_<频段>.mat 对应。
cfg.targetNames   = {'dayuanzhui', 'zhuizhuti', ...
                     'qiudizhui', 'qiudizhuizhuti'};
cfg.targetIndices = 1:4;

% 动态角度数据。
cfg.angleDir        = fullfile(projectRoot, '等效俯仰角生成');
cfg.angleFileFormat = 'angle_target%d_%s.mat';
cfg.angleVariable   = 'angle_target';

% 静态数据的角度坐标。默认列1对应0°，相邻列间隔0.1°。
cfg.staticAngleStartDeg = 0;
cfg.staticAngleStepDeg  = 0.1;

% 慢时间参数，仅用于检查动态角度文件的列数。
cfg.prfHz                = 20;
cfg.observationTimeSec   = 5;
cfg.validateSampleCount  = true;

% 设为正整数时只处理前若干组运动参数，[] 表示处理全部样本。
cfg.maximumCases = [];

% 窄带输入及输出。
cfg.narrowband.band       = 'X';
cfg.narrowband.staticRoot = fullfile(projectRoot, '窄带数据');
cfg.narrowband.fileSuffixByBand.C = '.mat';
cfg.narrowband.fileSuffixByBand.S = '.mat';
cfg.narrowband.fileSuffixByBand.X = '.mat';
cfg.narrowband.variableByBand.C = 'data';
cfg.narrowband.variableByBand.S = 'data';
cfg.narrowband.variableByBand.X = 'data';
% 静态数据有多个频点时：'ifft_center_bin' 或 'first_frequency'。
cfg.narrowband.multiFrequencyMode = 'ifft_center_bin';
% [] 自动选择 fftshift 后的中心距离单元；也可指定正整数。
cfg.narrowband.rangeBinIndex = 51;
cfg.narrowband.outputNameFormat = 'rcs_%s.mat';
cfg.narrowband.outputVariable   = 'rcs_zhaidai_cell';

% 宽带输入及输出。不同频段的现有文件名和变量名并不完全一致。
cfg.wideband.band       = 'X';
cfg.wideband.staticRoot = fullfile(projectRoot, '宽带数据');
cfg.wideband.fileSuffixByBand.C = '_800MHz.mat';
cfg.wideband.fileSuffixByBand.S = '_600MHz.mat';
cfg.wideband.fileSuffixByBand.X = '.mat';
cfg.wideband.variableByBand.C = 'data';
cfg.wideband.variableByBand.S = 'data';
cfg.wideband.variableByBand.X = 'feko_data';
cfg.wideband.applyFftShift    = true;
cfg.wideband.outputNameFormat = 'hrrp_%s.mat';
cfg.wideband.outputVariable   = 'hrrp_kuandai_cell';

% 保存和显示。
cfg.saveResult    = true;
cfg.outputDir     = fullfile(projectRoot, '生成数据');
cfg.matFileVersion = '-v7.3';  % 宽带四目标数据可能超过2 GB
cfg.showProgress  = true;
cfg.showPreview   = true;
cfg.previewTarget = 4;
cfg.previewCase   = 1;
cfg.previewPower  = 0.2;

%% ======================== 合并外部配置 ========================
if nargin < 1 || isempty(userConfig)
    userConfig = struct();
end
assert(isstruct(userConfig), 'userConfig 必须是结构体。');
cfg = mergeStructRecursive(cfg, userConfig);

assert(cfg.generateNarrowband || cfg.generateWideband, ...
    'generateNarrowband 和 generateWideband 不能同时为 false。');
validateCommonConfiguration(cfg);

if ~isfolder(cfg.outputDir) && cfg.saveResult
    mkdir(cfg.outputDir);
end

outputs = struct();

%% ======================== 生成窄带动态序列 ========================
if cfg.generateNarrowband
    band = normalizeBand(cfg.narrowband.band);
    suffix = getBandSetting( ...
        cfg.narrowband.fileSuffixByBand, band, '窄带文件后缀');
    staticVariable = getBandSetting( ...
        cfg.narrowband.variableByBand, band, '窄带静态变量名');

    validateInputFiles(cfg, cfg.narrowband.staticRoot, band, suffix);
    rcsCell = cell(1, numel(cfg.targetNames));

    for targetIndex = cfg.targetIndices
        [staticData, dynamicAngles] = loadTargetInputs( ...
            cfg, cfg.narrowband.staticRoot, band, suffix, ...
            staticVariable, targetIndex);

        dynamicEcho = sampleStaticEcho(staticData, dynamicAngles, ...
            cfg.staticAngleStartDeg, cfg.staticAngleStepDeg, ...
            targetIndex, band, '窄带');

        if size(dynamicEcho, 1) == 1
            narrowSequence = permute(dynamicEcho(1, :, :), [3 2 1]);
        else
            switch lower(cfg.narrowband.multiFrequencyMode)
                case 'ifft_center_bin'
                    rangeProfiles = fftshift(ifft(dynamicEcho, [], 1), 1);
                    rangeBinIndex = cfg.narrowband.rangeBinIndex;
                    if isempty(rangeBinIndex)
                        rangeBinIndex = ceil((size(rangeProfiles, 1) + 1) / 2);
                    end
                    validateattributes(rangeBinIndex, {'numeric'}, ...
                        {'scalar', 'integer', '>=', 1, ...
                         '<=', size(rangeProfiles, 1)});
                    narrowSequence = permute( ...
                        rangeProfiles(rangeBinIndex, :, :), [3 2 1]);

                case 'first_frequency'
                    rangeBinIndex = cfg.narrowband.rangeBinIndex;
                    if isempty(rangeBinIndex)
                        rangeBinIndex = ceil((size(rangeProfiles, 1) + 1) / 2);
                    end
                    narrowSequence = permute(dynamicEcho(rangeBinIndex, :, :), [3 2 1]);

                otherwise
                    error(['narrowband.multiFrequencyMode 只能是 ' ...
                        '''ifft_center_bin'' 或 ''first_frequency''。']);
            end
        end

        rcsCell{targetIndex} = narrowSequence;
        showTargetProgress(cfg, '窄带', band, targetIndex, narrowSequence);
    end

    outputs.(cfg.narrowband.outputVariable) = rcsCell;
    if cfg.saveResult
        outputFile = fullfile(cfg.outputDir, sprintf( ...
            cfg.narrowband.outputNameFormat, band));
        saveNamedVariable(outputFile, ...
            cfg.narrowband.outputVariable, rcsCell, cfg.matFileVersion);
        fprintf('窄带结果已保存：%s\n', outputFile);
    end
end

%% ======================== 生成宽带 HRRP ========================
if cfg.generateWideband
    band = normalizeBand(cfg.wideband.band);
    suffix = getBandSetting( ...
        cfg.wideband.fileSuffixByBand, band, '宽带文件后缀');
    staticVariable = getBandSetting( ...
        cfg.wideband.variableByBand, band, '宽带静态变量名');

    validateInputFiles(cfg, cfg.wideband.staticRoot, band, suffix);
    hrrpCell = cell(1, numel(cfg.targetNames));

    for targetIndex = cfg.targetIndices
        [staticData, dynamicAngles] = loadTargetInputs( ...
            cfg, cfg.wideband.staticRoot, band, suffix, ...
            staticVariable, targetIndex);
        assert(size(staticData, 1) >= 2, ...
            '目标%d的宽带静态数据只有%d个频点，无法生成HRRP。', ...
            targetIndex, size(staticData, 1));

        dynamicEcho = sampleStaticEcho(staticData, dynamicAngles, ...
            cfg.staticAngleStartDeg, cfg.staticAngleStepDeg, ...
            targetIndex, band, '宽带');

        hrrpData = ifft(dynamicEcho, [], 1);
        if cfg.wideband.applyFftShift
            hrrpData = fftshift(hrrpData, 1);
        end

        hrrpCell{targetIndex} = hrrpData;
        showTargetProgress(cfg, '宽带HRRP', band, targetIndex, hrrpData);
    end

    outputs.(cfg.wideband.outputVariable) = hrrpCell;
    if cfg.saveResult
        outputFile = fullfile(cfg.outputDir, sprintf( ...
            cfg.wideband.outputNameFormat, band));
        saveNamedVariable(outputFile, ...
            cfg.wideband.outputVariable, hrrpCell, cfg.matFileVersion);
        fprintf('宽带结果已保存：%s\n', outputFile);
    end
end

%% ======================== 可选结果预览 ========================
if cfg.showPreview
    validateattributes(cfg.previewTarget, {'numeric'}, ...
        {'scalar', 'integer', '>=', 1, '<=', numel(cfg.targetNames)});
    validateattributes(cfg.previewCase, {'numeric'}, ...
        {'scalar', 'integer', 'positive'});
    validateattributes(cfg.previewPower, {'numeric'}, ...
        {'scalar', 'real', 'positive'});

    if cfg.generateNarrowband
        narrowData = outputs.(cfg.narrowband.outputVariable){cfg.previewTarget};
        assert(~isempty(narrowData), '预览目标没有生成窄带数据。');
        assert(cfg.previewCase <= size(narrowData, 1), ...
            'previewCase 超出窄带样本数。');
        figure('Color', 'w');
        plot(abs(narrowData(cfg.previewCase, :)), 'LineWidth', 1.2);
        grid on;
        xlabel('慢时间采样点');
        ylabel('回波幅度');
        title(sprintf('目标%d窄带动态回波（第%d组）', ...
            cfg.previewTarget, cfg.previewCase));
    end

    if cfg.generateWideband
        wideData = outputs.(cfg.wideband.outputVariable){cfg.previewTarget};
        assert(~isempty(wideData), '预览目标没有生成宽带数据。');
        assert(cfg.previewCase <= size(wideData, 3), ...
            'previewCase 超出宽带样本数。');
%         figure('Color', 'w');
        figure;
        imagesc(abs(wideData(:, :, cfg.previewCase)) .^ cfg.previewPower);
%         set(gca, 'YDir', 'normal');
%         colormap hot;l
%         colorbar;
        xlabel('慢时间采样点');
        ylabel('距离单元');
%         title(sprintf('目标%d HRRP（第%d组）', ...
%             cfg.previewTarget, cfg.previewCase));
    end
end

fprintf('动态回波数据生成完成。\n');
end

%% ======================== 本地函数 ========================
function validateCommonConfiguration(cfg)
    validateattributes(cfg.targetIndices, {'numeric'}, ...
        {'vector', 'integer', 'positive', 'nonempty'});
    assert(all(cfg.targetIndices <= numel(cfg.targetNames)), ...
        'targetIndices 超出 targetNames 的有效范围。');
    assert(numel(unique(cfg.targetIndices)) == numel(cfg.targetIndices), ...
        'targetIndices 不能包含重复序号。');
    validateattributes(cfg.staticAngleStartDeg, {'numeric'}, ...
        {'scalar', 'real', 'finite'});
    validateattributes(cfg.staticAngleStepDeg, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'});
    validateattributes(cfg.prfHz, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'});
    validateattributes(cfg.observationTimeSec, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'});
    if ~isempty(cfg.maximumCases)
        validateattributes(cfg.maximumCases, {'numeric'}, ...
            {'scalar', 'integer', 'positive'});
    end
    assert(any(strcmp(cfg.matFileVersion, {'-v7', '-v7.3'})), ...
        'matFileVersion 只能是 ''-v7'' 或 ''-v7.3''。');
end

function band = normalizeBand(configuredBand)
    band = upper(strtrim(char(configuredBand)));
    assert(numel(band) == 1, '频段必须是单个字符，例如 ''C''。');
end

function value = getBandSetting(settings, band, description)
    assert(isfield(settings, band), ...
        '%s中没有频段 %s 的配置。', description, band);
    value = settings.(band);
end

function validateInputFiles(cfg, staticRoot, band, suffix)
    missingFiles = {};
    for targetIndex = cfg.targetIndices
        staticFile = fullfile(staticRoot, band, ...
            [cfg.targetNames{targetIndex}, suffix]);
        angleFile = fullfile(cfg.angleDir, sprintf( ...
            cfg.angleFileFormat, targetIndex, band));
        if ~isfile(staticFile)
            missingFiles{end + 1} = staticFile; %#ok<AGROW>
        end
        if ~isfile(angleFile)
            missingFiles{end + 1} = angleFile; %#ok<AGROW>
        end
    end
    assert(isempty(missingFiles), '缺少以下输入文件：\n%s', ...
        strjoin(missingFiles, newline));
end

function [staticData, dynamicAngles] = loadTargetInputs( ...
        cfg, staticRoot, band, suffix, staticVariable, targetIndex)
    staticFile = fullfile(staticRoot, band, ...
        [cfg.targetNames{targetIndex}, suffix]);
    angleFile = fullfile(cfg.angleDir, sprintf( ...
        cfg.angleFileFormat, targetIndex, band));

    staticData = loadRequiredVariable( ...
        staticFile, staticVariable, '静态回波');
    dynamicAngles = loadRequiredVariable( ...
        angleFile, cfg.angleVariable, '动态角度');

    assert(ismatrix(staticData) && ~isempty(staticData), ...
        '静态回波必须是“频点×角度”的非空二维矩阵：%s', staticFile);
    assert(ismatrix(dynamicAngles) && ~isempty(dynamicAngles), ...
        '动态角度必须是“样本×慢时间”的非空二维矩阵：%s', angleFile);
    assert(isreal(dynamicAngles) && all(isfinite(dynamicAngles(:))), ...
        '动态角度中包含复数、NaN或Inf：%s', angleFile);

    if cfg.validateSampleCount
        expectedSamples = numel(0:1/cfg.prfHz:cfg.observationTimeSec);
        assert(size(dynamicAngles, 2) == expectedSamples, ...
            ['动态角度文件有%d个慢时间点，配置要求%d个。' ...
             '请检查PRF和观测时间：%s'], ...
            size(dynamicAngles, 2), expectedSamples, angleFile);
    end

    if ~isempty(cfg.maximumCases)
        dynamicAngles = dynamicAngles( ...
            1:min(cfg.maximumCases, size(dynamicAngles, 1)), :);
    end
end

function value = loadRequiredVariable(fileName, variableName, description)
    fileVariables = whos('-file', fileName);
    availableNames = {fileVariables.name};
    assert(ismember(variableName, availableNames), ...
        '%s文件 %s 中不存在变量 %s；现有变量：%s', ...
        description, fileName, variableName, strjoin(availableNames, ', '));
    loaded = load(fileName, variableName);
    value = loaded.(variableName);
    assert(isnumeric(value), '%s变量必须是数值数组：%s', description, fileName);
end

function dynamicEcho = sampleStaticEcho(staticData, dynamicAngles, ...
        angleStartDeg, angleStepDeg, targetIndex, band, dataType)
    angleIndices = round((dynamicAngles - angleStartDeg) ./ angleStepDeg) + 1;
    invalidMask = angleIndices < 1 | angleIndices > size(staticData, 2);
    if any(invalidMask(:))
        invalidAngles = dynamicAngles(invalidMask);
        availableEndDeg = angleStartDeg + ...
            (size(staticData, 2) - 1) * angleStepDeg;
        error(['目标%d的%s%s数据角度范围为%.1f°~%.1f°，' ...
               '但动态角度中存在%.3f°~%.3f°的越界值。'], ...
            targetIndex, band, dataType, angleStartDeg, availableEndDeg, ...
            min(invalidAngles), max(invalidAngles));
    end

    % 转置后按列展开，使慢时间点在每组运动参数内部连续排列。
    flatAngleIndices = reshape(angleIndices.', 1, []);
    sampledColumns = staticData(:, flatAngleIndices);
    dynamicEcho = reshape(sampledColumns, size(staticData, 1), ...
        size(dynamicAngles, 2), size(dynamicAngles, 1));
end

function saveNamedVariable(fileName, variableName, value, matVersion)
    saveStruct = struct();
    saveStruct.(variableName) = value;
    save(fileName, '-struct', 'saveStruct', matVersion);
end

function showTargetProgress(cfg, dataType, band, targetIndex, value)
    if cfg.showProgress
        fprintf('%s %s：目标%d（%s），输出尺寸 %s。\n', ...
            band, dataType, targetIndex, cfg.targetNames{targetIndex}, ...
            mat2str(size(value)));
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
