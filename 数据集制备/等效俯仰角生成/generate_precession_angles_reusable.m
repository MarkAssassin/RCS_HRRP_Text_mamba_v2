%% 批量生成进动目标的等效俯仰角变化曲线
% 依赖：chouqujiaodu_jindong_600.m 及其相关函数。
% 输出 angle_target 的每一行对应一组参数，每一列对应一个慢时间采样点。

clear;
clc;
close all;

%% ======================== 集中配置区 ========================
% 频段：'P'、'C'、'L'、'S' 或 'X'，不区分大小写。
cfg.frequencyBand = 'X';

% 不同频段对应的初始俯仰角（°）。
cfg.initialPitchAnglesByBand.P = 31:1:43;
cfg.initialPitchAnglesByBand.C = 31:1:43;
cfg.initialPitchAnglesByBand.L = 31:1:43;
cfg.initialPitchAnglesByBand.S = 21:1:33;
cfg.initialPitchAnglesByBand.X = 41:1:53;

% 运动与采样参数。
cfg.prfHz                   = 20;          % 脉冲重复频率（Hz）
cfg.observationTimeSec      = 5;           % 观测时间（s）
cfg.spinFrequencyHz         = 3;           % 自旋频率（Hz）
cfg.precessionFrequenciesHz = 2:0.2:4;     % 进动频率（Hz）
cfg.precessionAnglesDeg     = 2:0.15:3.5;  % 进动角（°）

% 角度生成函数及其所在目录。
cfg.generatorFunction = @chouqujiaodu_jindong_600;
cfg.functionDir = fileparts(mfilename('fullpath'));

% 保存设置。
cfg.saveResult       = true;
cfg.outputDir        = fileparts(mfilename('fullpath'));
cfg.outputNameFormat = 'angle_target4_%s.mat';
cfg.outputVariable   = 'angle_target';
cfg.saveMetadata     = false;  % true 时额外保存参数表、时间轴和 cfg

% 显示设置。
cfg.showProgress = true;
cfg.showPreview  = true;
cfg.previewRow   = 1;

%% ======================== 参数准备与检查 ========================
band = upper(strtrim(char(cfg.frequencyBand)));
assert(isfield(cfg.initialPitchAnglesByBand, band), ...
    '不支持频段 %s；可选频段为 P、C、L、S、X。', band);

validateattributes(cfg.prfHz, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'});
validateattributes(cfg.observationTimeSec, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'});
validateattributes(cfg.spinFrequencyHz, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'nonnegative'});
validateattributes(cfg.precessionFrequenciesHz, {'numeric'}, ...
    {'vector', 'real', 'finite', 'positive', 'nonempty'});
validateattributes(cfg.precessionAnglesDeg, {'numeric'}, ...
    {'vector', 'real', 'finite', 'nonnegative', 'nonempty'});

initialPitchAnglesDeg = cfg.initialPitchAnglesByBand.(band);
validateattributes(initialPitchAnglesDeg, {'numeric'}, ...
    {'vector', 'real', 'finite', 'nonempty'});

% 保证脚本从任意当前目录启动时都能找到同目录下的依赖函数。
if ~contains(path, cfg.functionDir)
    addpath(cfg.functionDir);
end

timeSec = 0:1/cfg.prfHz:cfg.observationTimeSec;
sampleCount = numel(timeSec);
caseCount = numel(initialPitchAnglesDeg) * ...
    numel(cfg.precessionFrequenciesHz) * numel(cfg.precessionAnglesDeg);

% 预分配，避免在循环中逐行扩展数组。
angleTarget = zeros(caseCount, sampleCount);
% 参数列依次为：初始俯仰角、进动频率、进动角。
caseParameters = zeros(caseCount, 3);

%% ======================== 批量生成角度曲线 ========================
caseIndex = 0;
for pitchIndex = 1:numel(initialPitchAnglesDeg)
    initialPitchDeg = initialPitchAnglesDeg(pitchIndex);

    if cfg.showProgress
        fprintf('正在处理频段 %s：初始俯仰角 %.2f°（%d/%d）\n', ...
            band, initialPitchDeg, pitchIndex, numel(initialPitchAnglesDeg));
    end

    for frequencyIndex = 1:numel(cfg.precessionFrequenciesHz)
        precessionFrequencyHz = ...
            cfg.precessionFrequenciesHz(frequencyIndex);

        for angleIndex = 1:numel(cfg.precessionAnglesDeg)
            precessionAngleDeg = cfg.precessionAnglesDeg(angleIndex);
            caseIndex = caseIndex + 1;

            curve = cfg.generatorFunction( ...
                precessionAngleDeg, ...
                cfg.spinFrequencyHz, ...
                precessionFrequencyHz, ...
                cfg.observationTimeSec, ...
                initialPitchDeg, ...
                cfg.prfHz);

            validateattributes(curve, {'numeric'}, ...
                {'vector', 'real', 'finite', 'numel', sampleCount}, ...
                func2str(cfg.generatorFunction), 'curve');

            angleTarget(caseIndex, :) = curve(:).';
            caseParameters(caseIndex, :) = [ ...
                initialPitchDeg, precessionFrequencyHz, precessionAngleDeg];
        end
    end
end

% 与原程序保持兼容的工作区变量名。
angle_target = angleTarget;

%% ======================== 保存结果 ========================
outputName = sprintf(cfg.outputNameFormat, band);
outputFile = fullfile(cfg.outputDir, outputName);

if cfg.saveResult
    if ~isfolder(cfg.outputDir)
        mkdir(cfg.outputDir);
    end

    saveStruct = struct();
    saveStruct.(cfg.outputVariable) = angleTarget;

    if cfg.saveMetadata
        saveStruct.case_parameters = caseParameters;
        saveStruct.parameter_columns = { ...
            'initial_pitch_deg', ...
            'precession_frequency_hz', ...
            'precession_angle_deg'};
        saveStruct.time_sec = timeSec;
        saveStruct.cfg = cfg;
    end

    save(outputFile, '-struct', 'saveStruct');
end

%% ======================== 可选预览 ========================
if cfg.showPreview
    validateattributes(cfg.previewRow, {'numeric'}, ...
        {'scalar', 'integer', '>=', 1, '<=', caseCount});

    figure('Color', 'w');
    plot(timeSec, angleTarget(cfg.previewRow, :), 'LineWidth', 1.2);
    grid on;
    xlabel('时间（s）', 'FontSize', 13);
    ylabel('等效俯仰角（°）', 'FontSize', 13);
    title(sprintf('第 %d 组进动俯仰角曲线', cfg.previewRow), ...
        'FontSize', 13);
end

fprintf('生成完成：%d 组参数，每组 %d 个慢时间采样点。\n', ...
    caseCount, sampleCount);
if cfg.saveResult
    fprintf('结果已保存至：%s\n', outputFile);
end
