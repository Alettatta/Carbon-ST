% 清空环境
clear;
clc;

% 加载 CSV 数据
data = readtable('2025-1-2.csv'); % 替换为实际文件名

% 提取省份名称和数据
province_names = data.Properties.VariableNames; % 第一行为省份名称
values = table2array(data); % 提取数值部分

% 定义参数
[num_days, num_provinces] = size(values); % 行数为天数，列数为省份数
fft_results = cell(num_provinces, 1); % 用于存储各省份的FFT结果

% 设置省份颜色
province_colors = lines(num_provinces); % 使用 `lines` 函数生成不同颜色

% 自定义颜色映射（新疆和辽宁颜色相同，湖南和湖北颜色相同，山东和江苏颜色相同）
color_map = containers.Map;
color_map('Xinjiang') = province_colors(1, :);  % 新疆
color_map('Liaoning') = province_colors(1, :); % 辽宁
color_map('Hunan') = province_colors(2, :); % 湖南
color_map('Hubei') = province_colors(2, :); % 湖北
color_map('Shandong') = province_colors(3, :); % 山东
color_map('Jiangsu') = province_colors(3, :); % 江苏
% 其他省份使用lines中不同的颜色

% 采样频率 (假设每天采样一次，可调整为实际情况)
Fs = 1; 

% FFT 分析
for i = 1:num_provinces
    signal = values(:, i); % 提取当前省份的时间序列
    L = length(signal); % 信号长度
    
    % 傅里叶变换
    Y = fft(signal);
    P2 = abs(Y / L); % 双边谱
    P1 = P2(1:L/2+1); % 单边谱
    P1(2:end-1) = 2*P1(2:end-1); % 单边幅值
    f = Fs * (0:(L/2)) / L; % 频率坐标
    
    fft_results{i} = struct('frequency', f, 'amplitude', P1);
    
    % 获取省份名称
    province = province_names{i};
    
    % 根据省份名称选择颜色
    if isKey(color_map, province)
        color = color_map(province);
    else
        color = province_colors(i, :); % 默认颜色
    end
    
    % 绘图
    figure;
    sgtitle(['FFT Analysis for ' province], 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    
    % 原始信号
    subplot(2, 1, 1);
    plot(signal, 'LineWidth', 1.5, 'Color', color);
    ylabel('Original Signal', 'FontSize', 12, 'FontName', 'Times New Roman');
    set(gca, 'FontSize', 10);
    
    % 频域信号
    subplot(2, 1, 2);
    plot(f, P1, 'LineWidth', 1.5, 'Color', color);
    ylabel('Amplitude Spectrum', 'FontSize', 12, 'FontName', 'Times New Roman');
    xlabel('Frequency (Hz)', 'FontSize', 12, 'FontName', 'Times New Roman');
    set(gca, 'FontSize', 10);
end

% 保存结果
save('fft_results.mat', 'fft_results', 'province_names');
