% Corrected companion to the supplied FFT.m. No Signal Processing Toolbox needed.
% The delivered TIFFs are generated and verified by generate_fft.py.
clear; clc;
task_root = fileparts(fileparts(mfilename('fullpath')));
data = readtable(fullfile(task_root,'inputs','source_series.csv'), ...
    'VariableNamingRule','preserve');
province_names = data.Properties.VariableNames;
province_names = province_names(~ismember(province_names, {'Day','total'}));
assert(numel(province_names)==31, 'Expected 31 province columns');
N = 180; Fs = 1; % Samples per day, hence frequency unit cycles/day.
values = table2array(data(end-N+1:end,province_names));
assert(all(isfinite(values(:))), 'Non-finite input');
output = fullfile(task_root,'results','matlab_reference');
if ~exist(output,'dir'), mkdir(output); end
t = (0:N-1)'; tc = t-mean(t);
w = 0.5-0.5*cos(2*pi*t/N); % Periodic Hann, same as np.hanning(N+1)[:-1].
f = Fs*(0:floor(N/2))'/N;
fft_results = cell(31,1);
metrics = zeros(31,6);
for i=1:31
    signal = values(:,i);
    centered = signal-mean(signal); % Demean.
    slope = (tc'*centered)/(tc'*tc);
    residual = centered-slope*tc; % Least-squares linear detrend.
    windowed = residual.*w;
    Y = fft(windowed);
    P = abs(Y(1:floor(N/2)+1)).^2/(Fs*sum(w.^2));
    if mod(N,2)==0, P(2:end-1)=2*P(2:end-1);
    else, P(2:end)=2*P(2:end); end
    positive_f = f(2:end); positive_P = P(2:end); % Exclude DC.
    assert(sum(positive_P)>eps^2*N*max(max(abs(signal)),1)^2*100, ...
        'No resolvable residual variation; dominant period undefined');
    normalized_power = positive_P/sum(positive_P);
    [~,peak] = max(normalized_power);
    dominant_period = 1/positive_f(peak);
    long_mask = positive_f<=1/30;
    medium_mask = positive_f>1/30 & positive_f<=1/7;
    short_mask = positive_f>1/7;
    band_ratios = [sum(normalized_power(long_mask)), ...
        sum(normalized_power(medium_mask)),sum(normalized_power(short_mask))];
    assert(abs(sum(band_ratios)-1)<1e-12);
    fft_results{i} = struct('province',province_names{i}, ...
        'frequency_cycles_per_day',positive_f,'normalized_power',normalized_power, ...
        'dominant_period_days',dominant_period,'band_energy_ratios',band_ratios, ...
        'demeaned',centered,'detrended',residual,'hann_window',w,'windowed',windowed);
    metrics(i,:) = [positive_f(peak),dominant_period,normalized_power(peak)*100,band_ratios*100];
    fig=figure('Visible','off','Color','white','Position',[100 100 1000 760]);
    subplot(2,1,1); plot(1:N,signal,'LineWidth',1.2);
    title(province_names{i},'Interpreter','none');
    xlabel('Forecast day'); ylabel('Forecast (source units)');
    subplot(2,1,2); semilogy(positive_f,normalized_power*100,'LineWidth',1.2);
    xlabel('Frequency (cycles/day)'); ylabel('Normalized power (%)');
    title(sprintf('Dominant: %.2f d | L/M/H: %.1f/%.1f/%.1f%%',dominant_period,band_ratios*100));
    xlim([0 0.5]); grid on;
    exportgraphics(fig,fullfile(output,[strrep(province_names{i},' ','_') '.tif']),'Resolution',300);
    close(fig);
end
summary=array2table(metrics,'VariableNames',{'dominant_frequency_cycles_per_day', ...
    'dominant_period_days','peak_power_percent','long_period_power_percent', ...
    'medium_period_power_percent','short_period_power_percent'});
summary=addvars(summary,province_names','Before',1,'NewVariableNames','province');
writetable(summary,fullfile(output,'spectral_metrics_31.csv'));
save(fullfile(output,'fft_results.mat'),'fft_results','province_names','Fs','N');
