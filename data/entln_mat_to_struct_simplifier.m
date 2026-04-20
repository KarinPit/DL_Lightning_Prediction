% entln_mat_to_struct_simplifier.m
% Converts ENTLN .mat files (MATLAB table format) to simple structs
% that Python/scipy can read. Run this in MATLAB on your local machine,
% then upload the _struct.mat files to the server.
%
% Output files are saved alongside the originals as:
%   ENTLN_pulse_<CaseName>_struct.mat

cases = {
    'Case1_Nov_2022_23_25',
    'Case2_Jan_2023_11_16',
    'Case3_Mar_2023_13_15',
    'Case4_Apr_2023_09_13',
    'Case5_Jan_2024_26_31'
};

% ── Update this path to where your .mat files are ────────────────────────────
base = '/path/to/Lightning_Data/ENTLN/Cases/';

for i = 1:numel(cases)
    fname = fullfile(base, sprintf('ENTLN_pulse_%s.mat', cases{i}));

    if ~isfile(fname)
        fprintf('Skipping (not found): %s\n', fname);
        continue
    end

    fprintf('Loading: %s\n', fname);
    S = load(fname);
    f = fieldnames(S);
    T = S.(f{1});   % the table

    % Show column names so we can verify
    fprintf('  Columns: %s\n', strjoin(T.Properties.VariableNames, ', '));

    % Build struct — try both lat/lon and closest_lats/closest_longs
    entln = struct();
    if ismember('lat', T.Properties.VariableNames)
        entln.lat = T.lat;
        entln.lon = T.lon;
    elseif ismember('closest_lats', T.Properties.VariableNames)
        entln.lat = T.closest_lats;
        entln.lon = T.closest_longs;
    else
        fprintf('  WARNING: no lat/lon column found for %s\n', cases{i});
        continue
    end

    % UTC — convert to string format Python expects: 'dd-mmm-yyyy HH:MM:SS'
    if ismember('UTC', T.Properties.VariableNames)
        entln.UTC = datestr(T.UTC, 'dd-mmm-yyyy HH:MM:SS');
    else
        % find any datetime column
        for c = T.Properties.VariableNames
            if isdatetime(T.(c{1}))
                entln.UTC = datestr(T.(c{1}), 'dd-mmm-yyyy HH:MM:SS');
                fprintf('  Using datetime column: %s\n', c{1});
                break
            end
        end
    end

    outname = fullfile(base, sprintf('ENTLN_pulse_%s_struct.mat', cases{i}));
    save(outname, 'entln', '-v7');
    fprintf('  Saved: %s\n', outname);
end

fprintf('\nDone! Upload the _struct.mat files to the server:\n');
fprintf('  /home/ec2-user/thesis-bucket/Lightning_Data/ENTLN/Cases/\n');
