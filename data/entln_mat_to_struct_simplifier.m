cases = {
    'Case1_Nov_2022_23_25',
    'Case2_Jan_2023_11_16',
    'Case3_Mar_2023_13_15',
    'Case4_Apr_2023_09_13',
    'Case5_Jan_2024_26_31'
};

base = '/home/ubuntu/Desktop/local_raw_data/ENTLN/';

for i = 1:numel(cases)
    fname = fullfile(base, sprintf('ENTLN_pulse_%s.mat', cases{i}));
    S = load(fname);
    f = fieldnames(S);
    T = S.(f{1});   % the table

    % Convert to simple struct Python can read
    entln.lat = T.lat;
    entln.lon = T.lon;
    entln.UTC = datestr(T.UTC, 'dd-mmm-yyyy HH:MM:SS');

    outname = fullfile(base, sprintf('ENTLN_pulse_%s_struct.mat', cases{i}));
    save(outname, 'entln', '-v7');
    fprintf('Saved: %s\n', outname);
end
