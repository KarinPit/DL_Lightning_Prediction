%% ===========================
%  Lightning Pipeline (ALL-IN-ONE)
%  - Build coarse grid
%  - Process MODEL (LPI/KI), ILDN, ENTLN into interval NetCDFs
%  - Keep ensemble members separated in MODEL outputs
%  - Run over all configured variables
%  - Loops over all cases automatically
%  Karin Pitlik
%  ===========================

%% -------- CASES TO PROCESS --------
cases = {
    'Case1_Nov_2022_23_25',
    'Case2_Jan_2023_11_16',
    'Case3_Mar_2023_13_15',
    'Case4_Apr_2023_09_13',
    'Case5_Jan_2024_26_31',
    'Case6_Nov_2025_24_25'
};

for case_idx = 1:numel(cases)

%% -------- USER CONFIG --------
cfg = struct();

cfg.case_name = cases{case_idx};
fprintf('\n\n========================================\n');
fprintf('PROCESSING CASE %d/%d: %s\n', case_idx, numel(cases), cfg.case_name);
fprintf('========================================\n');

% cfg.variable_types = {'lpi', 'KI', 'ds', 'wdiag', 'flux_prod', 'prec_rate', 'cape2d'};
cfg.variable_types = {'lpi', 'KI', 'ds', 'prec_rate', 'cape2d'};
cfg.sub_variables = containers.Map;
cfg.sub_variables('wdiag') = {'wmax_layer', 'mflux_mean_layer', 'wplus_mean_layer'};

% Auto-parse dates from case name
[cfg.loop_start_date, cfg.loop_end_date] = parse_case_dates(cfg.case_name);
fprintf('Date range: %s --> %s\n', datestr(cfg.loop_start_date), datestr(cfg.loop_end_date));

cfg.interval_hours  = 1;
cfg.resolution_km   = 4;   % supported: 4, 12, 24, 40, 80
cfg.relevant_lat    = 32;

% Geographic filter
cfg.min_lat = 27.296;
cfg.max_lat = 36.598;
cfg.min_lon = 27.954;
cfg.max_lon = 39.292;

% Plot / output settings
cfg.ens_to_plot = '00';

%% -------- Global derived config --------
cfg.interval_name = sprintf('%d_hours', cfg.interval_hours);
cfg.time_step     = hours(cfg.interval_hours);

cfg.bin_width_lat = cfg.resolution_km / 111.32;
cfg.bin_width_lon = cfg.resolution_km / (111.32 * cosd(cfg.relevant_lat));
cfg.grid_label    = grid_lbl(cfg.resolution_km);

%% -------- Paths --------
cfg.main_path = '/home/ubuntu/Desktop/';

cfg.coords_folder_path = sprintf('%s/local_raw_data', cfg.main_path);
cfg.coords_file_path   = fullfile(cfg.coords_folder_path, 'lpi_4km_output_2022-01-24_00_10_00.nc');

cfg.entln_mat_path = sprintf('%s/local_raw_data/ENTLN/ENTLN_pulse_%s.mat', cfg.main_path, cfg.case_name);

cfg.entln_out_root = sprintf('%s/local_processed_data/%s/ENTLN/%s/%s/', ...
    cfg.main_path, cfg.case_name, cfg.grid_label, cfg.interval_name);

mkif(cfg.entln_out_root);

%% -------- Prepare coarse grid from reference coordinates --------
lat_grid = ncread(cfg.coords_file_path, 'xlat');
lon_grid = ncread(cfg.coords_file_path, 'xlong');
[m,n]    = size(lat_grid);

[cfg.downgrade_factor, ~] = factor_by_res(cfg.resolution_km);

if cfg.downgrade_factor == 1
    lat_grid_coarse = lat_grid;
    lon_grid_coarse = lon_grid;
else
    m_trim = floor(m / cfg.downgrade_factor) * cfg.downgrade_factor;
    n_trim = floor(n / cfg.downgrade_factor) * cfg.downgrade_factor;

    lat_trim = lat_grid(1:m_trim, 1:n_trim);
    lon_trim = lon_grid(1:m_trim, 1:n_trim);

    lat_blocks = reshape(lat_trim, ...
        cfg.downgrade_factor, m_trim/cfg.downgrade_factor, ...
        cfg.downgrade_factor, n_trim/cfg.downgrade_factor);

    lon_blocks = reshape(lon_trim, ...
        cfg.downgrade_factor, m_trim/cfg.downgrade_factor, ...
        cfg.downgrade_factor, n_trim/cfg.downgrade_factor);

    lat_grid_coarse = squeeze(mean(mean(lat_blocks,1),3));
    lon_grid_coarse = squeeze(mean(mean(lon_blocks,1),3));
end

%% -------- Build time edges --------
time_edges = datetime(cfg.loop_start_date:cfg.time_step:cfg.loop_end_date, ...
    'Format', 'yyyy-MM-dd HH:mm:ss');

%% -------- Pass 1: WRITE interval NetCDFs for all variables --------
for v = 1:numel(cfg.variable_types)
    variable_type = cfg.variable_types{v};

    if isKey(cfg.sub_variables, lower(variable_type))
        subvars = cfg.sub_variables(lower(variable_type));
    else
        subvars = {variable_type};
    end

    for sv = 1:numel(subvars)
        current_var = subvars{sv};
        var_cfg = prepare_variable_config(cfg, variable_type, current_var);

        fprintf('\n========================================\n');
        fprintf('Processing variable group: %s | sub-variable: %s\n', variable_type, current_var);
        fprintf('========================================\n');

        mkif(var_cfg.model_out_root);
        mkif(var_cfg.fig_output_dir);

        for t = 1:(length(time_edges)-1)
            sdate = time_edges(t);
            edate = time_edges(t+1);

            process_model_files(var_cfg.read_var_name, var_cfg.model_raw_folder, var_cfg.model_out_root, ...
                var_cfg.coords_file_path, sdate, edate, var_cfg.downgrade_factor, ...
                lat_grid_coarse, lon_grid_coarse, var_cfg.spatial_op);

            process_entln_pulse(var_cfg.entln_mat_path, var_cfg.entln_out_root, ...
                sdate, edate, var_cfg.min_lat, var_cfg.max_lat, var_cfg.min_lon, var_cfg.max_lon, ...
                var_cfg.bin_width_lat, var_cfg.bin_width_lon, lat_grid_coarse, lon_grid_coarse);
        end
    end
end

fprintf('\nFinished case: %s\n', cfg.case_name);

end % end case loop
fprintf('\n\nAll cases processed!\n');

%% ===========================
%           FUNCTIONS
% ===========================

function var_cfg = prepare_variable_config(cfg, variable_type, current_var)
    var_cfg = cfg;

    % variable_type = folder name
    % current_var   = actual variable to read/write
    var_cfg.variable_type = variable_type;
    var_cfg.variable_type_upper = upper(variable_type);
    var_cfg.variable_type_lower = lower(variable_type);

    var_cfg.read_var_name = current_var;
    var_cfg.read_var_name_lower = lower(current_var);

    % Special handling for exact NetCDF variable names
    if strcmpi(current_var, 'KI')
        var_cfg.read_var_name = 'KI';
    elseif strcmpi(current_var, 'ds')
        var_cfg.read_var_name = 'ds';
    end

    switch lower(current_var)
        case 'lpi'
            var_cfg.spatial_op  = 'mean';
            var_cfg.temporal_op = 'mean';
        case 'ki'
            var_cfg.spatial_op  = 'mean';
            var_cfg.temporal_op = 'mean';
        case 'ds'
            var_cfg.spatial_op  = 'mean';
            var_cfg.temporal_op = 'sum';
        case 'wmax_layer'
            var_cfg.spatial_op  = 'mean';
            var_cfg.temporal_op = 'mean';
        case 'mflux_mean_layer'
            var_cfg.spatial_op  = 'mean';
            var_cfg.temporal_op = 'mean';
        case 'wplus_mean_layer'
            var_cfg.spatial_op  = 'mean';
            var_cfg.temporal_op = 'mean';
        case 'flux_prod'
            var_cfg.spatial_op  = 'mean';
            var_cfg.temporal_op = 'mean';
        case 'prec_rate'
            var_cfg.spatial_op  = 'mean';
            var_cfg.temporal_op = 'sum';
        case 'cape2d'
            var_cfg.spatial_op  = 'mean';
            var_cfg.temporal_op = 'mean';
        otherwise
            error('Unsupported variable: %s', current_var);
    end

    % Folder still comes from variable_type
    var_cfg.model_raw_folder = sprintf('%s/local_raw_data/%s/Ens/Raw/%s/', ...
        var_cfg.main_path, var_cfg.case_name, var_cfg.variable_type_upper);

    % Save outputs under the actual variable name
    var_cfg.model_out_root = sprintf('%s/local_processed_data/%s/%s/%s/%s/', ...
        var_cfg.main_path, var_cfg.case_name, var_cfg.variable_type_upper, var_cfg.grid_label, var_cfg.interval_name);

    disp(var_cfg.model_out_root);

    var_cfg.fig_output_dir = sprintf('%s/local_processed_data/Graphs/%s/%s/%s/%s/', ...
        var_cfg.main_path, var_cfg.case_name, upper(current_var), ...
        var_cfg.grid_label, var_cfg.interval_name);
end

function process_model_files(varname, variable_folder_path, output_path, coords_file_path, ...
                             start_date, end_date, downgrade_factor, ...
                             lat_grid_coarse, lon_grid_coarse, spatial_op)

    mkif(output_path);

    file_list = dir(fullfile(variable_folder_path, '*.nc'));

    % Store data separately for each ensemble member
    ens_map = containers.Map('KeyType','char','ValueType','any');

    for i = 1:numel(file_list)
        fname = file_list(i).name;

        % Extract timestamp from filename
        date_match = regexp(fname, '\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}', 'match');
        if isempty(date_match)
            continue;
        end
        fdate = datetime(date_match{1}, 'InputFormat','yyyy-MM-dd_HH:mm:ss');

        % Extract ensemble id from filename
        ens_match = regexp(fname, '_(\d{1,2})_', 'tokens');

        if isempty(ens_match)
            if strcmpi(varname, 'wdiag')
                ens_id = '00';
            else
                ens_id = '00';
                warning('Could not extract ensemble id from file: %s. Using default ens_id = %s', fname, ens_id);
            end
        else
            ens_id = sprintf('%02d', str2double(ens_match{1}{1}));
        end

        % Keep only files within the current interval
        if fdate >= start_date && fdate < end_date
            fpath = fullfile(variable_folder_path, fname);

            try
                if strcmpi(varname,'ds')
                    lneg = ncread(fpath, 'lneg');
                    lpos = ncread(fpath, 'lpos');
                    lneu = ncread(fpath, 'lneu');
                    cube = lneg + lpos + lneu;
                else
                    cube = ncread(fpath, varname);
                end

                % Spatial downscaling
                if downgrade_factor ~= 1
                    [dm, dn, nt] = size(cube);
                    dm_trim = floor(dm/downgrade_factor) * downgrade_factor;
                    dn_trim = floor(dn/downgrade_factor) * downgrade_factor;
                    cube    = cube(1:dm_trim, 1:dn_trim, :);

                    tmp = reshape(cube, ...
                        downgrade_factor, dm_trim/downgrade_factor, ...
                        downgrade_factor, dn_trim/downgrade_factor, ...
                        nt);

                    cube = spatial_block_agg(tmp, spatial_op);
                end

                % Initialize ensemble entry if needed
                if ~isKey(ens_map, ens_id)
                    ens_struct.combined_data = [];
                    ens_struct.file_dates = datetime.empty(0,1);
                    ens_map(ens_id) = ens_struct;
                end

                ens_struct = ens_map(ens_id);

                if isempty(ens_struct.combined_data)
                    ens_struct.combined_data = cube;
                else
                    ens_struct.combined_data = cat(3, ens_struct.combined_data, cube);
                end

                ens_struct.file_dates(end+1,1) = fdate;
                ens_map(ens_id) = ens_struct;

            catch ME
                fprintf('Error reading %s: %s\n', fname, ME.message);
            end
        end
    end

    if isempty(keys(ens_map))
        fprintf('No MODEL files in %s–%s. Skipping write.\n', ...
            datestr(start_date), datestr(end_date));
        return;
    end

    % Write one file per ensemble member
    ens_keys = keys(ens_map);

    for k = 1:numel(ens_keys)
        ens_id = ens_keys{k};
        ens_struct = ens_map(ens_id);

        combined_data = ens_struct.combined_data;
        file_dates = ens_struct.file_dates;

        if isempty(combined_data)
            continue;
        end

        % Sort by time
        [file_dates, idx] = sort(file_dates);
        combined_data = combined_data(:,:,idx);

        [num_lat, num_lon, num_time] = size(combined_data);

        outfile = fullfile(output_path, sprintf('%s_%s_%s.nc', ...
            ens_id, ...
            datestr(start_date,'yyyy-mm-dd_HH_MM_SS'), ...
            datestr(end_date,'yyyy-mm-dd_HH_MM_SS')));

        if exist(outfile,'file')
            delete(outfile);
        end

        nccreate(outfile, varname, ...
            'Dimensions', {'lat',num_lat,'lon',num_lon,'time',num_time}, ...
            'Datatype','single');

        nccreate(outfile, 'xlat', ...
            'Dimensions', {'lat',num_lat,'lon',num_lon}, ...
            'Datatype','single');

        nccreate(outfile, 'xlong', ...
            'Dimensions', {'lat',num_lat,'lon',num_lon}, ...
            'Datatype','single');

        ncwrite(outfile, varname, combined_data);
        ncwrite(outfile, 'xlat', lat_grid_coarse);
        ncwrite(outfile, 'xlong', lon_grid_coarse);

        if ~isempty(file_dates)
            tref = datetime(1900,1,1);
            tsecs = seconds(file_dates - tref);

            nccreate(outfile, 'time', ...
                'Dimensions', {'time', num_time}, ...
                'Datatype','double');

            ncwrite(outfile, 'time', tsecs);
            ncwriteatt(outfile, 'time', 'units', 'seconds since 1900-01-01 00:00:00');
            ncwriteatt(outfile, 'time', 'calendar', 'gregorian');
        end

        ncwriteatt(outfile, '/', 'description', 'Merged model data within interval (spatially aggregated)');
        ncwriteatt(outfile, '/', 'created_by', 'Karin Pitlik');
        ncwriteatt(outfile, '/', 'creation_date', char(datetime('now','Format','yyyy-MM-dd_HH:mm:ss')));
        ncwriteatt(outfile, '/', 'spatial_op', spatial_op);
        ncwriteatt(outfile, '/', 'ensemble_id', ens_id);

        fprintf('Saved %s\n', outfile);
    end
end

function out2d = temporal_agg(cube3d, op)
    % cube3d: [lat x lon x time]
    switch lower(op)
        case 'sum'
            out2d = nansum(cube3d, 3);
        case 'mean'
            out2d = mean(cube3d, 3, 'omitnan');
        otherwise
            error('temporal_op must be ''sum'' or ''mean''.');
    end
end

function out = spatial_block_agg(tmp5d, op)
    % tmp5d shape: [df x nlat x df x nlon x time]
    switch lower(op)
        case 'sum'
            out = squeeze(nansum(nansum(tmp5d,1),3));
        case 'mean'
            out = squeeze(mean(mean(tmp5d,1,'omitnan'),3,'omitnan'));
        otherwise
            error('spatial_op must be ''sum'' or ''mean''.');
    end
end

function process_ildn_data(ildn_file_path, output_path, start_date, end_date, ...
                           min_lat, max_lat, min_lon, max_lon, ...
                           bin_width_lat, bin_width_lon, ...
                           lat_grid_coarse, lon_grid_coarse)

    mkif(output_path);
    S = load(ildn_file_path);
    fname = fieldnames(S);
    T = S.(fname{1});

    % Geographic and temporal filtering
    geo = T.lon >= min_lon & T.lon <= max_lon & T.lat >= min_lat & T.lat <= max_lat;
    T = T(geo,:);
    T.UTC = datetime(T.UTC,'InputFormat','yyyy-MMM-dd HH:mm:ss.SS');
    T = T(T.UTC >= start_date & T.UTC < end_date,:);

    % 2D histogram
    [N, Xedges, Yedges] = histcounts2(T.lon, T.lat, 'BinWidth',[bin_width_lon bin_width_lat]);

    % Map histogram to nearest model grid center
    Xc = Xedges(1:end-1) + diff(Xedges)/2;
    Yc = Yedges(1:end-1) + diff(Yedges)/2;
    [Ygrid, Xgrid] = meshgrid(Yc, Xc);
    grid_points = [lon_grid_coarse(:), lat_grid_coarse(:)];
    idx = knnsearch(grid_points, [Xgrid(:), Ygrid(:)]);
    [row,col] = ind2sub(size(lat_grid_coarse), idx);

    N_adjusted = accumarray([row,col], N(:), size(lat_grid_coarse));

    outfile = fullfile(output_path, sprintf('%s_%s.nc', ...
        datestr(start_date,'yyyy-mm-dd_HH_MM_SS'), datestr(end_date,'yyyy-mm-dd_HH_MM_SS')));

    if exist(outfile,'file')
        delete(outfile);
    end

    [num_lat, num_lon] = size(N_adjusted);
    nccreate(outfile,'ildn','Dimensions',{'lat',num_lat,'lon',num_lon},'Datatype','double');
    nccreate(outfile,'closest_lats','Dimensions',{'lat',num_lat,'lon',num_lon},'Datatype','double');
    nccreate(outfile,'closest_longs','Dimensions',{'lat',num_lat,'lon',num_lon},'Datatype','double');

    ncwrite(outfile,'ildn',N_adjusted);
    ncwrite(outfile,'closest_lats',lat_grid_coarse);
    ncwrite(outfile,'closest_longs',lon_grid_coarse);

    ncwriteatt(outfile,'/','description','ILDN histogram mapped to model grid');
    ncwriteatt(outfile,'/','created_by','Karin Pitlik');
    ncwriteatt(outfile,'/','creation_date', char(datetime('now','Format','yyyy-MM-dd_HH:mm:ss')));
end

function process_entln_pulse(entln_file_path, output_path, start_date, end_date, ...
                             min_lat, max_lat, min_lon, max_lon, ...
                             bin_width_lat, bin_width_lon, ...
                             lat_grid_coarse, lon_grid_coarse)

    mkif(output_path);
    S = load(entln_file_path);
    fname = fieldnames(S);
    T = S.(fname{1});

    % Geographic and temporal filtering
    geo = T.lon >= min_lon & T.lon <= max_lon & T.lat >= min_lat & T.lat <= max_lat;
    T = T(geo,:);
    T.UTC = datetime(T.UTC,'InputFormat','dd-MMM-yyyy HH:mm:ss');
    T = T(T.UTC >= start_date & T.UTC < end_date,:);

    % 2D histogram
    [N, Xedges, Yedges] = histcounts2(T.lon, T.lat, 'BinWidth',[bin_width_lon bin_width_lat]);

    % Map histogram to nearest model grid center
    Xc = Xedges(1:end-1) + diff(Xedges)/2;
    Yc = Yedges(1:end-1) + diff(Yedges)/2;
    [Ygrid, Xgrid] = meshgrid(Yc, Xc);
    grid_points = [lon_grid_coarse(:), lat_grid_coarse(:)];
    idx = knnsearch(grid_points, [Xgrid(:), Ygrid(:)]);
    [row,col] = ind2sub(size(lat_grid_coarse), idx);

    N_adjusted = accumarray([row,col], N(:), size(lat_grid_coarse));

    outfile = fullfile(output_path, sprintf('%s_%s.nc', ...
        datestr(start_date,'yyyy-mm-dd_HH_MM_SS'), datestr(end_date,'yyyy-mm-dd_HH_MM_SS')));

    if exist(outfile,'file')
        delete(outfile);
    end

    [num_lat, num_lon] = size(N_adjusted);
    nccreate(outfile,'ildn','Dimensions',{'lat',num_lat,'lon',num_lon},'Datatype','double');
    nccreate(outfile,'closest_lats','Dimensions',{'lat',num_lat,'lon',num_lon},'Datatype','double');
    nccreate(outfile,'closest_longs','Dimensions',{'lat',num_lat,'lon',num_lon},'Datatype','double');

    ncwrite(outfile,'ildn',N_adjusted);
    ncwrite(outfile,'closest_lats',lat_grid_coarse);
    ncwrite(outfile,'closest_longs',lon_grid_coarse);

    ncwriteatt(outfile,'/','description','ENTLN pulse histogram mapped to model grid');
    ncwriteatt(outfile,'/','created_by','Karin Pitlik');
    ncwriteatt(outfile,'/','creation_date', char(datetime('now','Format','yyyy-MM-dd_HH:mm:ss')));
end

function gf = gaussian2d(sz, sigma)
    [x,y] = meshgrid(linspace(-2,2,sz), linspace(-2,2,sz));
    gf = exp(-(x.^2 + y.^2)/(2*sigma^2));
    gf = gf / sum(gf(:));
end

function [factor, lbl] = factor_by_res(res_km)
    switch res_km
        case 4
            factor = 1; lbl = '4by4';
        case 12
            factor = 3; lbl = '12by12';
        case 24
            factor = 6; lbl = '24by24';
        case 40
            factor = 10; lbl = '40by40';
        case 80
            factor = 20; lbl = '80by80';
        otherwise
            error('Unsupported resolution_km: %d', res_km);
    end
end

function s = grid_lbl(res_km)
    switch res_km
        case 4
            s = '4by4';
        case 12
            s = '12by12';
        case 24
            s = '24by24';
        case 40
            s = '40by40';
        case 80
            s = '80by80';
        otherwise
            error('Unsupported resolution');
    end
end

function mkif(pth)
    if ~exist(pth,'dir')
        mkdir(pth);
    end
end

function [start_date, end_date] = parse_case_dates(case_name)
    % Format: CaseN_Mon_YYYY_DD1_DD2
    % e.g. Case5_Jan_2024_26_31 -> start: 2024-01-26, end: 2024-02-01
    parts = strsplit(case_name, '_');
    month_str = parts{2};
    year      = str2double(parts{3});
    day_start = str2double(parts{4});
    day_end   = str2double(parts{5});

    month_names = {'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'};
    month_num = find(strcmpi(month_names, month_str));

    start_date = datetime(year, month_num, day_start, 0, 0, 0);
    end_date   = datetime(year, month_num, day_end,   0, 0, 0) + days(1);
end