%% ===========================
%  Lightning Pipeline (ALL-IN-ONE)
%  - Build coarse grid
%  - Process MODEL (LPI/KI), ILDN, ENTLN into interval NetCDFs
%  - Keep ensemble members separated in MODEL outputs
%  - Plot triptychs per interval and save .fig
%  Karin Pitlik
%  ===========================

%% -------- Configure once --------
case_name   = 'Case6_Nov_2025_24_25';

% What to process
variable_type = 'wmax_layer';   
variable_type_upper  = upper(variable_type);
variable_type_lower  = lower(variable_type);

if strcmpi(variable_type, 'KI')
    variable_type_lower = 'KI';
    variable_type_nc = variable_type_lower;
elseif strcmpi(variable_type, 'ds')
    variable_type_lower = 'ds';
else
    variable_type_nc = variable_type_lower;
end

% Aggregation controls
% spatial_op: operation for spatial downscaling per time slice
% temporal_op: operation for time aggregation during plotting
switch variable_type_lower
    case 'lpi' 
        spatial_op  = 'mean';  temporal_op = 'mean';  
    case 'KI' 
        spatial_op  = 'mean';  temporal_op = 'mean';   
    case 'ds' 
        spatial_op  = 'mean';  temporal_op = 'sum';   
    case 'wmax_layer' 
        spatial_op  = 'mean';  temporal_op = 'mean';   
    case 'flux_up' 
        spatial_op  = 'mean';  temporal_op = 'mean';   
    case 'prec_rate' 
        spatial_op  = 'mean';  temporal_op = 'sum';   
    case 'cape2d' 
        spatial_op  = 'mean';  temporal_op = 'mean';   
    otherwise
        error('Unsupported variable type: %s', variable_type_lower);
end

%% Interval config
interval_hours = 1;
interval_name  = sprintf('%d_hours', interval_hours);
time_step      = hours(interval_hours);

% Date range
loop_start_date = datetime('2025-11-24_00:00:00','InputFormat','yyyy-MM-dd_HH:mm:ss');
loop_end_date   = datetime('2025-11-26_00:00:00','InputFormat','yyyy-MM-dd_HH:mm:ss');

% Geographic filter
min_lat = 27.296;  max_lat = 36.598;
min_lon = 27.954;  max_lon = 39.292;

% Target grid resolution in km
resolution_km = 24;   % allowed values: 4, 12, 24
relevant_lat  = 32;   

% Derived bin widths in degrees
bin_width_lat = resolution_km / 111.32;
bin_width_lon = resolution_km / (111.32 * cosd(relevant_lat));

% Paths
model_raw_folder = sprintf('/Users/karinpitlik/Desktop/DataScience/Thesis/%s/Ens/Raw/%s/', ...
                           case_name, variable_type_upper);

coords_folder_path = '/Users/karinpitlik/Desktop/DataScience/Thesis/NetCDF/Other/lpi_4km_output_2022-01_24_27/';
coords_file_path   = fullfile(coords_folder_path, 'lpi_4km_output_2022-01-24_00_10_00.nc');

ildn_mat_path  = sprintf('/Users/karinpitlik/Desktop/DataScience/Thesis/ILDN/Cases_Mats/ILDN_%s.mat', case_name);
entln_mat_path = sprintf('/Users/karinpitlik/Desktop/DataScience/Thesis/ENTLN/Pulse_Cases_Mats/ENTLN_pulse_%s.mat', case_name);

model_out_root = sprintf('%s/proccesed/%s/%s/', model_raw_folder, grid_lbl(resolution_km), interval_name);
ildn_out_root  = sprintf('/Users/karinpitlik/Desktop/DataScience/Thesis/ILDN/ILDN_%s/%s/%s/', case_name, grid_lbl(resolution_km), interval_name);
entln_out_root = sprintf('/Users/karinpitlik/Desktop/DataScience/Thesis/ENTLN/ENTLN_pulse_%s/%s/%s/', case_name, grid_lbl(resolution_km), interval_name);

fig_output_dir = sprintf('/Users/karinpitlik/Desktop/DataScience/Thesis/%s/Ens/Graphs/%s/%s/%s/', ...
                         case_name, variable_type_upper, grid_lbl(resolution_km), interval_name);

mkif(model_out_root); 
mkif(ildn_out_root); 
mkif(entln_out_root); 
mkif(fig_output_dir);

% Ensemble member to plot in Pass 2
ens_to_plot = '00';

%% -------- Prepare coarse grid from reference coordinates --------
lat_grid = ncread(coords_file_path, 'xlat');
lon_grid = ncread(coords_file_path, 'xlong');
[m,n]    = size(lat_grid);

[downgrade_factor, ~] = factor_by_res(resolution_km);

if downgrade_factor == 1
    lat_grid_coarse = lat_grid;
    lon_grid_coarse = lon_grid;
else
    m_trim = floor(m / downgrade_factor) * downgrade_factor;
    n_trim = floor(n / downgrade_factor) * downgrade_factor;

    lat_trim = lat_grid(1:m_trim, 1:n_trim);
    lon_trim = lon_grid(1:m_trim, 1:n_trim);

    lat_blocks = reshape(lat_trim, ...
        downgrade_factor, m_trim/downgrade_factor, ...
        downgrade_factor, n_trim/downgrade_factor);

    lon_blocks = reshape(lon_trim, ...
        downgrade_factor, m_trim/downgrade_factor, ...
        downgrade_factor, n_trim/downgrade_factor);

    % Coarse grid is defined by block means
    lat_grid_coarse = squeeze(mean(mean(lat_blocks,1),3));
    lon_grid_coarse = squeeze(mean(mean(lon_blocks,1),3));
end

%% -------- Build time edges --------
time_edges = datetime(loop_start_date:time_step:loop_end_date, 'Format', 'yyyy-MM-dd HH:mm:ss');

%% -------- Pass 1: WRITE interval NetCDFs --------
for t = 1:(length(time_edges)-1)
    sdate = time_edges(t);
    edate = time_edges(t+1);

    % MODEL
    process_model_files(variable_type_lower, model_raw_folder, model_out_root, ...
                        coords_file_path, sdate, edate, downgrade_factor, ...
                        lat_grid_coarse, lon_grid_coarse, spatial_op);

    % ENTLN
    % process_entln_pulse(entln_mat_path, entln_out_root, ...
    %                     sdate, edate, min_lat, max_lat, min_lon, max_lon, ...
    %                     bin_width_lat, bin_width_lon, lat_grid_coarse, lon_grid_coarse);
end


%% ===========================
%           FUNCTIONS
% ===========================

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
        % Some variables such as wmax_layer may not include ensemble id in the filename
        ens_match = regexp(fname, '_(\d{1,2})_', 'tokens');
        
        if isempty(ens_match)
            if strcmpi(varname, 'wmax_layer')
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
                    % For dynamic schemes, sum components
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
        otherwise
            error('Unsupported resolution');
    end
end

function mkif(pth)
    if ~exist(pth,'dir')
        mkdir(pth);
    end
end