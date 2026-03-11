function run_nordic(subject_id, run, base_path, session, task)
% run_nordic  Denoise a single bold run with NIFTI_NORDIC.
%
%   run_nordic(subject_id, run)
%   run_nordic(subject_id, run, base_path)
%   run_nordic(subject_id, run, base_path, session)
%   run_nordic(subject_id, run, base_path, session, task)
%
%   Default base_path : /data/ds-valuecapture
%   Default session   : [] (no session sub-directory)
%   Default task      : valuecapture

if nargin < 3 || isempty(base_path)
    base_path = '/data/ds-valuecapture';
end

subject = sprintf('sub-%s', subject_id);

if nargin < 4 || isempty(session)
    session_str  = '';
    session_path = '';
else
    session_str  = sprintf('ses-%d_', session);
    session_path = sprintf('ses-%d/', session);
end

if nargin < 5 || isempty(task)
    task = 'valuecapture';
end

fn_magn_in = fullfile(base_path, subject, session_path, 'func', ...
    sprintf('%s_%stask-%s_run-%d_part-mag_bold.nii.gz', subject, session_str, task, run));
fn_phase_in = fullfile(base_path, subject, session_path, 'func', ...
    sprintf('%s_%stask-%s_run-%d_part-phase_bold.nii.gz', subject, session_str, task, run));

output_dir = fullfile(base_path, 'derivatives', 'nordic', subject, session_path);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fn_out      = sprintf('%s_%stask-%s_run-%d_rec-NORDIC_bold', subject_id, session_str, task, run);
fn_out_full = fullfile(output_dir, [fn_out, '.nii']);

disp(['fn_magn_in:  ', fn_magn_in]);
disp(['fn_phase_in: ', fn_phase_in]);
disp(['fn_out_full: ', fn_out_full]);

ARG.temporal_phase   = 1;
ARG.phase_filter_width = 10;
ARG.DIROUT           = output_dir;

if exist(fn_out_full, 'file')
    disp(['Deleting existing output file: ', fn_out_full]);
    delete(fn_out_full);
end

disp(['Running NIFTI_NORDIC with output: ', fn_out]);
NIFTI_NORDIC(fn_magn_in, fn_phase_in, fn_out, ARG);

if ~exist(fn_out_full, 'file')
    error(['NIFTI_NORDIC did not create the output file: ', fn_out_full]);
end

% Gzip and move to BIDS func directory
final_output = fullfile(base_path, subject, session_path, 'func', ...
    sprintf('%s_%stask-%s_run-%d_rec-NORDIC_bold.nii.gz', subject, session_str, task, run));

gzip(fn_out_full);
movefile([fn_out_full, '.gz'], final_output);
delete(fn_out_full);

disp(['Compressed file created: ', final_output]);
end
