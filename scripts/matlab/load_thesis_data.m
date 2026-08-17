function T = load_thesis_data(name, dataDir)
%LOAD_THESIS_DATA  Read one main-body figure's data table.
%
%   T = LOAD_THESIS_DATA(NAME) reads results/thesis/<NAME>.csv, written by
%   scripts/make_thesis_figures.py, and returns it as a MATLAB table.
%
%   T = LOAD_THESIS_DATA(NAME, DATADIR) reads from DATADIR instead.
%
%   The Python side owns the extraction: it reads the sweep archives, resolves
%   the unit conventions (three error columns coexist on disk and two of them
%   are not on the same scale), excludes rows whose provenance makes them
%   incomparable, and writes one tidy table per figure with the unit stated in
%   every column name. This function does no arithmetic whatsoever. A figure
%   rendered here therefore plots the same numbers as the Python reference plot
%   beside the CSV, and any disagreement between the two is a plotting error
%   rather than a difference of interpretation.
%
%   Empty cells arrive as NaN for numeric columns and as an empty string for
%   text columns, which is what a missing measurement should look like on a
%   logarithmic axis: absent, not zero.
%
%   Inputs
%   ------
%   name : char or string
%       Table stem, with or without the .csv suffix, e.g. 'F1_accuracy_vs_N_1D'.
%   dataDir : char or string, optional
%       Directory holding the tables. Defaults to ../../results/thesis relative
%       to this file.
%
%   Outputs
%   -------
%   T : table
%       The data table.
%
%   Example
%   -------
%       T = load_thesis_data('F4_kappa_scaling');
%       oneD = T(T.dim == 1 & T.order == 2, :);
%       loglog(oneD.N, oneD.kappa, 'o-');
%
%   See also THESIS_STYLE, READTABLE.

    arguments
        name     (1,:) char
        dataDir  (1,:) char = ''
    end

    if isempty(dataDir)
        here    = fileparts(mfilename('fullpath'));
        dataDir = fullfile(here, '..', '..', 'results', 'thesis');
    end

    [~, stem, ext] = fileparts(name);
    if isempty(ext)
        stem = name;
    end
    path = fullfile(dataDir, [stem '.csv']);

    if ~isfile(path)
        error('load_thesis_data:missing', ...
              ['No table at %s.\n' ...
               'Generate the tables first:\n' ...
               '    python scripts/make_thesis_figures.py'], path);
    end

    opts = detectImportOptions(path, 'TextType', 'string');
    opts = setvartype(opts, opts.VariableNames( ...
        strcmp(opts.VariableTypes, 'char')), 'string');
    T = readtable(path, opts);
end
