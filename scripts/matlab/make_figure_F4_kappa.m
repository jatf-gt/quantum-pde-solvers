% MAKE_FIGURE_F4_KAPPA  Condition-number scaling, 1-D against the 2-D/3-D strips.
%
% Worked template for rendering a main-body figure in MATLAB from the tables
% written by scripts/make_thesis_figures.py. Figure F4 is used as the example
% because it is the simplest of the set; every other figure follows the same
% four steps.
%
%   1. Read the table. No arithmetic is done here — see LOAD_THESIS_DATA.
%   2. Select each series by its identifying columns.
%   3. Plot, with the axes, units and reference lines the Python reference plot
%      uses, so the two renderings can be compared directly.
%   4. Apply THESIS_STYLE and export at the final width.
%
% The corresponding Python reference plot is results/thesis/F4_kappa_scaling.png.
% Compare the two before adopting the MATLAB version: they must agree point for
% point, and a visible difference means a selection step here is wrong.
%
% Physical content
% ----------------
% The one-dimensional Poisson operator has kappa = O(N^2). Every quantum
% linear-system algorithm carries kappa in its query complexity — QSVT carries it
% in the polynomial degree specifically — so that scaling is what makes a direct
% quantum solve of the whole system intractable at useful resolutions.
% Decomposing the 2-D and 3-D problems into one-dimensional strips replaces that
% operator with a strip operator whose transverse coupling contributes a diagonal
% shift, bounding kappa_row by 3 in two dimensions and by 2 in three,
% independently of N. That bound is the architectural claim of this work.

clear; close all;

T = load_thesis_data('F4_kappa_scaling');

% -- Presentation constants ----------------------------------------------------
% Shared with the Python layer so that a dimension keeps one colour across every
% figure in the thesis.
colours = containers.Map({1, 2, 3}, ...
                         {[0.12 0.47 0.71], [0.17 0.63 0.17], [0.84 0.15 0.16]});
markers = containers.Map({1, 2, 3}, {'o', 's', '^'});
lineFor = containers.Map({2, 4}, {'-', '--'});

fig = figure('Units', 'centimeters', 'Position', [2 2 14 10], 'Color', 'w');
ax  = axes(fig); hold(ax, 'on');

for dim = [1 2 3]
    for order = [2 4]
        sel = T(T.dim == dim & T.order == order, :);
        if isempty(sel), continue; end
        [N, idx] = sort(sel.N);
        kappa    = sel.kappa(idx);

        if dim == 1
            symbol = '\kappa';
        else
            symbol = '\kappa_{\mathrm{row}}';
        end

        loglog(ax, N, kappa, ...
               'LineStyle',       lineFor(order), ...
               'Marker',          markers(dim), ...
               'Color',           colours(dim), ...
               'MarkerFaceColor', 'none', ...
               'LineWidth',       1.4, ...
               'MarkerSize',      6, ...
               'DisplayName', sprintf('$%d$-D, order %d ($%s$)', ...
                                      dim, order, symbol));
    end
end

% -- Reference lines -----------------------------------------------------------
% The O(N^2) guide is anchored, not fitted: it indicates the slope the
% one-dimensional operator should follow and must never be read as a fit.
Nref = [4 8 16 32 64];
loglog(ax, Nref, 0.6 * Nref.^2, 'k:', 'LineWidth', 1.0, ...
       'DisplayName', '$\mathcal{O}(N^{2})$');

yline(ax, 3, '-.', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.0, ...
      'HandleVisibility', 'off');
yline(ax, 2, '-.', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.0, ...
      'HandleVisibility', 'off');
text(ax, 40, 3.4, '$\kappa_{\mathrm{row}} \to 3$ (2-D)', ...
     'Interpreter', 'latex', 'FontSize', 8, 'Color', [0.4 0.4 0.4]);
text(ax, 40, 1.72, '$\kappa_{\mathrm{row}} \to 2$ (3-D)', ...
     'Interpreter', 'latex', 'FontSize', 8, 'Color', [0.4 0.4 0.4]);

set(ax, 'XScale', 'log', 'YScale', 'log');
xlabel(ax, '$N$');
ylabel(ax, 'condition number');
xticks(ax, [4 8 16 32 64 128 256]);
xlim(ax, [3.5 300]);
legend(ax, 'Location', 'northwest', 'NumColumns', 2);

thesis_style(ax);

% -- Export --------------------------------------------------------------------
% Exported at the width it is finally placed at. Scaling a figure in LaTeX
% rescales its text with it and undoes the font matching thesis_style applies.
outDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', ...
                  'results', 'thesis', 'matlab');
if ~isfolder(outDir), mkdir(outDir); end
exportgraphics(fig, fullfile(outDir, 'F4_kappa_scaling.pdf'), ...
               'ContentType', 'vector', 'BackgroundColor', 'none');
exportgraphics(fig, fullfile(outDir, 'F4_kappa_scaling.png'), ...
               'Resolution', 400);

fprintf('Wrote F4_kappa_scaling.{pdf,png} to %s\n', outDir);
